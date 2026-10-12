"""Interactive console to add a trained bot to an existing Curvytron room.

The bot plays the games on its own; from the console you do everything a player
can do on curvytron.com: chat, get ready, vote to kick, change name and colour,
spectate, and (as room master) change the room settings and launch the game.

    uv run python play_room.py
    uv run python play_room.py --model runs/big/model_best.pt \\
        --room "https://www.curvytron.com/#/room/My%20Room" --name rlbot --color "#00ffcc"

Anything not given on the command line is asked interactively. Type /help in
the console for the commands; plain text is sent to the chat.
"""
import argparse
import glob
import json
import os
import random
import re
import threading
import time
from urllib.parse import urlparse, unquote, parse_qs

import torch
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.patch_stdout import patch_stdout

from curvybot import CurvyBot
from curvysim.core import BONUS_NAMES
from curvysim.policy import load_policy

COLOR_RE = re.compile(r'^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$')
NAME_MAX = 25

HELP = """\
commands (anything not starting with / is sent to the chat):
  /help                      this help
  /players                   players in the room (ready state, scores, room master)
  /ready                     toggle the bot's ready state
  /auto on|off               get ready automatically (on joining and after each game), once another
                             player is in the room (default off, or --auto-ready)
  /kick <player>             vote to kick a player (again to withdraw); the room master kicks directly
  /votes                     ongoing kick votes
  /name <new name>           change the bot's nickname
  /color <#rrggbb>           change the bot's colour
  /spectate                  remove the bot from the players and just watch
  /play                      add the bot back as a player
  /greedy on|off             bot picks its most likely move (on) or samples (off, default)
  /aggression [x]            hunting vs surviving: kill bonus the bot plays for (0 = only survive)
  /status                    connection, latency, game state and the bot's record
  /quit                      leave the room and exit
room master only:
  /launch                    start (or cancel) the game countdown
  /maxscore <n>              score to win the game (0 = default: (players - 1) x 10)
  /bonuses                   list the bonuses and whether they are enabled
  /bonus <name>              toggle a bonus (e.g. /bonus BonusSelfFast)
  /bonusrate <-1..1>         how often bonuses appear (0 = default)
  /open on|off               open the room, or make it private (password shown)"""


def parse_room(text):
    """'https://www.curvytron.com/#/room/My%20Room?password=1234' -> (server, room, password).
    A bare room name means curvytron.com."""
    text = text.strip()
    if '/room/' not in text:
        return 'curvytron.com', text, None
    url = urlparse(text if '://' in text else 'https://' + text)
    host = url.netloc or 'curvytron.com'
    frag = url.fragment or url.path              # '#/room/<name>?password=...'
    path, _, query = frag.partition('?')
    room = path.split('/room/', 1)[1].strip('/')
    for _ in range(3):   # the browser shows names encoded twice (space -> %2520)
        decoded = unquote(room)
        if decoded == room:
            break
        room = decoded
    password = parse_qs(query or url.query).get('password', [None])[0]
    server = 'curvytron.com' if 'curvytron.com' in host else host
    return server, room, password


def find_models():
    paths = glob.glob('runs/*/model_*.pt') + glob.glob('runs/*/pool/*.pt')
    return sorted(paths, key=os.path.getmtime, reverse=True)


def ask(prompt, default=None, check=None):
    while True:
        suffix = f' [{default}]' if default else ''
        value = input(f'{prompt}{suffix}: ').strip() or (default or '')
        error = check(value) if check else None
        if not error:
            return value
        print(f'  {error}')


def check_color(value, lobby=True):
    m = COLOR_RE.match(value)
    if not m:
        return 'use the #rrggbb format, e.g. #00ffcc'
    r, g, b = (int(x, 16) for x in m.groups())
    if lobby and (r * 0.4 + g * 0.5 + b * 0.3) / 255 <= 0.3:
        return 'too dark: the server rejects this colour'
    return None


def check_name(value):
    if not value:
        return 'the name cannot be empty'
    if len(value) > NAME_MAX:
        return f'at most {NAME_MAX} characters'
    return None


class RoomBot(CurvyBot):
    """CurvyBot plus the room protocol: chat, players, ready, kick votes, room settings."""

    def __init__(self, model, room, password, **kwargs):
        super().__init__(model, **kwargs)
        self.room = room
        self.password = password
        self.players = {}        # player id -> dict(name, color, ready, client)
        self.scores = {}         # player id -> score in the current game
        self.master = None       # client id of the room master
        self.config = {}
        self.votes = {}          # target player id -> result
        self.in_game = False
        self.auto_ready = False
        self.want_play = True
        self.request_lock = threading.Lock()

    # -- requests ------------------------------------------------------------
    def request(self, name, data=None, timeout=5.0):
        """Send an event with a callback id and wait for the server's answer."""
        with self.request_lock:
            self.message_id += 1
            mid = self.message_id
        msg = [name, data, mid] if data is not None else [name, None, mid]
        self.ws.send(json.dumps([msg]))
        end = time.time() + timeout
        while mid not in self.message_responses:
            if time.time() > end:
                return {'success': False, 'error': 'no answer from the server'}
            time.sleep(0.01)
        return self.message_responses.pop(mid)

    def send(self, name, data=None):
        self.ws.send(json.dumps([[name, data]] if data is not None else [[name]]))

    def later(self, fn, *args):
        """Run a blocking request from an event handler without stalling the receive thread."""
        threading.Thread(target=fn, args=args, daemon=True).start()

    # -- room ----------------------------------------------------------------
    def enter_room(self):
        res = self.request('room:join', {'name': self.room, 'password': self.password})
        if not res.get('success'):
            return res.get('error', 'could not join')
        room = res['room']
        self.master = res.get('master')
        self.config = room.get('config', {})
        self.in_game = bool(room.get('game'))
        for p in room.get('players', []):
            self.players[p['id']] = dict(p)
        self._add_players(room.get('players', []))
        for v in res.get('votes', []):
            self.votes[v['target']] = v['result']
        for m in res.get('messages', [])[-5:]:
            self.print_chat(m)
        return None

    def add_player(self):
        if self.player_id is not None:
            return 'the bot is already playing'
        if self.in_game:
            self.want_play = True
            return 'a game is running: the bot will join when it ends'
        res = self.request('player:add', {'name': self.name, 'color': self.color})
        if not res.get('success'):
            return res.get('error', 'could not add the bot')
        self.want_play = True
        t = time.time() + 3
        while self.player_id is None and time.time() < t:
            time.sleep(0.05)
        self.maybe_auto_ready()
        return None

    def maybe_auto_ready(self):
        """Get ready if auto-ready is on and someone else is playing: alone, the bot would be the
        only (ready) player and the server would start solo games, locking others out."""
        if self.auto_ready and self.player_id is not None and not self.in_game and len(self.players) >= 2:
            err = self.set_ready(True)
            if err:
                print(f'[bot] {err}')

    def remove_player(self):
        if self.player_id is None:
            return 'the bot is not playing'
        self.want_play = False
        res = self.request('player:remove', {'player': self.player_id})
        return None if res.get('success') else 'could not remove the bot'

    def set_ready(self, ready=None):
        if self.player_id is None:
            return 'the bot is not playing'
        me = self.players.get(self.player_id, {})
        if ready is not None and me.get('ready') == ready:
            return None
        res = self.request('room:ready', {'player': self.player_id})
        return None if res.get('success') else res.get('error', 'could not change ready state')

    def is_master(self):
        return self.master is not None and self.master == self.client_id

    def find_player(self, text):
        text = text.strip().lstrip('#')
        if text.isdigit() and int(text) in self.players:
            return int(text)
        exact = [pid for pid, p in self.players.items() if p['name'].lower() == text.lower()]
        if exact:
            return exact[0]
        prefix = [pid for pid, p in self.players.items() if p['name'].lower().startswith(text.lower())]
        return prefix[0] if len(prefix) == 1 else None

    def pname(self, pid):
        p = self.players.get(pid)
        return p['name'] if p else f'player {pid}'

    def print_chat(self, m):
        name = m.get('name') or f"spectator {m.get('client')}"
        print(f"[chat] {name}: {m.get('content', '')}")

    # -- server events -------------------------------------------------------
    def _process_recvd(self, recvd):
        for message in json.loads(recvd):
            head = message[0]
            body = message[1] if len(message) > 1 else None
            if isinstance(head, int):
                continue
            try:
                self.on_room_event(head, body)
            except Exception as exc:   # never let a display problem kill the receive thread
                print(f'[warning] could not handle {head}: {exc!r}')
        super()._process_recvd(recvd)

    def on_room_event(self, head, body):
        if head == 'room:join':
            p = body['player']
            self.players[p['id']] = dict(p)
            mine = p['name'] == self.name and p['client'] == self.client_id
            print(f"[room] {'the bot' if mine else p['name']} joined the players")
            if not mine and self.player_id is not None:
                self.later(self.maybe_auto_ready)
        elif head == 'room:leave':
            pid = body['player']
            print(f"[room] {self.pname(pid)} left")
            self.players.pop(pid, None)
            self.scores.pop(pid, None)
            if pid == self.player_id:
                self.player_id = None
            self.game.players.setdefault(pid, None)   # the base client pops it and fails on unknown ids
        elif head == 'player:ready':
            if body['player'] in self.players:
                self.players[body['player']]['ready'] = body['ready']
            print(f"[room] {self.pname(body['player'])} is {'ready' if body['ready'] else 'not ready'}")
        elif head == 'player:name':
            old = self.pname(body['player'])
            if body['player'] in self.players:
                self.players[body['player']]['name'] = body['name']
            print(f"[room] {old} is now called {body['name']}")
        elif head == 'player:color':
            if body['player'] in self.players:
                self.players[body['player']]['color'] = body['color']
        elif head == 'room:talk':
            self.print_chat(body)
        elif head == 'room:master':
            self.master = body['client']
            print('[room] the bot is now the room master' if self.is_master() else '[room] new room master')
        elif head == 'vote:new':
            self.votes[body['target']] = body['result']
            print(f"[vote] kick vote against {self.pname(body['target'])} started")
        elif head == 'vote:close':
            self.votes.pop(body['target'], None)
            print(f"[vote] kick vote against {self.pname(body['target'])} closed: "
                  f"{'kicked' if body['result'] else 'not kicked'}")
        elif head == 'room:kick':
            print(f"[room] {self.pname(body)} was kicked"
                  + (' (the bot!)' if body == self.player_id else ''))
            if body == self.player_id:
                self.want_play = False
        elif head == 'room:launch:start':
            print('[room] game starting in 5 seconds...')
        elif head == 'room:launch:cancel':
            print('[room] game start cancelled')
        elif head.startswith('room:config:') and 'name' not in body:   # 'name': lobby-wide copy of the event
            key = head.split(':', 2)[2]
            if key == 'bonus':
                self.config.setdefault('bonuses', {})[body['bonus']] = body['enabled']
                print(f"[config] {body['bonus']} {'enabled' if body['enabled'] else 'disabled'}")
            elif key == 'max-score':
                self.config['maxScore'] = body['maxScore']
                print(f"[config] max score: {body['maxScore'] or 'default'}")
            elif key == 'variable':
                self.config.setdefault('variables', {})[body['variable']] = body['value']
                print(f"[config] {body['variable']} = {body['value']}")
            elif key == 'open':
                self.config['open'] = body['open']
                self.config['password'] = body.get('password')
                print(f"[config] room is {'open' if body['open'] else 'private, password ' + str(body.get('password'))}")
        elif head == 'room:game:start':
            self.in_game = True
            self.scores = {pid: 0 for pid in self.players}
            self.votes.clear()   # the server drops kick votes when a game starts (no vote:close)
            print('[game] game started' + ('' if self.player_id is not None else ' (the bot is spectating)'))
        elif head == 'score':
            pid, score = body
            self.scores[pid] = score
        elif head == 'end':
            self.in_game = False
            ranking = sorted(self.scores.items(), key=lambda kv: -kv[1])
            print('[game] game over: ' + ', '.join(f'{self.pname(pid)} {s}' for pid, s in ranking))
            for p in self.players.values():
                p['ready'] = False
            if self.player_id is None and self.want_play:
                self.later(self._rejoin)
            elif self.player_id is not None and self.auto_ready:
                self.later(self._ready_after_game)
        elif head == 'room:close':
            print('[room] the room was closed')

    def _rejoin(self):
        time.sleep(1.0)
        err = self.add_player()
        print(f'[bot] {err}' if err else '[bot] joined the players')

    def _ready_after_game(self):
        time.sleep(1.5)
        self.maybe_auto_ready()


class Console:
    def __init__(self, bot):
        self.bot = bot

    def run(self):
        names = lambda: [p['name'] for p in self.bot.players.values()]
        commands = ['/help', '/players', '/ready', '/auto', '/kick', '/votes', '/name', '/color', '/spectate',
                    '/play', '/greedy', '/aggression', '/status', '/quit', '/launch', '/maxscore', '/bonuses', '/bonus',
                    '/bonusrate', '/open']
        session = PromptSession()
        print('type /help for the commands; plain text goes to the chat')
        with patch_stdout(raw=True):
            while True:
                completer = WordCompleter(commands + names() + list(BONUS_NAMES), sentence=True)
                try:
                    line = session.prompt('> ', completer=completer).strip()
                except (EOFError, KeyboardInterrupt):
                    line = '/quit'
                if not line:
                    continue
                if line == '/quit':
                    return
                try:
                    out = self.handle(line)
                except Exception as exc:
                    out = f'error: {exc!r}'
                if out:
                    print(out)

    def handle(self, line):
        bot = self.bot
        if not line.startswith('/'):
            res = bot.request('room:talk', line[:140])
            return None if res.get('success') else 'message not sent (flood filter?)'
        cmd, _, arg = line.partition(' ')
        arg = arg.strip()
        if cmd == '/help':
            return HELP
        if cmd == '/players':
            if not bot.players:
                return 'no players'
            rows = []
            for pid, p in sorted(bot.players.items()):
                tags = []
                if pid == bot.player_id:
                    tags.append('bot')
                if p.get('client') == bot.master:
                    tags.append('room master')
                if pid in bot.votes:
                    tags.append('kick vote open')
                score = f" score {bot.scores.get(pid, 0)}" if bot.in_game else ''
                ready = '' if bot.in_game else (' ready' if p.get('ready') else ' not ready')
                rows.append(f"  #{pid} {p['name']} {p.get('color', '')}{ready}{score}"
                            + (f"  ({', '.join(tags)})" if tags else ''))
            return '\n'.join(rows)
        if cmd == '/ready':
            if bot.in_game:
                return 'a game is running'
            return bot.set_ready()
        if cmd == '/auto':
            bot.auto_ready = arg != 'off'
            if bot.auto_ready:
                bot.maybe_auto_ready()
            return f"auto-ready {'on' if bot.auto_ready else 'off'} (only once another player is in the room)"
        if cmd == '/kick':
            pid = bot.find_player(arg)
            if pid is None:
                return f'no single player matches "{arg}" (see /players)'
            if bot.player_id is None:
                return 'only players can vote: use /play first'
            res = bot.request('player:kick', {'player': pid})
            if not res.get('success'):
                return 'vote not accepted'
            if bot.is_master():
                return f'{bot.pname(pid)} kicked'
            return f"vote against {bot.pname(pid)} {'registered' if res.get('kicked') else 'withdrawn'}"
        if cmd == '/votes':
            return '\n'.join(f'  kick {bot.pname(t)}' for t in bot.votes) or 'no open votes'
        if cmd == '/name':
            err = check_name(arg)
            if err:
                return err
            if bot.player_id is None:
                bot.name = arg
                return f'name set to {arg} (used when the bot joins the players)'
            res = bot.request('room:name', {'player': bot.player_id, 'name': arg})
            if res.get('success'):
                bot.name = res['name']
                return f"name changed to {res['name']}"
            return res.get('error', 'name not changed')
        if cmd == '/color':
            err = check_color(arg)
            if err:
                return err
            if bot.player_id is None:
                bot.color = arg
                return f'colour set to {arg}'
            res = bot.request('room:color', {'player': bot.player_id, 'color': arg})
            if res.get('success'):
                bot.color = res['color']
                return f"colour changed to {res['color']}"
            return 'colour not accepted (too dark?)'
        if cmd == '/spectate':
            return bot.remove_player() or 'the bot is now spectating'
        if cmd == '/play':
            return bot.add_player() or 'the bot joined the players'
        if cmd == '/aggression':
            lo, hi = getattr(bot, 'trained_aggression', (0.0, 1.0))
            if not arg:
                return f'aggression {bot.aggression} (trained range {lo}-{hi}; 0 = only survive)'
            bot.aggression = float(arg)
            note = '' if lo <= bot.aggression <= hi else f' (outside the trained range {lo}-{hi}!)'
            return f'aggression set to {bot.aggression}{note}'
        if cmd == '/greedy':
            bot.greedy = arg == 'on'
            return f"greedy {'on' if bot.greedy else 'off'}"
        if cmd == '/status':
            role = 'room master' if bot.is_master() else 'player' if bot.player_id is not None else 'spectator'
            return (f'room "{bot.room}" | the bot is {role} as {bot.name} ({bot.color}) | '
                    f"{'game running' if bot.in_game else 'in the lobby'} | latency {bot.rtt_ms:.0f} ms | "
                    f'rounds won {bot.wins}/{bot.rounds} | auto-ready {"on" if bot.auto_ready else "off"} | '
                    f'aggression {bot.aggression}')
        # room master
        master_cmds = ('/launch', '/maxscore', '/bonuses', '/bonus', '/bonusrate', '/open')
        if cmd in master_cmds and cmd != '/bonuses' and not bot.is_master():
            return 'only the room master can do this'
        if cmd == '/launch':
            bot.send('room:launch')
            return None
        if cmd == '/maxscore':
            res = bot.request('room:config:max-score', {'maxScore': int(arg or 0)})
            return None if res.get('success') else 'not changed'
        if cmd == '/bonuses':
            bonuses = bot.config.get('bonuses', {})
            return '\n'.join(f"  {'on ' if bonuses.get(b, True) else 'off'} {b}" for b in BONUS_NAMES)
        if cmd == '/bonus':
            match = [b for b in BONUS_NAMES if b.lower() == arg.lower()] or \
                [b for b in BONUS_NAMES if arg.lower() in b.lower()]
            if len(match) != 1:
                return f'unknown or ambiguous bonus "{arg}" (see /bonuses)'
            res = bot.request('room:config:bonus', {'bonus': match[0]})
            return None if res.get('success') else 'not changed'
        if cmd == '/bonusrate':
            res = bot.request('room:config:variable', {'variable': 'bonusRate', 'value': float(arg)})
            return None if res.get('success') else 'not changed (value must be between -1 and 1)'
        if cmd == '/open':
            res = bot.request('room:config:open', {'open': arg != 'off'})
            return None if res.get('success') else 'not changed'
        return f'unknown command {cmd} (see /help)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', help='checkpoint to play with')
    ap.add_argument('--room', help='room URL (https://www.curvytron.com/#/room/<name>) or name')
    ap.add_argument('--name', help='nickname of the bot')
    ap.add_argument('--color', help='colour of the bot, #rrggbb')
    ap.add_argument('--greedy', action='store_true', help='most likely move instead of sampling')
    ap.add_argument('--auto-ready', action='store_true',
                    help='get ready automatically on joining and after each game (also /auto on)')
    ap.add_argument('--aggression', type=float, default=None,
                    help='hunting vs surviving: kill bonus the bot plays for (default 0.2, 0 = only survive)')
    args = ap.parse_args()

    model_path = args.model
    if not model_path:
        models = find_models()
        if models:
            print('available models (newest first):')
            for i, m in enumerate(models[:15]):
                print(f'  {i + 1:2d}) {m}')
        def check_model(v):
            if v.isdigit() and models and 1 <= int(v) <= min(len(models), 15):
                return None
            return None if os.path.isfile(v) else 'type a number from the list or a path to a .pt file'
        choice = ask('model', '1' if models else None, check_model)
        model_path = models[int(choice) - 1] if choice.isdigit() else choice
    room_text = args.room or ask('room URL or name', check=lambda v: None if v else 'required')
    server, room, password = parse_room(room_text)
    name = args.name or ask('bot nickname', 'rlbot', check_name)
    color = args.color or ask('bot colour', '#%02x%02x%02x' % tuple(random.randint(120, 255) for _ in range(3)),
                              check_color)
    if args.name and check_name(args.name):
        raise SystemExit(check_name(args.name))
    if args.color and check_color(args.color):
        raise SystemExit(f'--color: {check_color(args.color)}')

    print(f'loading {model_path} ...')
    torch.set_num_threads(1)
    model, ckpt = load_policy(model_path)
    decide_every = ckpt.get('args', {}).get('frame_skip', 3)
    lo, hi = (float(v) for v in str(ckpt.get('args', {}).get('kill_reward', '0,1')).split(','))
    aggression = 0.2 if args.aggression is None else args.aggression
    if not lo <= aggression <= hi:
        print(f'note: aggression {aggression} is outside the trained range {lo}-{hi}')
    bot = RoomBot(model, room, password, decide_every=decide_every, greedy=args.greedy, aggression=aggression,
                  name=name, color=color)
    bot.trained_aggression = (lo, hi)
    bot.auto_ready = args.auto_ready
    print('compiling the bot (first time ~1 min, then cached) ...')
    bot.warmup()

    print(f'connecting to {server} ...')
    bot.start()
    bot.connect_to_server(server)
    err = bot.enter_room()
    if err:
        bot.join(1)
        raise SystemExit(f'could not join room "{room}": {err}')
    print(f'joined room "{room}" ({len(bot.players)} players)')
    err = bot.add_player()
    if err:
        print(f'[bot] {err}')
    elif not bot.auto_ready:
        print(f'[bot] playing as {bot.name}, not ready: type /ready (or /auto on to stay ready after each game)')
    else:
        print(f'[bot] playing as {bot.name}'
              + ('' if len(bot.players) >= 2 else ', will get ready when another player joins'))
    try:
        Console(bot).run()
    finally:
        try:
            bot.send('room:leave')
        except Exception:
            pass
        bot.join(1)
        print('bye')


if __name__ == '__main__':
    main()
