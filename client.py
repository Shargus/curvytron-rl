from __future__ import absolute_import, division, print_function

import websocket
import json
import threading
from collections import defaultdict
import numpy as np

from skimage import draw

websocket.enableTrace(False)

# MONKEY PATCHING for websocket.WebSocket.connect
def _connect(self, url, **options):
    """
    MONKEY-PATCHED VERSION OF websocket.WebSocket.connect

    Connect to url. url is websocket url scheme.
    ie. ws://host:port/resource
    You can customize using 'options'.
    See websocket.WebSocket.connect for more details on options.
    """
    self.sock_opt.timeout = options.get("timeout", self.sock_opt.timeout)
    self.sock, addrs = websocket._http.connect(
        url, self.sock_opt, websocket._http.proxy_info(**options), options.pop("socket", None)
    )

    try:
        self.handshake_response = websocket._handshake.handshake(self.sock, url, *addrs, **options)
        for _ in range(options.pop("redirect_limit", 3)):
            if self.handshake_response.status in websocket._handshake.SUPPORTED_REDIRECT_STATUSES:
                # THIS IS THE KEY PART: WE NEED TO CONVERT THE REDIRECT URL TO USE WS(S) PROTOCOL
                redirect_url = self.handshake_response.headers["location"]
                # Convert HTTP(S) redirect to WS(S) protocol
                if redirect_url.startswith('https://'):
                    redirect_url = 'wss://' + redirect_url[8:]
                elif redirect_url.startswith('http://'):
                    redirect_url = 'ws://' + redirect_url[7:]
                self.sock.close()
                self.sock, addrs = websocket._http.connect(
                    redirect_url,
                    self.sock_opt,
                    websocket._http.proxy_info(**options),
                    options.pop("socket", None),
                )
                self.handshake_response = websocket._handshake.handshake(
                    self.sock, redirect_url, *addrs, **options
                )
        self.connected = True
    except:
        if self.sock:
            self.sock.close()
            self.sock = None
        raise
websocket.WebSocket.connect = _connect

__all__ = ['CurvytronClient']


def hex_to_rgb(hexstring):
    """Convert hex color string to RGB tuple.

    Args:
        hexstring (str): color in hex format, e.g. '#ff0000'

    Returns:
        tuple: (r, g, b) values corresponding to the input hex color
    """
    r = int(hexstring[1:3], 16)
    g = int(hexstring[3:5], 16)
    b = int(hexstring[5:], 16)
    return (r,g,b)


class Game(object):
    """The game state.

    Args:
        players (dict): A dictionary of all players in the game.
        trails (defaultdict): A dictionary of all trails in the game.
    """

    def __init__(self):
        self.players = {}
        self.trails = defaultdict(list)


class Server(object):
    """The server state.

    Args:
        address (str): The server address.
        rooms (list): A list of all rooms on the server.
    """
    def __init__(self):
        self.address = ''
        self.rooms = []


class Player(object):
    """The player state.

    Args:
        name (str): The player's name.
        color (str): The player's color in hex format, e.g. '#ff0000'.
        angle (float): The player's current angle in radians.
        printing (bool): Whether the player is currently printing a trail.
        updated (bool): Whether the player's state has been updated since the last action.
        position (tuple): The player's current position as (x, y) coordinates.
        draw_position (tuple): The player's current position scaled to the dimensions of the canvas, as (draw_x, draw_y) coordinates.
        alive (bool): Whether the player is currently alive in the game.
    """

    def __init__(self, name, color, angle=None, printing=False, updated=False, position=None, draw_position=None):
        self.name = name
        self.color = color
        self.printing = printing
        self.updated = updated
        self.angle = angle
        self.position = position
        self.draw_position = draw_position
        self.alive=True

    def set_property(self, property, value):
        return setattr(self, property, value)


class CurvytronClient(threading.Thread):
    """Client for interacting with a curvytron server.

    Args:
        name (str): The player's name.
        color (str): The player's color in hex format, e.g. '#ff0000'.
        width (int): The width and height of the canvas to return in get_canvas().
        verbose (bool): Whether to print verbose output of client actions and messages received from the server.
        n_frames (int): If stepped is True, the number of frames to wait for between actions.
        stepped (bool): Whether to wait for n_frames frames between actions. If False, the client will not wait between actions and will return the most recent canvas in get_canvas().
    """
    BONUS_NAMES = [
        'BonusSelfSmall',
        'BonusSelfSlow',
        'BonusSelfFast',
        'BonusSelfMaster',
        'BonusEnemySlow',
        'BonusEnemyFast',
        'BonusEnemyBig',
        'BonusEnemyInverse',
        'BonusEnemyStraightAngle',
        'BonusGameBorderless',
        'BonusAllColor',
        'BonusGameClear'
    ]
    BONUS_PRESETS = {
        'all': BONUS_NAMES,
        'speed': [
            'BonusSelfFast',
            'BonusEnemyFast'
            ],
        'supersize': [
            'BonusEnemyBig'
            ],
        'solo': [
            'BonusSelfSmall',
            'BonusSelfSlow',
            'BonusSelfFast',
            'BonusSelfMaster',
            'BonusGameBorderless',
            'BonusGameClear'
            ],
        'none': None
    }
    FETCH_ROOMS = '[["room:fetch"]]'
    WHOAMI = '[["whoami",null,{msg_id}]]'
    MAKE_ROOM = '[["room:create",{{"name":"{room_name}"}},{msg_id}]]'
    PLAYER_MOVE = '[["player:move",{{"avatar":{player_id},"move":{action}}}]]'
    JOIN_ROOM = '[["room:join",{{"name":"{room_name}","password":null}},{msg_id}]]'
    ADD_PLAYER = '[["player:add",{{"name":"{player_name}","color":"{player_color}"}},{msg_id}]]'
    READY = '[["ready"]]'
    PLAYER_READY = '[["room:ready", {{"player": {player_id}}}, {msg_id}]]'
    BONUS_CHANGE_ROOM = '[["room:config:bonus",{{"bonus":"{bonus_name}","enabled":{enabled}}}, {msg_id}]]'
    BOT_READY = '[["bot:ready"]]'
    MAX_SCORE = '[["room:config:max-score",{{"maxScore":{score}}},{msg_id}]]'

    def __init__(self, name='Random User', color="#ffffff", width=200, verbose=False, n_frames=3, stepped=False):
        super(CurvytronClient, self).__init__()

        self.message_id = -1

        self.ws = websocket.WebSocket()
        self.alive = threading.Event()
        self.alive.set()
        self.verbose = verbose
        self.connected = False

        self.name = name
        self.color = color
        self.game_state = None
        self.active_game = False
        self.active_round = False
        self.server = Server()
        self.game = Game()
        self.message_responses = {}
        self.player_alive = True
        self.round_score = 0
        self.bg_color = hex_to_rgb("#222222")

        self.stepped = stepped
        if not self.stepped:
            self.n_frames = 0
        else:
            self.n_frames = n_frames

        self.last_action = 0

        self.board_size = 0
        self.trails = None
        self.width = width
        self.scale = 1
        self.position = None
        self.angle = None

        self.client_id = None
        self.player_id = None

    def connect_to_server(self, server):
        """Connect to the provided server and request client ID.
        
        Args:
            server (str): address of the curvytron server to connect to, e.g. 'localhost:8080' or 'curvytron.com'
        """
        
        # Decide which protocol to use based on server address
        if "curvytron.com" in server:
            protocol = 'wss'    # use secure websocket for curvytron.com
        else:
            protocol = 'ws'     # use regular websocket for localhost or other non-secure servers
        
        # Connect to the curvytron server. The receive thread sets a 0.05 s socket timeout for its
        # loop, far too short for a TLS handshake over the internet: use a separate one here.
        self.ws.connect(f'{protocol}://{server}', timeout=15)
        self.ws.settimeout(0.05)
        self.connected = True
        
        # Request client ID from server
        msg_id = self._send_message(self.WHOAMI)
        self._wait_for_reply(msg_id)
        self.client_id = self.message_responses[msg_id]
        self._send_message(self.FETCH_ROOMS)
        self.server.address = server

    def create_room(self, target_room=None):
        """Create room called target_room with bonuses listed in bonuses

        Args:
            target_room (str): name of the room to create
        """
        for room in self.server.rooms:
            assert room['name'] != target_room, "Room already exists"
        self._send_message(self.MAKE_ROOM, {'room_name': target_room})
        self._wait_for_reply(self.message_id)
        assert (self.message_responses[self.message_id]['success'])

    def get_canvas(self):
        """
        Get the current game canvas as a numpy array. Also updates the trails and position of the player in self.game.

        Returns:
            canvas (np.array of shape (width, width, 3)): the current game canvas as a numpy array of RGB values
        """
        if self.stepped:
            frames = 0
            while frames < self.n_frames:
                while not all([self.game.players[player].updated for player in self.game.players.keys() if self.game.players[player].alive]):
                    continue
                for player in self.game.players.keys():
                    self.game.players[player].updated = False
                frames += 1
                self._send_message(self.BOT_READY)

        board = self.trails.copy()
        width = np.ceil(1.2 * self.scale)
        for player in self.game.players.keys():
            draw_x, draw_y = self.game.players[player].draw_position
            rr, cc = draw.disk(
                center=(draw_y, draw_x),
                radius=width / 2,
                shape=board.shape
                )
            color = hex_to_rgb(self.game.players[player].color)
            draw.set_color(board, (rr, cc), color)

        return board

    def join(self, timeout=None):
        """Closes the websocket connection and stops the client thread.
        
        Args:
            timeout (float): number of seconds to wait for the client thread to finish after closing the websocket connection. If None, will wait indefinitely.
        """
        self.alive.clear()
        self.ws.close()
        super(CurvytronClient, self).join(timeout)

    def join_room(self, target_room, on_bonuses=None, max_score=3):
        """Joins or creates a room.
        
        If the provided room exists, join it. Otherwise, create and configure the room,
        and then join it.

        Args:
            target_room (str): name of the room to join or create and join if it doesn't exist
            on_bonuses (list or str): list of bonuses to turn on in the room, or preset name for bonuses to turn on in the room. See CurvytronClient.BONUS_PRESETS for preset names and corresponding bonuses.
            max_score (int): max score to set for the room (i.e. score at which game ends)
        """
        for room in self.server.rooms:
            if room['name'] == target_room:
                joined_existing = True
                break
        else:
            joined_existing = False
            self.create_room(target_room)

        # Join room
        msg_id = self._send_message(self.JOIN_ROOM, {'room_name': target_room})
        self._wait_for_reply(self.message_id)
        assert (self.message_responses[msg_id]['success'])
        self._add_players(self.message_responses[msg_id]['room']['players'])

        msg_id = self._send_message(self.ADD_PLAYER)
        self._wait_for_reply(msg_id)
        assert (self.message_responses[msg_id]['success']), self.message_responses[msg_id]
        if not joined_existing:
            self.set_bonuses(on_bonuses=on_bonuses)
            self.set_max_score(max_score=max_score)

    def run(self):
        """Main loop for receiving messages from the server and updating client state accordingly.
        
        This should be run in a separate thread from the main thread to allow for simultaneous sending of actions and receiving of messages.
        """
        self.ws.settimeout(0.05)
        while self.alive.is_set():
            if self.connected:
                try:
                    recvd = self.ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue

                if recvd:
                    self._process_recvd(recvd)


    def send_action(self, action):
        """Sends an action to the server.
        
        Args:
            action (int in [-1, 0, 1]): the action to send to the server
        """
        if not self.last_action == action:
            self._send_message(self.PLAYER_MOVE, {'action': action})
            self.last_action = action
        if self.stepped:
            self._send_message(self.BOT_READY)

    def send_ready(self):
        """Tells the server that the client is ready to start the game.
        """
        assert (self.player_id is not None)
        msg_id = self._send_message(self.PLAYER_READY)
        self._wait_for_reply(msg_id)
        assert (self.message_responses[msg_id]['success'])

    def set_bonuses(self, on_bonuses=None):
        """Sets bonuses in the room.
        
        Args:
            on_bonuses (list or str): list of bonuses to turn on in the room, or preset name for bonuses to turn on in the room. See CurvytronClient.BONUS_PRESETS for preset names and corresponding bonuses.
        """
        BONUS_NAMES = self.BONUS_NAMES

        if on_bonuses in self.BONUS_PRESETS:
            on_bonuses = self.BONUS_PRESETS[on_bonuses]

        if on_bonuses is None:
            off_bonuses = BONUS_NAMES
            for bonus_name in off_bonuses:
                self._send_message(
                    self.BONUS_CHANGE_ROOM,
                    {'bonus_name': bonus_name, 'enabled': 'false'}
                    )
        else:
            invalid_bonus_names = [bb for bb in on_bonuses if bb not in BONUS_NAMES]
            assert len(invalid_bonus_names) == 0, f"{invalid_bonus_names} are not in valid bonus names ({BONUS_NAMES})"
            off_bonuses = [bb for bb in BONUS_NAMES if bb not in on_bonuses]
            for bonus_name in off_bonuses:
                self._send_message(
                    self.BONUS_CHANGE_ROOM,
                    {'bonus_name': bonus_name, 'enabled': 'false'}
                )

    def set_max_score(self, max_score=3):
        """Sets the max score for the game (i.e. score at which game ends).
        """
        self._send_message(self.MAX_SCORE,{'score':max_score})

    def _add_players(self, players):
        """Adds players to self.game.players based on the provided list of player dicts.

        Args:
            players (list): list of player dicts, where each dict contains 'id', 'name', and 'color' keys corresponding to the player's ID, name, and color in hex format, respectively
        """
        for player in players:
            self.game.players[player['id']] = Player(**{
                'name': player['name'],
                'color': '#ffffff',
                'printing': False}
                )

    def _parse_message(self, message):
        """Parses a message received from the server and updates client state accordingly.

        Args:
            message (list): message received from the server, parsed from JSON into a list. The first element of the list is the message type, and the second element is the message body, which can be a dict or list depending on the message type.
        """
        if self.verbose:
            print("[{}] parsing message: {}".format(self.name, message))

        head = message[0]
        if len(message) > 1:
            body = message[1]

        if head == 'position':
            pid, x, y = body
            if self.stepped:
                self.game.players[pid].updated = True
            if self.game.players[pid].printing:
                self._update_trails(body)
            self._update_position(body)

        elif head == 'angle':
            pid, angle = body
            angle /= 100.0
            self.game.players[pid].angle = angle
            if pid == self.player_id:
                self.angle = angle

        elif head == 'property':
            self.game.players[body[0]].set_property(body[1], body[2])

        elif head == 'die':
            if body[0] == self.player_id:
                self.player_alive = False
            self.game.players[body[0]].alive = False

        elif head == 'score:round':
            if body[0] == self.player_id:
                self.round_score = body[1]

        elif head == "game:start":  # message received at start of subsequent rounds
            for player in self.game.players.keys():
                self.game.players[player].alive = True
            self.active_round = True

        elif head == "room:game:start":  # message received at start of game
            self.active_game = True
            self.board_size = int(np.sqrt((80 * 80) + ((len(self.game.players) - 1) * (80 * 80) / 5.0)))
            self.scale = float(self.width) / self.board_size
            self.trails = np.ones((self.width, self.width, 3), dtype=np.uint8) * 34  # set bg color
            self._send_message(self.READY)

        elif head == "game:stop":  # message received at end of round
            self.active_round = False

        elif head == "round:new":  # message received at start of round
            self.last_action = 0
            self.active_round = True
            self.round_score = 0
            self.game.trails = defaultdict(list)
            self.player_alive = True
            for k in self.game.players.keys():
                self.game.players[k].printing = False
            self.trails = np.ones((self.width, self.width, 3), dtype=np.uint8) * 34  # set bg color

        elif head == "end":  # message received at end of game
            self.active_game = False

        elif head == 'room:join':
            player_details = body['player']
            self.game.players[player_details['id']] = Player(**{'name': player_details['name'],
                                                                'color': player_details['color'],
                                                                'printing': False})
            if player_details['name'] == self.name and player_details['client'] == self.client_id:
                self.player_id = player_details['id']

        elif head == 'room:open':
            self.server.rooms.append(body)

        elif head == 'room:leave':
            _ = self.game.players.pop(body['player'])

        elif isinstance(head, int):
            self.message_responses[head] = body

        else:
            pass

    def _process_recvd(self, recvd):
        """Processes a message received from the server.

        Args:
            recvd (str): message received from the server in JSON string format. This can contain multiple messages in a list, which will be parsed and processed separately.
        """
        messages = json.loads(recvd)
        for message in messages:
            self._parse_message(message)

    def _recv_message(self, timeout=None, default=None):
        """Receives a message from the server, with optional timeout.
        
        Args:
            timeout (float): number of seconds to wait for a message before timing out. If None, will wait indefinitely.
            default: value to return if timeout is reached before a message is received. If None, will return None on timeout.
        
        Returns:
            message (list): message received from the server, parsed from JSON into a list. The first element of the list is the message type, and the second element is the message body, which can be a dict or list depending on the message type. If timeout is reached before a message is received, returns the value of default.
        """
        if timeout:
            prev_timeout = self.ws.gettimeout()
            self.ws.settimeout(timeout)
            try:
                message = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                message = default
            finally:
                self.ws.settimeout(prev_timeout)
        else:
            message = json.loads(self.ws.recv())
        return message

    def _send_message(self, message, message_args=None):
        """Send a message to the server
        
        Args:
            message (str): message template to send, with placeholders for formatting with message_args
            message_args (dict): dict of values to substitute into message template

        Returns:
            message_id (int): if the message template contains a {msg_id} placeholder, the message ID used in the sent message. Otherwise, returns None.
        """
        if message_args is None:
            message_args = {}
        if '{msg_id}' in message:
            self.message_id += 1
        self.ws.send(
            message.format(
                msg_id=self.message_id,
                client_id=self.client_id,
                player_id=self.player_id,
                player_name=self.name,
                player_color=self.color,
                **message_args
            )
        )
        if '{msg_id}' in message:
            return self.message_id

    def _update_position(self, message):
        """Updates the position of the player in self.game based on a position message received from the server.
        
        Args:
            message (list): position message received from the server, in the format [pid, x, y], where pid is the player ID, and x and y are the player's coordinates on the board in the range [0, 100]
        """
        pid, x, y = message
        draw_x = (x / 100.0) * self.scale
        draw_y = (y / 100.0) * self.scale

        if pid == self.player_id:
            self.position = (draw_y, draw_x)
        self.game.players[pid].position=(x,y)
        self.game.players[pid].draw_position = (draw_x,draw_y)

    def _update_trails(self, message):
        """Updates the trails on the canvas based on a position message received from the server.
        
        Args:
            message (list): position message received from the server, in the format [pid, x, y], where pid is the player ID, and x and y are the player's coordinates on the board in the range [0, 100]
        """
        pid, x, y = message
        width = 1.2 * self.scale
        draw_x = (x / 100.0) * self.scale
        draw_y = (y / 100.0) * self.scale
        rr, cc = draw.disk(
            center=(draw_y, draw_x),
            radius=width / 2,
            shape=self.trails.shape
            )
        color = hex_to_rgb(self.game.players[pid].color)
        draw.set_color(self.trails, (rr,cc), color)

    def _wait_for_reply(self, message_id):
        """Waits for a reply from the server with the specified message ID.
        
        Args:
            message_id (int): the ID of the message for which to wait
        """
        while message_id not in self.message_responses:
            continue
