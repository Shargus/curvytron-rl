# curvytron-rl

This repository contains code for a local curvytron client which interacts with a
curvytron server (either locally hosted or the official one), and an environment
interface for reinforcement learning agents.


## Install

```bash
uv sync
```

***
To see the environment drawing the players, run the following in a python script, replacing "server_address" with the actual address of your curvytron server (e.g. 127.0.0.1:8080) (or use "www.curvytron.com" to run on the website):

```python
import curvytron
import matplotlib.pyplot as plt
serveraddress = "127.0.0.1:8080"  # if you're running a local server
# serveraddress = "curvytron.com"  # if you want to run on the game website
env = curvytron.CurvytronEnv(
    server=serveraddress,
    room='room',
    name='random_boi',
    color='#ff0090'
    )
env.reset()
episode_over = False
while not episode_over:
    ob, r, episode_over = env.step(2)
    plt.imshow(ob.pixels)
    plt.pause(0.005)
print("done")
```

## Training a strong bot (curvysim + self-play PPO)

Training through the websocket server runs at wall-clock speed (60 frames/s per
room), far too slow for RL. `curvysim/` is a numba re-implementation of the
server game logic (`curvytron/src/server`): movement and turning, trail bodies
and the trail-latency rule, holes (PrintManager), borders and borderless mode,
all 12 bonuses with their stacking rules, spawn rules, board size per player
count and round scoring, for up to 16 players. Network latency is simulated too.
It runs hundreds of rounds in parallel, so training is limited by the network,
not by the game. The policy trained there is then played on a real server by
`curvybot.py`, which rebuilds the same simulator state from the websocket events,
so the observations online match the ones used in training. The simulator was
checked against the real server: every death the server reported was reproduced
by the bot's mirrored state under the simulator's collision rules.

| file | purpose |
| --- | --- |
| `curvysim/core.py` | simulator (one round per environment) |
| `curvysim/obs.py` | observations, lookahead search / heuristic bot, optional egocentric map |
| `curvysim/env.py` | batched multi-agent environment with auto-reset and latency |
| `curvysim/policy.py` | actor-critic network (MLP, or CNN+MLP with `--map-size`) |
| `train_selfplay.py` | self-play PPO with an opponent pool |
| `evaluate.py` | win rates of a checkpoint vs the heuristic or another checkpoint |
| `curvybot.py` | plays a checkpoint on a real server |

### Observation

All features are egocentric (relative to the avatar's heading):

* 64 raycasts around the avatar: distance to the nearest trail or wall, plus a near-range and a hit-is-wall channel
* the 7 nearest opponents: relative position and heading, speed, size, printing/invincible/inverted/straight-angle state, turning direction
* every bonus on the map (up to 20, nearest first): type, straight-line distance and direction, and the grid path around trails (reachable at all, path length, whether the bot gets there before every opponent)
* own state: position, speed, size, active effects with their remaining time, time before trails start, distance since the last hole
* latency: the current round trip (in frames) and the last 8 moves sent
* lookahead: for each move, sent now (i.e. after the moves already in flight), how long the avatar survives holding it, then with the best follow-up (two horizons)
* territory: on a 64x64 grid of the whole board, the share of free space the avatar reaches before any opponent (Voronoi), and the share it can reach at all, now and after each move held for ~0.8 s
* optional (`--map-size 48`): a 48x48 egocentric occupancy map (trails, own trail, walls) around the head, fed to a CNN
* optional (`--global-map`): a 64x64 map of the whole board (trails, own head and heading, opponents' heads and headings, bonuses by type), fed to a second CNN, so the bot can spot opportunities far away

Actions are left / straight / right, chosen every 3 server frames (50 ms). The
discount (0.998 per decision) gives a ~25 s horizon, enough to plan a move to the
other side of a 13-player board (~9 s at base speed).

### Latency

On a remote server, the state the bot sees is ~RTT/2 old and its moves need
another ~RTT/2 to arrive, so a move takes effect about one round trip after the
state it was computed from. The simulator models this with a per-player action
queue: each player in each round gets a delay sampled from `--delay` (frames, 1
frame = 16.7 ms; default `0,8`, i.e. up to ~133 ms). Online, the bot reads the
round trip from the server's `latency` events and fills the same queue with the
moves it has sent but the server has not applied yet. The bot prints the latency
after each round: if your games on curvytron.com show values above ~130 ms, train
with a wider range (e.g. `--delay 2,12`).

### Training

Each round has 2-13 players (weighted towards 4-13, see `--n-probs`) and a
random bonus setup: no bonuses, all bonuses (the server default) or a random
subset. 20% of the rounds have a league exploiter (`--p-exploit`, see below). Of
the others, 35% are pure self-play: every player is the current bot
(`--p-selfplay`). The rest are mixed: the bot controls one up to a third of the
players, and each other player is (`--p-human`, `--p-heuristic`):

* human-like (45%): the heuristic or a past version of the bot, with human
  handicaps: a reaction delay of 150-300 ms (`--human-delay`), occasional steering
  noise (`--human-noise`), and confusion when a malus changes its controls, i.e. a
  high chance of steering wrong right after inverted controls, 90 degree turns,
  a speed-up, growing or a borderless board, fading over 1-2.5 s
  (`--human-confusion` scales it, table in `curvysim/env.py`)
* the lookahead heuristic (20%)
* a past version of the bot (35%), from a pool sampled with prioritized fictitious
  self-play (versions it does worse against are picked more)

Bot opponents shrug off enemy maluses (they adapt within 50 ms), humans do not.
The bot sees which opponents are human-like, so it can learn to use maluses
against humans without wasting effort on them against bots; when playing,
`--opponents human` (default) or `bots` sets what it is told. The progress log and
tensorboard (`bonuses/...`) show bonuses picked per round with and without
human-like opponents, and evaluations add a 6-player game against human-like
heuristics (also part of the score that selects `model_best.pt`).

The reward follows Curvytron scoring: an avatar dying k-th gets k points,
the survivor n-1, and the return of a round is `2 * points / (n - 1) - 1`
(+1 for the survivor, -1 for the first out).

On top of that, a player whose trail kills an opponent gets a kill bonus
("aggression", hunting vs surviving). Each player draws its own bonus per round
from `--kill-reward` (default 0-1) and sees it as an input, so a single trained
model learns the whole range of play styles and you choose one when playing
(`--aggression`, default 0.2, which favours surviving; 0 = survive only). For scale:
a duel is worth +-1, and in a 10-player round every opponent death already gives
each survivor +0.22. Evaluations and logged returns measure placement only (at
aggression `--eval-aggression`, default 0.2); the progress log also shows kills
per round, split by low and high aggression.

Three additions push the bot away from simply circling in a safe corner:

- **League exploiters** (`--exploiters`, default 2; `--p-exploit`, default 20% of
  rounds). An exploiter is a copy of the main agent that trains only in rounds
  against the current main agent, taking one slot (up to a third of the slots in
  big rounds) with a higher aggression (`--exploiter-kill-reward`, default 0.6-1:
  the top of the main agent's range, which it has seen as an input).
  Once its placement score against the main agent reaches `--exploiter-target`
  (default 0.65), or after `--exploiter-max-age` updates, a frozen copy joins the
  opponent pool and the exploiter restarts from the latest main agent.
- **Trap rounds** (`--p-trap`, default 15%). The round starts with the learner
  enclosed by a closed trail, its own or an opponent's (`--trap-own`), whose only
  ways out are one or two holes of the usual length. Trails print from the first frame.
- **Territory shaping** (`--territory-coef`, default 0.5). A potential-based term
  `coef * (gamma * phi' - phi)`, where phi is the share of the free board the
  player reaches before every opponent (0 when its round ends). It gives a dense
  signal for claiming and cutting off space without changing the optimal policy.
  That share is computed on a 64x64 grid with an 8-connected search, so that trail
  holes count as passages.

```bash
# GPU server (recommended): CNN over the egocentric map, larger network
uv run python train_selfplay.py --run-name big --device cuda --num-envs 1024 --threads 32 \
    --map-size 48 --global-map --hidden 1024,512 --minibatch 32768 --total-steps 3e9

# CPU only
uv run python train_selfplay.py --run-name cpu --device cpu --hidden 256,256 --num-envs 256 --threads 6

tensorboard --logdir runs
```

GPU memory: with both maps, the rollout (~5-8 GB with 1024 envs) is kept on the GPU
and a 32768 minibatch needs ~20-25 GB of activations, so the command above needs a
large GPU (40-80 GB). On a ~20 GB GPU add `--rollout-on-cpu --minibatch 8192`: the
rollout then stays in RAM and only minibatches are sent to the GPU (~7 GB), at the
cost of slower updates. The progress log shows how the time splits between
simulating and learning, and the peak GPU memory since the previous report
(`gpu mem` line, also in tensorboard as `perf/gpu_peak_reserved_gb`).

Set `--threads` to about the number of physical cores (it drives both torch and
the numba simulator). The first start compiles the simulator (~1-2 minutes, then cached).
Memory is about 2 MB per environment, plus the rollout (~6 GB with both maps
and 1024 envs, held in RAM and on the GPU during the update). With everything enabled,
the simulator produces ~1k agent-steps/s per core; the territory search and the maps
are the expensive part.

Every `--log-every` seconds (default 60) the trainer prints a progress report:
speed and ETA, round results against each opponent type and per table size, how
the bot dies (walls, its own trail, others' trails) and PPO health. Each
evaluation adds a block with the change since the previous one. The output is
plain text, flushed line by line, and is also appended to `runs/<run-name>/train.log`,
so it works the same in a terminal or in a Slurm `sbatch` output file.

Sample videos: every `--video-every` updates (default: at each evaluation) the
trainer plays one round with all bonuses on (`--video-players`, default 6) and
saves it as an animated GIF, both in `runs/<run-name>/videos/` and in tensorboard
(Images tab, `video/sample_round`, with a slider over training steps). By default
(`--video-opponent mixed`) the round is composed like a mixed training round, plus
an exploiter; `self`, `heuristic` and `human` fill the other slots with one kind.
Each head shows its player kind: **B** bot being trained, **E** exploiter, **L**
lookahead heuristic, **P** past version, **H** human-like. The bot being followed
has a white ring. Bonuses use the game's own icons (`curvysim/assets/bonus.png`,
from Curvytron under its MIT license): green affects the catcher, red its enemies,
grey everybody; a legend beside the board names them and the letters. The border
turns dashed blue in borderless mode. Frames are 400 px (plus the legend), about
2-3 MB per GIF.

Outputs go to `runs/<run-name>/`: `model_latest.pt`, a checkpoint per evaluation in
`pool/`, and tensorboard logs. The evaluations play against the lookahead
heuristic in 2, 4 and 8 player rounds (`eval/win_heur_*`: rate of surviving the
round, `eval/ret_heur_*`: mean round return, 0 = average placement) and against an
older snapshot in 6 player rounds. `--resume` starts a new run from a policy (e.g.
to fine-tune with a different `--delay` or player mix).

Every `--checkpoint-every` updates (default 25) the run also writes a full
checkpoint, `state.pt`: weights and optimizer state of the main agent and the
exploiters, the opponent pool, step counters, best score, recent stats and random
generator states. Starting a run whose directory has a `state.pt` resumes from it
(the same command works for a fresh start and a restart), and tensorboard drops the
points logged after the checkpoint. On SIGTERM (preemption, `scancel`, time limit)
the run checkpoints and exits at the next safe point. Only the rounds in progress
and the updates since the last checkpoint are lost. `jobs/train_bot.job` uses this
to run as a preemptable, requeued SLURM job.

```bash
uv run python evaluate.py --model runs/big/model_latest.pt --players 8              # vs heuristic
uv run python evaluate.py --model runs/big/model_latest.pt --opponent runs/big/pool/model_02000.pt --players 2
```

### Joining a room on curvytron.com (interactive)

```bash
uv run python play_room.py
```

It asks for the model (lists the checkpoints in `runs/`, newest first), the room
URL as shown in the browser (e.g. `https://www.curvytron.com/#/room/My%20Room`,
including `?password=...` for private rooms), the bot's nickname and its colour.
They can also be given as `--model`, `--room`, `--name`, `--color`. The bot then
joins the room and plays every game on its own, while the console lets you do
what a player does in the browser:

* plain text is sent to the room chat, and the chat is shown in the console
* `/ready`, `/auto on|off` (auto-ready: off by default, `--auto-ready` to start with it on;
  it only gets ready once another player is in the room, so the bot never starts solo
  games that lock others out), `/spectate`, `/play`
* `/kick <player>` to vote (or kick directly as room master), `/votes`, `/players`
* `/name`, `/color`, `/greedy on|off`, `/status` (role, latency, rounds won), `/quit`
* `/aggression [x]`: show or change how much the bot hunts (also `--aggression` at start)
* `/opponents human|bots`: what the bot is told about the other players (default human)
* as room master: `/launch`, `/maxscore`, `/bonuses`, `/bonus <name>`, `/bonusrate`, `/open on|off`

If a game is already running when the bot arrives, it watches and joins the
players when the game ends (the server does not allow joining mid-game).
`/help` lists all commands, with tab completion for commands and player names.

### Playing on a server (scripted)

```bash
uv run python curvybot.py --model runs/big/model_latest.pt --server curvytron.com --room <room> --games 0
```

The bot joins the room (or creates it, with `--bonuses` and `--max-score`), waits
until the room has `--wait-players` players (default 2) and gets ready. Then join
the room from the browser. For a local test, start a server with
`cd curvytron && node bin/curvytron.js` and use `--server 127.0.0.1:8080`;
several bots can share a room (one process each). By default the bot samples its
moves from the policy; `--greedy` takes the most likely move instead.
