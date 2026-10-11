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

## TODO

* adapt client for step-by-step version of curvytron
    * Need to handle more message passing, probably want something equivalent to only taking every Nth frame
* add message handling for room close, player leaving etc.
* add the other methods and attributes required to make this a full gym environment.

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
subset. The learner's opponents in a round are either itself (pure self-play,
60%), the lookahead heuristic (10%), or a past snapshot from the pool, sampled
with prioritized fictitious self-play (snapshots it does worse against are picked
more). The reward follows Curvytron scoring: an avatar dying k-th gets k points,
the survivor n-1, and the return of a round is `2 * points / (n - 1) - 1`
(+1 for the survivor, -1 for the first out).

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
trainer plays one round with all bonuses on (`--video-players`, default 6; the other
avatars are the bot itself or the heuristic, `--video-opponent`) and saves it as an
animated GIF, both in `runs/<run-name>/videos/` and in tensorboard (Images tab,
`video/sample_round`, with a slider over training steps). The bot being followed
has a white ring; bonus rings are green (affect the catcher), red (affect enemies)
or blue (whole game), and the border turns dashed blue in borderless mode.

Outputs go to `runs/<run-name>/`: `model_latest.pt`, a checkpoint per evaluation in
`pool/`, and tensorboard logs. The evaluations play against the lookahead
heuristic in 2, 4 and 8 player rounds (`eval/win_heur_*`: rate of surviving the
round, `eval/ret_heur_*`: mean round return, 0 = average placement) and against an
older snapshot in 6 player rounds. `--resume` continues from a checkpoint (e.g.
to fine-tune with a different `--delay` or player mix).

```bash
uv run python evaluate.py --model runs/big/model_latest.pt --players 8              # vs heuristic
uv run python evaluate.py --model runs/big/model_latest.pt --opponent runs/big/pool/model_02000.pt --players 2
```

### Playing on a server

```bash
uv run python curvybot.py --model runs/big/model_latest.pt --server curvytron.com --room <room> --games 0
```

The bot joins the room (or creates it, with `--bonuses` and `--max-score`), waits
until the room has `--wait-players` players (default 2) and gets ready. Then join
the room from the browser. For a local test, start a server with
`cd curvytron && node bin/curvytron.js` and use `--server 127.0.0.1:8080`;
several bots can share a room (one process each). By default the bot samples its
moves from the policy; `--greedy` takes the most likely move instead.
