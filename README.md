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
