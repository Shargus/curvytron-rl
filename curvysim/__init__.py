"""Fast Curvytron simulator, observations and environments for RL."""
from .core import BONUS_NAMES, board_size
from .obs import OBS_DIM
from .env import BatchEnv

__all__ = ['BONUS_NAMES', 'board_size', 'OBS_DIM', 'BatchEnv']
