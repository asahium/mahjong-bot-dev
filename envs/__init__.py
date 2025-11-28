"""
Mahjong Gymnasium Environments
Supports both MCR (Chinese Official) and Riichi (Japanese) Mahjong.
"""

from .mcr_env import MCRMahjongEnv
from .riichi_env import RiichiMahjongEnv

__all__ = ["MCRMahjongEnv", "RiichiMahjongEnv"]

