"""
Riichi Mahjong Game Engine
Japanese Mahjong with support for EMA and Tenhou rules
"""

from .player import RiichiPlayer, PlayerState
from .game import RiichiGame, RiichiPhase, RiichiAction, RiichiActionType
from .scoring import RiichiScorer, Yaku, YakuType
from .dora import DoraSystem
from .rules import RuleSet, EMA_RULES, TENHOU_RULES

__version__ = "0.1.0"
__all__ = [
    "RiichiPlayer",
    "PlayerState",
    "RiichiGame",
    "RiichiPhase",
    "RiichiAction",
    "RiichiActionType",
    "RiichiScorer",
    "Yaku",
    "YakuType",
    "DoraSystem",
    "RuleSet",
    "EMA_RULES",
    "TENHOU_RULES",
]

