"""
MCR Mahjong Game Engine
Chinese Official Mahjong (Mahjong Competition Rules)
"""

from .tiles import Tile, TileSuit, TileSet
from .player import Player, Meld, MeldType
from .wall import Wall
from .game import Game, GamePhase, Action, ActionType

__version__ = "0.1.0"
__all__ = [
    "Tile",
    "TileSuit", 
    "TileSet",
    "Player",
    "Meld",
    "MeldType",
    "Wall",
    "Game",
    "GamePhase",
    "Action",
    "ActionType",
]

