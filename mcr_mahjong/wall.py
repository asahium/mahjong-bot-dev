"""
MCR Mahjong Wall Module

Handles the wall (tile pile), dealing, and drawing.
"""

import random
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .tiles import Tile, TileSet


@dataclass
class Wall:
    """
    Represents the Mahjong wall.
    
    The wall consists of 136 tiles arranged in a specific pattern.
    In MCR, there's no dead wall or dora indicators like in Japanese Mahjong.
    
    Attributes:
        tiles: Remaining tiles in the wall
        dealt_count: Number of tiles that have been dealt/drawn
    """
    tiles: List[Tile] = field(default_factory=list)
    dealt_count: int = 0
    _initial_count: int = 136
    
    def __post_init__(self):
        if not self.tiles:
            self._create_wall()
    
    def _create_wall(self) -> None:
        """Create and shuffle a new wall"""
        tile_set = TileSet.create_full_set()
        self.tiles = list(tile_set.tiles)
        self._initial_count = len(self.tiles)
        self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the wall"""
        random.shuffle(self.tiles)
    
    def draw(self) -> Optional[Tile]:
        """
        Draw one tile from the wall.
        Returns None if wall is empty.
        """
        if not self.tiles:
            return None
        tile = self.tiles.pop()
        self.dealt_count += 1
        return tile
    
    def draw_many(self, count: int) -> List[Tile]:
        """
        Draw multiple tiles from the wall.
        Returns fewer tiles if wall doesn't have enough.
        """
        drawn = []
        for _ in range(count):
            tile = self.draw()
            if tile is None:
                break
            drawn.append(tile)
        return drawn
    
    def deal_hands(self, num_players: int = 4) -> List[List[Tile]]:
        """
        Deal initial hands to all players.
        Each player gets 13 tiles.
        
        Returns list of hands (each hand is a list of 13 tiles).
        """
        hands = [[] for _ in range(num_players)]
        
        # Deal 4 tiles at a time, 3 rounds
        for _ in range(3):
            for player_idx in range(num_players):
                tiles = self.draw_many(4)
                hands[player_idx].extend(tiles)
        
        # Deal 1 final tile to each player
        for player_idx in range(num_players):
            tile = self.draw()
            if tile:
                hands[player_idx].append(tile)
        
        return hands
    
    @property
    def remaining(self) -> int:
        """Number of tiles remaining in the wall"""
        return len(self.tiles)
    
    @property
    def is_empty(self) -> bool:
        """Check if wall is empty"""
        return len(self.tiles) == 0
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the wall with optional random seed"""
        if seed is not None:
            random.seed(seed)
        self.tiles = []
        self.dealt_count = 0
        self._create_wall()
    
    def copy(self) -> 'Wall':
        """Create a copy of the wall"""
        new_wall = Wall.__new__(Wall)
        new_wall.tiles = [Tile(t.suit, t.value, t.id) for t in self.tiles]
        new_wall.dealt_count = self.dealt_count
        new_wall._initial_count = self._initial_count
        return new_wall
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __repr__(self) -> str:
        return f"Wall({self.remaining} tiles remaining)"

