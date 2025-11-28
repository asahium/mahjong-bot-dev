"""
MCR Mahjong Player Module

Handles player state, hand management, and melds.
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .tiles import Tile, TileSet, TileSuit


class MeldType(IntEnum):
    """Types of melds (combinations) a player can have"""
    CHOW = 0      # 顺子 - Sequence of 3 consecutive tiles in same suit
    PONG = 1      # 刻子 - 3 identical tiles
    KONG = 2      # 杠 - 4 identical tiles (exposed)
    CONCEALED_KONG = 3  # 暗杠 - 4 identical tiles (concealed)


@dataclass
class Meld:
    """
    Represents a meld (combination) of tiles.
    
    Attributes:
        meld_type: Type of meld (Chow, Pong, Kong, Concealed Kong)
        tiles: List of tiles in the meld
        is_concealed: Whether the meld is concealed (hidden from other players)
        source_player: Index of player the tile was claimed from (for Chow/Pong/Kong)
        source_tile: The tile that was claimed to form this meld
    """
    meld_type: MeldType
    tiles: List[Tile]
    is_concealed: bool = False
    source_player: Optional[int] = None
    source_tile: Optional[Tile] = None
    
    def __post_init__(self):
        """Validate meld"""
        if self.meld_type == MeldType.CHOW:
            if len(self.tiles) != 3:
                raise ValueError("Chow must have exactly 3 tiles")
            # Validate sequence
            sorted_tiles = sorted(self.tiles)
            if not self._is_valid_sequence(sorted_tiles):
                raise ValueError("Invalid Chow sequence")
        elif self.meld_type == MeldType.PONG:
            if len(self.tiles) != 3:
                raise ValueError("Pong must have exactly 3 tiles")
            if not all(t == self.tiles[0] for t in self.tiles):
                raise ValueError("Pong tiles must be identical")
        elif self.meld_type in (MeldType.KONG, MeldType.CONCEALED_KONG):
            if len(self.tiles) != 4:
                raise ValueError("Kong must have exactly 4 tiles")
            if not all(t == self.tiles[0] for t in self.tiles):
                raise ValueError("Kong tiles must be identical")
    
    def _is_valid_sequence(self, tiles: List[Tile]) -> bool:
        """Check if tiles form a valid sequence"""
        if len(tiles) != 3:
            return False
        # Must be same numbered suit
        if tiles[0].suit not in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            return False
        if not all(t.suit == tiles[0].suit for t in tiles):
            return False
        # Values must be consecutive
        values = sorted(t.value for t in tiles)
        return values[1] == values[0] + 1 and values[2] == values[1] + 1
    
    @property
    def base_tile(self) -> Tile:
        """Get the base tile of the meld (first in sequence or the identical tile)"""
        if self.meld_type == MeldType.CHOW:
            return min(self.tiles)
        return self.tiles[0]
    
    def to_count_array(self) -> np.ndarray:
        """Convert meld to 34-element count array"""
        counts = np.zeros(34, dtype=np.int8)
        for tile in self.tiles:
            counts[tile.tile_index] += 1
        return counts
    
    def __repr__(self) -> str:
        return f"Meld({self.meld_type.name}, {self.tiles})"
    
    def __str__(self) -> str:
        tiles_str = " ".join(str(t) for t in sorted(self.tiles))
        concealed = "暗" if self.is_concealed else "明"
        return f"[{concealed}{self.meld_type.name}: {tiles_str}]"


@dataclass
class Player:
    """
    Represents a Mahjong player.
    
    Attributes:
        index: Player index (0-3, 0=East, 1=South, 2=West, 3=North)
        hand: Tiles in player's hand (concealed)
        melds: Declared melds (Chows, Pongs, Kongs)
        discards: Tiles the player has discarded
        is_riichi: Whether player has declared riichi (not used in MCR, kept for compatibility)
        score: Current score
    """
    index: int
    hand: TileSet = field(default_factory=TileSet)
    melds: List[Meld] = field(default_factory=list)
    discards: List[Tile] = field(default_factory=list)
    score: int = 0
    
    def add_tile(self, tile: Tile) -> None:
        """Add a tile to the player's hand"""
        self.hand.add(tile)
    
    def discard_tile(self, tile: Tile) -> bool:
        """
        Discard a tile from the player's hand.
        Returns True if successful, False if tile not in hand.
        """
        if self.hand.remove(tile):
            self.discards.append(tile)
            return True
        return False
    
    def remove_tile(self, tile: Tile) -> bool:
        """Remove a tile from hand without adding to discards"""
        return self.hand.remove(tile)
    
    def declare_meld(self, meld: Meld) -> None:
        """Add a declared meld"""
        self.melds.append(meld)
    
    def can_chow(self, tile: Tile) -> List[Tuple[Tile, Tile]]:
        """
        Check if player can form a Chow with the given tile.
        Returns list of possible pairs from hand that would complete the Chow.
        Only valid for numbered suits.
        """
        if tile.suit not in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            return []
        
        possible = []
        hand_counts = self.hand.to_count_array()
        value = tile.value
        suit_offset = tile.suit * 9  # Offset for this suit in the count array
        
        # Check for X-1, X (tile is X+1)
        if value >= 3:
            idx1, idx2 = suit_offset + value - 3, suit_offset + value - 2
            if hand_counts[idx1] > 0 and hand_counts[idx2] > 0:
                t1 = Tile.from_index(idx1)
                t2 = Tile.from_index(idx2)
                possible.append((t1, t2))
        
        # Check for X-1, X+1 (tile is X)
        if 2 <= value <= 8:
            idx1, idx2 = suit_offset + value - 2, suit_offset + value
            if hand_counts[idx1] > 0 and hand_counts[idx2] > 0:
                t1 = Tile.from_index(idx1)
                t2 = Tile.from_index(idx2)
                possible.append((t1, t2))
        
        # Check for X, X+1 (tile is X-1)
        if value <= 7:
            idx1, idx2 = suit_offset + value, suit_offset + value + 1
            if hand_counts[idx1] > 0 and hand_counts[idx2] > 0:
                t1 = Tile.from_index(idx1)
                t2 = Tile.from_index(idx2)
                possible.append((t1, t2))
        
        return possible
    
    def can_pong(self, tile: Tile) -> bool:
        """Check if player can form a Pong with the given tile"""
        return self.hand.count(tile) >= 2
    
    def can_kong(self, tile: Tile) -> bool:
        """Check if player can form a Kong with the given tile (from discard)"""
        return self.hand.count(tile) >= 3
    
    def can_concealed_kong(self) -> List[Tile]:
        """
        Get list of tiles that player can declare concealed Kong with.
        (4 of the same tile in hand)
        """
        counts = self.hand.to_count_array()
        kong_tiles = []
        for idx, count in enumerate(counts):
            if count == 4:
                kong_tiles.append(Tile.from_index(idx))
        return kong_tiles
    
    def can_add_to_kong(self) -> List[Tile]:
        """
        Get list of tiles that can be added to existing Pongs to form Kongs.
        (Player has a tile in hand matching an exposed Pong)
        """
        addable = []
        for meld in self.melds:
            if meld.meld_type == MeldType.PONG:
                tile = meld.tiles[0]
                if self.hand.contains(tile):
                    addable.append(tile)
        return addable
    
    def get_hand_tiles(self) -> List[Tile]:
        """Get list of tiles in hand"""
        return list(self.hand.tiles)
    
    def get_all_tiles(self) -> List[Tile]:
        """Get all tiles (hand + melds)"""
        all_tiles = list(self.hand.tiles)
        for meld in self.melds:
            all_tiles.extend(meld.tiles)
        return all_tiles
    
    def get_hand_count_array(self) -> np.ndarray:
        """Get 34-element count array for hand"""
        return self.hand.to_count_array()
    
    def get_discards_count_array(self) -> np.ndarray:
        """Get 34-element count array for discards"""
        discards_set = TileSet(self.discards)
        return discards_set.to_count_array()
    
    @property
    def num_tiles_in_hand(self) -> int:
        """Number of tiles in hand"""
        return len(self.hand)
    
    @property
    def num_melds(self) -> int:
        """Number of declared melds"""
        return len(self.melds)
    
    @property
    def is_menzen(self) -> bool:
        """Check if hand is fully concealed (no exposed melds)"""
        return all(meld.is_concealed for meld in self.melds)
    
    def reset(self) -> None:
        """Reset player state for a new round"""
        self.hand = TileSet()
        self.melds = []
        self.discards = []
    
    def copy(self) -> 'Player':
        """Create a deep copy of the player"""
        new_player = Player(
            index=self.index,
            hand=self.hand.copy(),
            melds=list(self.melds),  # Melds are dataclasses, shallow copy is OK
            discards=list(self.discards),
            score=self.score
        )
        return new_player
    
    def __repr__(self) -> str:
        return f"Player({self.index}, hand={len(self.hand)}, melds={len(self.melds)})"
    
    def __str__(self) -> str:
        position_names = ["East", "South", "West", "North"]
        hand_str = str(self.hand)
        melds_str = " | ".join(str(m) for m in self.melds) if self.melds else "None"
        return f"{position_names[self.index]}: Hand[{hand_str}] Melds[{melds_str}]"

