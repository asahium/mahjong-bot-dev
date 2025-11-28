"""
MCR Mahjong Tiles System

Defines all 136 tiles used in Chinese Official Mahjong:
- 9 Characters (万) x4 = 36
- 9 Bamboos (条) x4 = 36  
- 9 Dots (筒) x4 = 36
- 4 Winds (东南西北) x4 = 16
- 3 Dragons (中发白) x4 = 12
Total: 136 tiles
"""

from enum import IntEnum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np


class TileSuit(IntEnum):
    """Tile suits in MCR Mahjong"""
    CHARACTERS = 0  # 万 (Wan) - Numbers 1-9
    BAMBOOS = 1     # 条 (Tiao) - Numbers 1-9
    DOTS = 2        # 筒 (Tong) - Numbers 1-9
    WINDS = 3       # 风 (Feng) - East, South, West, North
    DRAGONS = 4     # 箭 (Jian) - Red, Green, White


class WindType(IntEnum):
    """Wind tile types"""
    EAST = 0   # 东
    SOUTH = 1  # 南
    WEST = 2   # 西
    NORTH = 3  # 北


class DragonType(IntEnum):
    """Dragon tile types"""
    RED = 0    # 中 (Zhong)
    GREEN = 1  # 发 (Fa)
    WHITE = 2  # 白 (Bai)


@dataclass(frozen=True)
class Tile:
    """
    Represents a single Mahjong tile.
    
    Attributes:
        suit: The suit of the tile (Characters, Bamboos, Dots, Winds, Dragons)
        value: The value within the suit (1-9 for numbered suits, 0-3/0-2 for honors)
        id: Unique identifier for this specific tile instance (0-135)
    """
    suit: TileSuit
    value: int
    id: int = 0  # Instance ID (0-3 for each unique tile)
    
    def __post_init__(self):
        """Validate tile values"""
        if self.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            if not 1 <= self.value <= 9:
                raise ValueError(f"Numbered suits must have value 1-9, got {self.value}")
        elif self.suit == TileSuit.WINDS:
            if not 0 <= self.value <= 3:
                raise ValueError(f"Wind tiles must have value 0-3, got {self.value}")
        elif self.suit == TileSuit.DRAGONS:
            if not 0 <= self.value <= 2:
                raise ValueError(f"Dragon tiles must have value 0-2, got {self.value}")
    
    @property
    def is_honor(self) -> bool:
        """Check if tile is an honor tile (Wind or Dragon)"""
        return self.suit in (TileSuit.WINDS, TileSuit.DRAGONS)
    
    @property
    def is_terminal(self) -> bool:
        """Check if tile is a terminal (1 or 9 of numbered suits)"""
        if self.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            return self.value in (1, 9)
        return False
    
    @property
    def is_terminal_or_honor(self) -> bool:
        """Check if tile is terminal or honor"""
        return self.is_terminal or self.is_honor
    
    @property
    def is_simple(self) -> bool:
        """Check if tile is a simple (2-8 of numbered suits)"""
        if self.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            return 2 <= self.value <= 8
        return False
    
    @property
    def is_green(self) -> bool:
        """Check if tile is a green tile (for All Green pattern)"""
        if self.suit == TileSuit.BAMBOOS:
            return self.value in (2, 3, 4, 6, 8)
        if self.suit == TileSuit.DRAGONS:
            return self.value == DragonType.GREEN
        return False
    
    @property
    def tile_index(self) -> int:
        """
        Get unique index for this tile type (0-33).
        Used for encoding tile types (not instances).
        """
        if self.suit == TileSuit.CHARACTERS:
            return self.value - 1  # 0-8
        elif self.suit == TileSuit.BAMBOOS:
            return 9 + self.value - 1  # 9-17
        elif self.suit == TileSuit.DOTS:
            return 18 + self.value - 1  # 18-26
        elif self.suit == TileSuit.WINDS:
            return 27 + self.value  # 27-30
        else:  # DRAGONS
            return 31 + self.value  # 31-33
    
    def __eq__(self, other) -> bool:
        """Two tiles are equal if they have same suit and value (ignoring instance id)"""
        if not isinstance(other, Tile):
            return False
        return self.suit == other.suit and self.value == other.value
    
    def __hash__(self) -> int:
        return hash((self.suit, self.value))
    
    def __lt__(self, other) -> bool:
        """Comparison for sorting"""
        if not isinstance(other, Tile):
            return NotImplemented
        if self.suit != other.suit:
            return self.suit < other.suit
        return self.value < other.value
    
    def __repr__(self) -> str:
        return f"Tile({self.suit.name}, {self.value})"
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        if self.suit == TileSuit.CHARACTERS:
            return f"{self.value}万"
        elif self.suit == TileSuit.BAMBOOS:
            return f"{self.value}条"
        elif self.suit == TileSuit.DOTS:
            return f"{self.value}筒"
        elif self.suit == TileSuit.WINDS:
            wind_names = ["东", "南", "西", "北"]
            return wind_names[self.value]
        else:  # DRAGONS
            dragon_names = ["中", "发", "白"]
            return dragon_names[self.value]
    
    @classmethod
    def from_index(cls, tile_index: int, instance_id: int = 0) -> 'Tile':
        """
        Create a tile from its type index (0-33) and instance id (0-3).
        
        Args:
            tile_index: Tile type index (0-33)
            instance_id: Instance of this tile type (0-3)
        """
        if tile_index < 9:
            return cls(TileSuit.CHARACTERS, tile_index + 1, instance_id)
        elif tile_index < 18:
            return cls(TileSuit.BAMBOOS, tile_index - 9 + 1, instance_id)
        elif tile_index < 27:
            return cls(TileSuit.DOTS, tile_index - 18 + 1, instance_id)
        elif tile_index < 31:
            return cls(TileSuit.WINDS, tile_index - 27, instance_id)
        else:
            return cls(TileSuit.DRAGONS, tile_index - 31, instance_id)
    
    @classmethod
    def from_string(cls, s: str, instance_id: int = 0) -> 'Tile':
        """
        Create tile from string representation.
        
        Args:
            s: String like "1万", "9条", "东", "中", etc.
            instance_id: Instance of this tile type (0-3)
        """
        s = s.strip()
        
        # Handle numbered suits
        if len(s) == 2:
            value = int(s[0])
            suit_char = s[1]
            if suit_char == '万':
                return cls(TileSuit.CHARACTERS, value, instance_id)
            elif suit_char == '条':
                return cls(TileSuit.BAMBOOS, value, instance_id)
            elif suit_char == '筒':
                return cls(TileSuit.DOTS, value, instance_id)
        
        # Handle honor tiles
        wind_map = {"东": 0, "南": 1, "西": 2, "北": 3}
        dragon_map = {"中": 0, "发": 1, "白": 2}
        
        if s in wind_map:
            return cls(TileSuit.WINDS, wind_map[s], instance_id)
        elif s in dragon_map:
            return cls(TileSuit.DRAGONS, dragon_map[s], instance_id)
        
        raise ValueError(f"Cannot parse tile string: {s}")


class TileSet:
    """
    A collection of tiles with utility methods.
    Used to represent hands, melds, discards, etc.
    """
    
    # Total number of unique tile types
    NUM_TILE_TYPES = 34
    # Total tiles in a complete set
    NUM_TILES = 136
    # Copies of each tile type
    COPIES_PER_TYPE = 4
    
    def __init__(self, tiles: Optional[List[Tile]] = None):
        """Initialize tile set with optional list of tiles"""
        self.tiles: List[Tile] = list(tiles) if tiles else []
    
    def add(self, tile: Tile) -> None:
        """Add a tile to the set"""
        self.tiles.append(tile)
    
    def remove(self, tile: Tile) -> bool:
        """
        Remove a tile from the set (matches by suit and value).
        Returns True if removed, False if not found.
        """
        for i, t in enumerate(self.tiles):
            if t == tile:
                self.tiles.pop(i)
                return True
        return False
    
    def contains(self, tile: Tile) -> bool:
        """Check if tile is in the set"""
        return tile in self.tiles
    
    def count(self, tile: Tile) -> int:
        """Count occurrences of a tile type"""
        return sum(1 for t in self.tiles if t == tile)
    
    def count_by_index(self, tile_index: int) -> int:
        """Count tiles by type index"""
        return sum(1 for t in self.tiles if t.tile_index == tile_index)
    
    def sort(self) -> None:
        """Sort tiles by suit and value"""
        self.tiles.sort()
    
    def to_count_array(self) -> np.ndarray:
        """
        Convert to a 34-element array counting each tile type.
        Useful for hand analysis and encoding.
        """
        counts = np.zeros(self.NUM_TILE_TYPES, dtype=np.int8)
        for tile in self.tiles:
            counts[tile.tile_index] += 1
        return counts
    
    def to_binary_array(self) -> np.ndarray:
        """
        Convert to a 136-element binary array.
        Each position represents one of the 136 tile instances.
        """
        binary = np.zeros(self.NUM_TILES, dtype=np.int8)
        # Track how many of each type we've seen
        type_counts = np.zeros(self.NUM_TILE_TYPES, dtype=np.int8)
        
        for tile in self.tiles:
            idx = tile.tile_index
            instance = type_counts[idx]
            if instance < self.COPIES_PER_TYPE:
                binary[idx * 4 + instance] = 1
                type_counts[idx] += 1
        
        return binary
    
    def to_34_channel_array(self) -> np.ndarray:
        """
        Convert to a 34x4 array where each row is a tile type
        and each column indicates presence of that instance.
        """
        array = np.zeros((self.NUM_TILE_TYPES, self.COPIES_PER_TYPE), dtype=np.int8)
        type_counts = np.zeros(self.NUM_TILE_TYPES, dtype=np.int8)
        
        for tile in self.tiles:
            idx = tile.tile_index
            instance = int(type_counts[idx])
            if instance < self.COPIES_PER_TYPE:
                array[idx, instance] = 1
                type_counts[idx] += 1
        
        return array
    
    @classmethod
    def create_full_set(cls) -> 'TileSet':
        """Create a complete set of 136 tiles"""
        tiles = []
        instance_id = 0
        
        # Numbered suits (Characters, Bamboos, Dots)
        for suit in [TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS]:
            for value in range(1, 10):
                for copy in range(4):
                    tiles.append(Tile(suit, value, instance_id))
                    instance_id += 1
        
        # Winds
        for value in range(4):
            for copy in range(4):
                tiles.append(Tile(TileSuit.WINDS, value, instance_id))
                instance_id += 1
        
        # Dragons
        for value in range(3):
            for copy in range(4):
                tiles.append(Tile(TileSuit.DRAGONS, value, instance_id))
                instance_id += 1
        
        return cls(tiles)
    
    def get_unique_tiles(self) -> List[Tile]:
        """Get list of unique tile types in the set"""
        seen = set()
        unique = []
        for tile in self.tiles:
            key = (tile.suit, tile.value)
            if key not in seen:
                seen.add(key)
                unique.append(tile)
        return unique
    
    def copy(self) -> 'TileSet':
        """Create a copy of this tile set"""
        return TileSet([Tile(t.suit, t.value, t.id) for t in self.tiles])
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __iter__(self):
        return iter(self.tiles)
    
    def __getitem__(self, index):
        return self.tiles[index]
    
    def __repr__(self) -> str:
        return f"TileSet({len(self.tiles)} tiles)"
    
    def __str__(self) -> str:
        self.sort()
        return " ".join(str(t) for t in self.tiles)


# Convenience functions for creating specific tiles
def char(value: int, instance_id: int = 0) -> Tile:
    """Create a Characters tile (1-9万)"""
    return Tile(TileSuit.CHARACTERS, value, instance_id)

def bam(value: int, instance_id: int = 0) -> Tile:
    """Create a Bamboos tile (1-9条)"""
    return Tile(TileSuit.BAMBOOS, value, instance_id)

def dot(value: int, instance_id: int = 0) -> Tile:
    """Create a Dots tile (1-9筒)"""
    return Tile(TileSuit.DOTS, value, instance_id)

def wind(wind_type: WindType, instance_id: int = 0) -> Tile:
    """Create a Wind tile (东南西北)"""
    return Tile(TileSuit.WINDS, wind_type, instance_id)

def dragon(dragon_type: DragonType, instance_id: int = 0) -> Tile:
    """Create a Dragon tile (中发白)"""
    return Tile(TileSuit.DRAGONS, dragon_type, instance_id)


# Named wind tiles
EAST = Tile(TileSuit.WINDS, WindType.EAST)
SOUTH = Tile(TileSuit.WINDS, WindType.SOUTH)
WEST = Tile(TileSuit.WINDS, WindType.WEST)
NORTH = Tile(TileSuit.WINDS, WindType.NORTH)

# Named dragon tiles
RED_DRAGON = Tile(TileSuit.DRAGONS, DragonType.RED)
GREEN_DRAGON = Tile(TileSuit.DRAGONS, DragonType.GREEN)
WHITE_DRAGON = Tile(TileSuit.DRAGONS, DragonType.WHITE)

