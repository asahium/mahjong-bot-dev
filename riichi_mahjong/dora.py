"""
Dora System for Riichi Mahjong

Handles all dora-related functionality:
- Regular dora (from indicators)
- Uradora (under-dora, revealed for riichi wins)
- Kandora (revealed after kan)
- Akadora (red fives)
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcr_mahjong.tiles import Tile, TileSet, TileSuit


@dataclass
class DoraSystem:
    """
    Manages dora (bonus tiles) in Riichi Mahjong.
    
    Dora add han to a winning hand without being yaku themselves.
    Types of dora:
    - Regular dora: Based on indicator tiles
    - Uradora: Additional indicators revealed for riichi wins
    - Kandora: New dora revealed after each kan
    - Akadora: Red 5 tiles (if enabled)
    """
    
    # Indicator tiles
    dora_indicators: List[Tile] = field(default_factory=list)
    uradora_indicators: List[Tile] = field(default_factory=list)
    
    # Settings
    red_fives_enabled: bool = False
    num_red_fives: int = 0  # Total number (typically 3, one per suit)
    
    # Which tiles are red fives (for tracking specific instances)
    red_five_indices: Set[int] = field(default_factory=set)  # tile IDs
    
    def __post_init__(self):
        """Initialize red five tracking if enabled"""
        if self.red_fives_enabled and self.num_red_fives > 0:
            self._setup_red_fives()
    
    def _setup_red_fives(self):
        """Set up which 5 tiles are red"""
        # Typically: one red 5 each in characters, bamboos, dots
        # These are specific tile instance IDs
        # 5万 is tiles at index 4*4=16 through 19 (4 copies of 5万)
        # We designate one copy in each suit as red
        
        # 5万 (characters 5): tile indices 4*4 = 16-19, pick index 16
        # 5条 (bamboo 5): tile indices (9+4)*4 = 52-55, pick index 52  
        # 5筒 (dots 5): tile indices (18+4)*4 = 88-91, pick index 88
        
        if self.num_red_fives >= 1:
            self.red_five_indices.add(16)  # Red 5 characters
        if self.num_red_fives >= 2:
            self.red_five_indices.add(52)  # Red 5 bamboos
        if self.num_red_fives >= 3:
            self.red_five_indices.add(88)  # Red 5 dots
        if self.num_red_fives >= 4:
            # Fourth red (sometimes used)
            self.red_five_indices.add(17)  # Second red 5 characters
    
    def get_dora_tile(self, indicator: Tile) -> Tile:
        """
        Get the actual dora tile from an indicator.
        
        The dora is the next tile in sequence:
        - Numbers: 1->2->...->9->1
        - Winds: E->S->W->N->E
        - Dragons: Red->Green->White->Red
        """
        if indicator.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            # Numbered suits wrap 9->1
            next_val = indicator.value % 9 + 1
            return Tile(indicator.suit, next_val)
        elif indicator.suit == TileSuit.WINDS:
            # Winds: East->South->West->North->East
            next_val = (indicator.value + 1) % 4
            return Tile(TileSuit.WINDS, next_val)
        else:  # Dragons
            # Dragons: Red->Green->White->Red (0->1->2->0)
            next_val = (indicator.value + 1) % 3
            return Tile(TileSuit.DRAGONS, next_val)
    
    def get_all_dora_tiles(self) -> List[Tile]:
        """Get list of all current dora tiles"""
        return [self.get_dora_tile(ind) for ind in self.dora_indicators]
    
    def get_all_uradora_tiles(self) -> List[Tile]:
        """Get list of all uradora tiles"""
        return [self.get_dora_tile(ind) for ind in self.uradora_indicators]
    
    def add_dora_indicator(self, indicator: Tile) -> None:
        """Add a new dora indicator (e.g., after kan)"""
        self.dora_indicators.append(indicator)
    
    def add_uradora_indicator(self, indicator: Tile) -> None:
        """Add a new uradora indicator"""
        self.uradora_indicators.append(indicator)
    
    def count_dora(
        self, 
        tiles: List[Tile],
        include_uradora: bool = False,
        include_akadora: bool = True
    ) -> int:
        """
        Count total dora in a collection of tiles.
        
        Args:
            tiles: List of tiles to check
            include_uradora: Whether to count uradora (for riichi wins)
            include_akadora: Whether to count red fives
            
        Returns:
            Total dora count
        """
        count = 0
        
        # Get dora tiles
        dora_tiles = self.get_all_dora_tiles()
        if include_uradora:
            dora_tiles.extend(self.get_all_uradora_tiles())
        
        # Count regular dora
        for tile in tiles:
            for dora in dora_tiles:
                if tile == dora:  # Same suit and value
                    count += 1
        
        # Count akadora
        if include_akadora and self.red_fives_enabled:
            for tile in tiles:
                if tile.id in self.red_five_indices:
                    count += 1
        
        return count
    
    def count_dora_in_hand(
        self,
        hand: TileSet,
        melds: List,
        include_uradora: bool = False
    ) -> int:
        """
        Count dora in a player's complete hand (tiles + melds).
        
        Args:
            hand: Player's hand tiles
            melds: Player's declared melds
            include_uradora: Whether to include uradora
            
        Returns:
            Total dora count
        """
        all_tiles = list(hand.tiles)
        for meld in melds:
            all_tiles.extend(meld.tiles)
        
        return self.count_dora(all_tiles, include_uradora)
    
    def is_red_five(self, tile: Tile) -> bool:
        """Check if a specific tile instance is a red five"""
        if not self.red_fives_enabled:
            return False
        return tile.id in self.red_five_indices
    
    def get_dora_indicator_info(self) -> dict:
        """Get information about current dora state"""
        return {
            "indicators": self.dora_indicators,
            "dora_tiles": self.get_all_dora_tiles(),
            "num_indicators": len(self.dora_indicators),
            "uradora_revealed": len(self.uradora_indicators) > 0,
            "red_fives_enabled": self.red_fives_enabled,
        }
    
    def to_observation_array(self) -> np.ndarray:
        """
        Convert dora indicators to observation array.
        
        Returns:
            (5, 34) array where each row is a dora indicator (one-hot)
            Maximum 5 dora indicators (1 initial + 4 from kans)
        """
        obs = np.zeros((5, 34), dtype=np.int8)
        
        for i, indicator in enumerate(self.dora_indicators[:5]):
            obs[i, indicator.tile_index] = 1
        
        return obs
    
    def reset(self) -> None:
        """Reset for a new round"""
        self.dora_indicators = []
        self.uradora_indicators = []
    
    def copy(self) -> 'DoraSystem':
        """Create a copy of the dora system"""
        new_dora = DoraSystem(
            dora_indicators=list(self.dora_indicators),
            uradora_indicators=list(self.uradora_indicators),
            red_fives_enabled=self.red_fives_enabled,
            num_red_fives=self.num_red_fives,
            red_five_indices=set(self.red_five_indices),
        )
        return new_dora
    
    def __repr__(self) -> str:
        dora_str = ", ".join(str(self.get_dora_tile(i)) for i in self.dora_indicators)
        return f"DoraSystem(dora=[{dora_str}], red_fives={self.num_red_fives})"


def create_dora_system(
    initial_indicator: Tile,
    red_fives: int = 0
) -> DoraSystem:
    """
    Create a new dora system for a round.
    
    Args:
        initial_indicator: First dora indicator tile
        red_fives: Number of red five tiles (0, 3, or 4)
        
    Returns:
        Configured DoraSystem
    """
    system = DoraSystem(
        dora_indicators=[initial_indicator],
        red_fives_enabled=red_fives > 0,
        num_red_fives=red_fives,
    )
    
    if red_fives > 0:
        system._setup_red_fives()
    
    return system

