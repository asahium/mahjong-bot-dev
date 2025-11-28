"""
Riichi Mahjong Player Module

Handles player state including Riichi declaration and Furiten tracking.
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import numpy as np
import sys
from pathlib import Path

# Import from mcr_mahjong for shared tile system
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcr_mahjong.tiles import Tile, TileSet, TileSuit
from mcr_mahjong.player import Meld, MeldType


class PlayerState(IntEnum):
    """Player states in Riichi Mahjong"""
    NORMAL = 0          # Normal play
    RIICHI = 1          # Declared riichi
    DOUBLE_RIICHI = 2   # Declared riichi on first turn
    IPPATSU = 3         # One round after riichi (eligible for ippatsu)


@dataclass
class FuritenState:
    """
    Tracks furiten status for a player.
    
    Furiten prevents winning by ron in these cases:
    1. Permanent: Player has discarded a winning tile
    2. Temporary: Player passed on a winning ron opportunity
    3. Riichi furiten: Passed on ron while in riichi
    """
    permanent: bool = False      # Discarded a winning tile
    temporary: bool = False      # Passed on a ron this turn
    riichi_furiten: bool = False # Passed on ron while in riichi
    
    # Tiles that would complete the hand (for checking)
    waiting_tiles: Set[int] = field(default_factory=set)  # tile indices
    
    @property
    def is_furiten(self) -> bool:
        """Check if player is in any furiten state"""
        return self.permanent or self.temporary or self.riichi_furiten
    
    def reset_temporary(self):
        """Reset temporary furiten at start of player's turn"""
        self.temporary = False
    
    def set_passed_ron(self, in_riichi: bool):
        """Called when player passes on a ron opportunity"""
        self.temporary = True
        if in_riichi:
            self.riichi_furiten = True
    
    def update_permanent(self, discards: List[Tile]):
        """Update permanent furiten based on discards and waiting tiles"""
        for tile in discards:
            if tile.tile_index in self.waiting_tiles:
                self.permanent = True
                break


@dataclass
class RiichiPlayer:
    """
    Represents a player in Riichi Mahjong.
    
    Attributes:
        index: Player index (0=East, 1=South, 2=West, 3=North)
        hand: Tiles in hand (concealed)
        melds: Declared melds
        discards: Discarded tiles in order
        score: Current point total (starts at 25000 typically)
        state: Current player state (normal, riichi, etc.)
        furiten: Furiten tracking
    """
    index: int
    hand: TileSet = field(default_factory=TileSet)
    melds: List[Meld] = field(default_factory=list)
    discards: List[Tile] = field(default_factory=list)
    discard_info: List[dict] = field(default_factory=list)  # Extra info per discard
    score: int = 25000
    state: PlayerState = PlayerState.NORMAL
    furiten: FuritenState = field(default_factory=FuritenState)
    
    # Riichi-specific
    riichi_turn: int = -1           # Turn when riichi was declared (-1 if not)
    riichi_discard_index: int = -1  # Index of riichi discard in discard list
    ippatsu_valid: bool = False     # Whether ippatsu is still possible
    
    # Turn tracking
    first_draw_completed: bool = False  # For double riichi check
    
    def add_tile(self, tile: Tile) -> None:
        """Add a tile to the player's hand"""
        self.hand.add(tile)
    
    def discard_tile(self, tile: Tile, is_riichi: bool = False, is_tsumogiri: bool = False) -> bool:
        """
        Discard a tile from hand.
        
        Args:
            tile: Tile to discard
            is_riichi: Whether this is a riichi declaration discard
            is_tsumogiri: Whether discarding the just-drawn tile
            
        Returns:
            True if successful
        """
        if self.hand.remove(tile):
            self.discards.append(tile)
            self.discard_info.append({
                "riichi": is_riichi,
                "tsumogiri": is_tsumogiri,
                "turn": len(self.discards) - 1,
            })
            
            # Update permanent furiten
            self.furiten.update_permanent(self.discards)
            
            return True
        return False
    
    def remove_tile(self, tile: Tile) -> bool:
        """Remove tile from hand without adding to discards"""
        return self.hand.remove(tile)
    
    def declare_riichi(self, is_double: bool = False) -> bool:
        """
        Declare riichi.
        
        Args:
            is_double: Whether this is double riichi (first turn)
            
        Returns:
            True if riichi was declared
        """
        if self.score < 1000:
            return False  # Can't riichi without 1000 points
        
        if not self.is_menzen:
            return False  # Must be closed hand
        
        if is_double:
            self.state = PlayerState.DOUBLE_RIICHI
        else:
            self.state = PlayerState.RIICHI
        
        self.riichi_turn = len(self.discards)
        self.ippatsu_valid = True
        self.score -= 1000  # Riichi bet
        
        return True
    
    def cancel_ippatsu(self):
        """Cancel ippatsu eligibility (called when any call is made)"""
        self.ippatsu_valid = False
    
    def declare_meld(self, meld: Meld) -> None:
        """Add a declared meld"""
        self.melds.append(meld)
        # Any call cancels ippatsu for all players
    
    def can_chii(self, tile: Tile) -> List[Tuple[Tile, Tile]]:
        """
        Check if player can call chii (sequence) on the given tile.
        Only valid from player to the left (kamicha).
        
        Returns list of possible pairs from hand that complete the chii.
        """
        # Can't call while in riichi
        if self.state in (PlayerState.RIICHI, PlayerState.DOUBLE_RIICHI):
            return []
        
        if tile.suit not in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            return []
        
        possible = []
        hand_counts = self.hand.to_count_array()
        value = tile.value
        suit_offset = tile.suit * 9
        
        # Check for value-2, value-1 (tile completes right side)
        if value >= 3:
            idx1, idx2 = suit_offset + value - 3, suit_offset + value - 2
            if hand_counts[idx1] > 0 and hand_counts[idx2] > 0:
                t1 = Tile.from_index(idx1)
                t2 = Tile.from_index(idx2)
                possible.append((t1, t2))
        
        # Check for value-1, value+1 (tile is middle)
        if 2 <= value <= 8:
            idx1, idx2 = suit_offset + value - 2, suit_offset + value
            if hand_counts[idx1] > 0 and hand_counts[idx2] > 0:
                t1 = Tile.from_index(idx1)
                t2 = Tile.from_index(idx2)
                possible.append((t1, t2))
        
        # Check for value+1, value+2 (tile completes left side)
        if value <= 7:
            idx1, idx2 = suit_offset + value, suit_offset + value + 1
            if hand_counts[idx1] > 0 and hand_counts[idx2] > 0:
                t1 = Tile.from_index(idx1)
                t2 = Tile.from_index(idx2)
                possible.append((t1, t2))
        
        return possible
    
    def can_pon(self, tile: Tile) -> bool:
        """Check if player can call pon (triplet) on the given tile"""
        # Can't call while in riichi
        if self.state in (PlayerState.RIICHI, PlayerState.DOUBLE_RIICHI):
            return False
        return self.hand.count(tile) >= 2
    
    def can_kan(self, tile: Tile) -> bool:
        """Check if player can call kan from a discard"""
        # Can call open kan even in riichi if it doesn't change waits
        # But typically not allowed - simplified here
        if self.state in (PlayerState.RIICHI, PlayerState.DOUBLE_RIICHI):
            return False
        return self.hand.count(tile) >= 3
    
    def can_ankan(self) -> List[Tile]:
        """
        Get tiles that can form concealed kan (ankan).
        In riichi, only allowed if it doesn't change waiting tiles.
        """
        counts = self.hand.to_count_array()
        kan_tiles = []
        
        for idx, count in enumerate(counts):
            if count == 4:
                # In riichi, need additional check for wait changes
                # Simplified: allow ankan in riichi
                kan_tiles.append(Tile.from_index(idx))
        
        return kan_tiles
    
    def can_shouminkan(self) -> List[Tile]:
        """
        Get tiles that can be added to existing pon (shouminkan/kakan).
        """
        # Not allowed in riichi
        if self.state in (PlayerState.RIICHI, PlayerState.DOUBLE_RIICHI):
            return []
        
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
        """Get all tiles including melds"""
        all_tiles = list(self.hand.tiles)
        for meld in self.melds:
            all_tiles.extend(meld.tiles)
        return all_tiles
    
    def get_hand_count_array(self) -> np.ndarray:
        """Get 34-element count array for hand"""
        return self.hand.to_count_array()
    
    @property
    def num_tiles_in_hand(self) -> int:
        """Number of tiles in hand"""
        return len(self.hand)
    
    @property
    def is_menzen(self) -> bool:
        """Check if hand is fully concealed (no open melds)"""
        for meld in self.melds:
            if meld.meld_type != MeldType.CONCEALED_KONG:
                if not meld.is_concealed:
                    return False
        return True
    
    @property
    def is_riichi(self) -> bool:
        """Check if player is in riichi"""
        return self.state in (PlayerState.RIICHI, PlayerState.DOUBLE_RIICHI)
    
    @property
    def is_double_riichi(self) -> bool:
        """Check if player declared double riichi"""
        return self.state == PlayerState.DOUBLE_RIICHI
    
    def reset(self, starting_score: int = 25000) -> None:
        """Reset player for a new round"""
        self.hand = TileSet()
        self.melds = []
        self.discards = []
        self.discard_info = []
        self.score = starting_score
        self.state = PlayerState.NORMAL
        self.furiten = FuritenState()
        self.riichi_turn = -1
        self.riichi_discard_index = -1
        self.ippatsu_valid = False
        self.first_draw_completed = False
    
    def reset_round(self) -> None:
        """Reset for a new round keeping score"""
        current_score = self.score
        self.reset(current_score)
    
    def copy(self) -> 'RiichiPlayer':
        """Create a deep copy"""
        new_player = RiichiPlayer(
            index=self.index,
            hand=self.hand.copy(),
            melds=list(self.melds),
            discards=list(self.discards),
            discard_info=list(self.discard_info),
            score=self.score,
            state=self.state,
            furiten=FuritenState(
                permanent=self.furiten.permanent,
                temporary=self.furiten.temporary,
                riichi_furiten=self.furiten.riichi_furiten,
                waiting_tiles=set(self.furiten.waiting_tiles),
            ),
            riichi_turn=self.riichi_turn,
            riichi_discard_index=self.riichi_discard_index,
            ippatsu_valid=self.ippatsu_valid,
            first_draw_completed=self.first_draw_completed,
        )
        return new_player
    
    def __repr__(self) -> str:
        state_str = self.state.name if self.state != PlayerState.NORMAL else ""
        return f"RiichiPlayer({self.index}, tiles={len(self.hand)}, {state_str}, {self.score}pts)"
    
    def __str__(self) -> str:
        positions = ["East", "South", "West", "North"]
        status = f" [{self.state.name}]" if self.state != PlayerState.NORMAL else ""
        furiten = " [FURITEN]" if self.furiten.is_furiten else ""
        return f"{positions[self.index]}{status}{furiten}: {self.hand} ({self.score}pts)"

