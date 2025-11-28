"""
Shanten Calculator for Riichi Mahjong

Calculates the shanten number (distance to tenpai) for a hand.
Also provides ukeire (acceptance count) for tile efficiency.

Shanten values:
- -1: Complete hand (already won)
-  0: Tenpai (one tile away from winning)
-  1: Iishanten (one away from tenpai)
-  2+: Further from tenpai
"""

from typing import List, Tuple, Set, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ShantenResult:
    """Result of shanten calculation."""
    shanten: int  # -1 = complete, 0 = tenpai, 1+ = tiles away
    waiting_tiles: List[int]  # Tile indices that complete/improve hand
    ukeire: int  # Total number of useful tiles remaining


class ShantenCalculator:
    """
    Fast shanten calculator for Riichi Mahjong.
    
    Calculates shanten for:
    - Standard form (4 melds + 1 pair)
    - Chiitoitsu (7 pairs)
    - Kokushi musou (13 orphans)
    """
    
    # Terminal and honor tile indices
    TERMINALS = [0, 8, 9, 17, 18, 26]  # 1m, 9m, 1p, 9p, 1s, 9s
    HONORS = [27, 28, 29, 30, 31, 32, 33]  # E, S, W, N, Haku, Hatsu, Chun
    KOKUSHI_TILES = TERMINALS + HONORS  # 13 unique tiles for kokushi
    
    def __init__(self):
        """Initialize the calculator."""
        self._min_shanten = 8
    
    def calculate(
        self, 
        hand_counts: np.ndarray, 
        num_melds: int = 0,
        check_chiitoitsu: bool = True,
        check_kokushi: bool = True
    ) -> ShantenResult:
        """
        Calculate shanten for a hand.
        
        Args:
            hand_counts: 34-element array of tile counts
            num_melds: Number of called melds
            check_chiitoitsu: Whether to check for 7 pairs
            check_kokushi: Whether to check for 13 orphans
            
        Returns:
            ShantenResult with shanten value and waiting tiles
        """
        # Standard form shanten
        standard_shanten = self._calculate_standard(hand_counts.copy(), num_melds)
        
        best_shanten = standard_shanten
        
        # Chiitoitsu (only possible with closed hand)
        if check_chiitoitsu and num_melds == 0:
            chiitoi_shanten = self._calculate_chiitoitsu(hand_counts)
            best_shanten = min(best_shanten, chiitoi_shanten)
        
        # Kokushi (only possible with closed hand)
        if check_kokushi and num_melds == 0:
            kokushi_shanten = self._calculate_kokushi(hand_counts)
            best_shanten = min(best_shanten, kokushi_shanten)
        
        # Calculate waiting tiles (ukeire)
        waiting_tiles, ukeire = self._calculate_ukeire(
            hand_counts, num_melds, best_shanten
        )
        
        return ShantenResult(
            shanten=best_shanten,
            waiting_tiles=waiting_tiles,
            ukeire=ukeire
        )
    
    def _calculate_standard(self, counts: np.ndarray, num_melds: int) -> int:
        """
        Calculate standard form shanten (4 melds + 1 pair).
        
        Uses recursive decomposition to find minimum shanten.
        """
        self._min_shanten = 8
        sets_needed = 4 - num_melds
        
        # Try each tile as pair head
        for i in range(34):
            if counts[i] >= 2:
                counts[i] -= 2
                self._calculate_standard_recursive(counts, sets_needed, 0, 0)
                counts[i] += 2
        
        # Also try without pair (incomplete hand)
        self._calculate_standard_recursive(counts, sets_needed, 0, 1)
        
        return self._min_shanten
    
    def _calculate_standard_recursive(
        self, 
        counts: np.ndarray, 
        sets_needed: int,
        partial_sets: int,  # Partial sets (taatsu)
        need_pair: int  # 1 if still need pair, 0 otherwise
    ):
        """Recursively find minimum shanten."""
        # Calculate current shanten
        # shanten = sets_needed - complete_sets - partial_sets + need_pair - 1
        # But we track differently: shanten = 2*sets_needed - 1 + need_pair - 2*complete - partial
        current_shanten = (sets_needed * 2) - 1 + need_pair - (partial_sets)
        
        if current_shanten >= self._min_shanten:
            return  # Pruning
        
        if sets_needed == 0:
            self._min_shanten = min(self._min_shanten, need_pair - 1 if need_pair else -1)
            return
        
        # Find first non-zero tile
        first_idx = -1
        for i in range(34):
            if counts[i] > 0:
                first_idx = i
                break
        
        if first_idx == -1:
            # No more tiles, calculate final shanten
            final_shanten = (sets_needed * 2) - 1 + need_pair - partial_sets
            self._min_shanten = min(self._min_shanten, final_shanten)
            return
        
        # Try forming a triplet
        if counts[first_idx] >= 3:
            counts[first_idx] -= 3
            self._calculate_standard_recursive(counts, sets_needed - 1, partial_sets, need_pair)
            counts[first_idx] += 3
        
        # Try forming a sequence (numbered suits only, indices 0-26)
        if first_idx < 27:
            suit_start = (first_idx // 9) * 9
            pos_in_suit = first_idx - suit_start
            
            if pos_in_suit <= 6:  # Can form sequence starting here
                idx1, idx2, idx3 = first_idx, first_idx + 1, first_idx + 2
                if counts[idx1] > 0 and counts[idx2] > 0 and counts[idx3] > 0:
                    counts[idx1] -= 1
                    counts[idx2] -= 1
                    counts[idx3] -= 1
                    self._calculate_standard_recursive(counts, sets_needed - 1, partial_sets, need_pair)
                    counts[idx1] += 1
                    counts[idx2] += 1
                    counts[idx3] += 1
        
        # Try counting as partial set (pair for taatsu)
        if counts[first_idx] >= 2 and partial_sets < sets_needed:
            counts[first_idx] -= 2
            self._calculate_standard_recursive(counts, sets_needed, partial_sets + 1, need_pair)
            counts[first_idx] += 2
        
        # Try counting as partial sequence (ryanmen, kanchan, penchan)
        if first_idx < 27 and partial_sets < sets_needed:
            suit_start = (first_idx // 9) * 9
            pos_in_suit = first_idx - suit_start
            
            # Adjacent tile (ryanmen/penchan)
            if pos_in_suit <= 7:
                idx2 = first_idx + 1
                if counts[first_idx] > 0 and counts[idx2] > 0:
                    counts[first_idx] -= 1
                    counts[idx2] -= 1
                    self._calculate_standard_recursive(counts, sets_needed, partial_sets + 1, need_pair)
                    counts[first_idx] += 1
                    counts[idx2] += 1
            
            # Skip one tile (kanchan)
            if pos_in_suit <= 6:
                idx3 = first_idx + 2
                if counts[first_idx] > 0 and counts[idx3] > 0:
                    counts[first_idx] -= 1
                    counts[idx3] -= 1
                    self._calculate_standard_recursive(counts, sets_needed, partial_sets + 1, need_pair)
                    counts[first_idx] += 1
                    counts[idx3] += 1
        
        # Try removing single tile (isolated tile)
        counts[first_idx] -= 1
        self._calculate_standard_recursive(counts, sets_needed, partial_sets, need_pair)
        counts[first_idx] += 1
    
    def _calculate_chiitoitsu(self, counts: np.ndarray) -> int:
        """
        Calculate shanten for chiitoitsu (7 pairs).
        
        Shanten = 6 - pairs + max(0, 7 - distinct_tiles)
        """
        pairs = 0
        distinct = 0
        
        for count in counts:
            if count >= 2:
                pairs += 1
            if count >= 1:
                distinct += 1
        
        # Need 7 pairs
        # If we have less than 7 distinct tiles, we need more tiles
        shanten = 6 - pairs
        if distinct < 7:
            shanten += 7 - distinct
        
        return shanten
    
    def _calculate_kokushi(self, counts: np.ndarray) -> int:
        """
        Calculate shanten for kokushi musou (13 orphans).
        
        Need one of each terminal/honor + one pair among them.
        """
        unique_count = 0
        has_pair = False
        
        for idx in self.KOKUSHI_TILES:
            if counts[idx] >= 1:
                unique_count += 1
            if counts[idx] >= 2:
                has_pair = True
        
        # Shanten = 13 - unique_count - (1 if has_pair else 0)
        shanten = 13 - unique_count - (1 if has_pair else 0)
        
        return shanten
    
    def _calculate_ukeire(
        self, 
        counts: np.ndarray,
        num_melds: int,
        current_shanten: int
    ) -> Tuple[List[int], int]:
        """
        Calculate which tiles would improve the hand (ukeire).
        
        Returns list of tile indices and total count of useful tiles.
        """
        waiting_tiles = []
        total_ukeire = 0
        
        # Max 4 of each tile
        max_counts = np.full(34, 4)
        
        for tile_idx in range(34):
            if counts[tile_idx] >= 4:
                continue  # Already have 4
            
            # Try adding this tile
            counts[tile_idx] += 1
            new_shanten = self.calculate(
                counts, num_melds,
                check_chiitoitsu=(num_melds == 0),
                check_kokushi=(num_melds == 0)
            ).shanten
            counts[tile_idx] -= 1
            
            if new_shanten < current_shanten:
                waiting_tiles.append(tile_idx)
                # Count how many of this tile are available
                available = 4 - counts[tile_idx]
                total_ukeire += available
        
        return waiting_tiles, total_ukeire
    
    def get_best_discard(
        self, 
        counts: np.ndarray,
        num_melds: int = 0
    ) -> Tuple[int, int, int]:
        """
        Find the best tile to discard for maximum ukeire.
        
        Returns: (best_tile_idx, resulting_shanten, resulting_ukeire)
        """
        best_tile = -1
        best_ukeire = -1
        best_shanten = 99
        
        for tile_idx in range(34):
            if counts[tile_idx] == 0:
                continue
            
            # Try discarding this tile
            counts[tile_idx] -= 1
            result = self.calculate(counts, num_melds)
            counts[tile_idx] += 1
            
            # Prefer lower shanten, then higher ukeire
            if (result.shanten < best_shanten or 
                (result.shanten == best_shanten and result.ukeire > best_ukeire)):
                best_tile = tile_idx
                best_shanten = result.shanten
                best_ukeire = result.ukeire
        
        return best_tile, best_shanten, best_ukeire


def calculate_shanten(hand_counts: np.ndarray, num_melds: int = 0) -> int:
    """
    Convenience function to calculate shanten.
    
    Args:
        hand_counts: 34-element array of tile counts
        num_melds: Number of called melds
        
    Returns:
        Shanten value (-1 to 8)
    """
    calc = ShantenCalculator()
    return calc.calculate(hand_counts, num_melds).shanten


def get_ukeire(hand_counts: np.ndarray, num_melds: int = 0) -> Tuple[List[int], int]:
    """
    Get tiles that would improve the hand.
    
    Returns:
        Tuple of (waiting_tile_indices, total_ukeire_count)
    """
    calc = ShantenCalculator()
    result = calc.calculate(hand_counts, num_melds)
    return result.waiting_tiles, result.ukeire


if __name__ == "__main__":
    # Test cases
    calc = ShantenCalculator()
    
    # Test 1: Tenpai hand (waiting on 1m or 4m)
    # 1112345678999m - pure straight waiting
    hand1 = np.zeros(34, dtype=np.int8)
    hand1[0] = 3  # 1m x3
    hand1[1] = 1  # 2m x1
    hand1[2] = 1  # 3m x1
    hand1[3] = 1  # 4m x1
    hand1[4] = 1  # 5m x1
    hand1[5] = 1  # 6m x1
    hand1[6] = 1  # 7m x1
    hand1[7] = 1  # 8m x1
    hand1[8] = 3  # 9m x3
    
    result1 = calc.calculate(hand1, 0)
    print(f"Test 1 (tenpai): shanten={result1.shanten}, ukeire={result1.ukeire}")
    
    # Test 2: Chiitoitsu
    # 1122334455667m 7p
    hand2 = np.zeros(34, dtype=np.int8)
    hand2[0] = 2  # 1m
    hand2[1] = 2  # 2m
    hand2[2] = 2  # 3m
    hand2[3] = 2  # 4m
    hand2[4] = 2  # 5m
    hand2[5] = 2  # 6m
    hand2[15] = 1  # 7p
    
    result2 = calc.calculate(hand2, 0)
    print(f"Test 2 (chiitoitsu iishanten): shanten={result2.shanten}")
    
    # Test 3: Kokushi
    # 19m 19p 19s ESWN HHC + 1m
    hand3 = np.zeros(34, dtype=np.int8)
    hand3[0] = 2   # 1m x2
    hand3[8] = 1   # 9m
    hand3[9] = 1   # 1p
    hand3[17] = 1  # 9p
    hand3[18] = 1  # 1s
    hand3[26] = 1  # 9s
    hand3[27] = 1  # East
    hand3[28] = 1  # South
    hand3[29] = 1  # West
    hand3[30] = 1  # North
    hand3[31] = 1  # Haku
    hand3[32] = 1  # Hatsu
    hand3[33] = 1  # Chun
    
    result3 = calc.calculate(hand3, 0)
    print(f"Test 3 (kokushi tenpai): shanten={result3.shanten}")
    
    print("\nAll tests completed!")

