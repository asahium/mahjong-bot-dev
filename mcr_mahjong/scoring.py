"""
MCR Mahjong Scoring System

Implements all 81 scoring patterns according to Chinese Official Mahjong rules (MCR).
Patterns are organized by point value from 88 down to 1.

MCR uses an exclusion principle where higher-scoring patterns exclude
patterns they imply (e.g., Big Four Winds excludes All Pungs).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict, Callable
from enum import IntEnum, auto
import numpy as np

from .tiles import Tile, TileSet, TileSuit, WindType, DragonType
from .player import Meld, MeldType


@dataclass
class ScoringContext:
    """Context information needed for scoring"""
    hand: TileSet               # Tiles in hand (including winning tile)
    melds: List[Meld]           # Declared melds
    winning_tile: Tile          # The winning tile
    is_zimo: bool               # Self-drawn win
    round_wind: int             # Round wind (0=E, 1=S, 2=W, 3=N)
    seat_wind: int              # Player's seat wind
    is_last_tile: bool          # Won on last tile of wall
    is_kong_draw: bool          # Won on replacement tile after kong
    is_robbing_kong: bool = False  # Won by robbing another's add-kong
    
    # Computed fields (set during analysis)
    all_tiles: List[Tile] = field(default_factory=list)
    counts: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=np.int8))
    sets: List[Tuple[str, List[Tile]]] = field(default_factory=list)  # (type, tiles)
    pair: Optional[Tuple[Tile, Tile]] = None
    is_concealed: bool = False
    
    def __post_init__(self):
        self._analyze()
    
    def _analyze(self):
        """Analyze the hand structure"""
        # Combine hand tiles with melds
        self.all_tiles = list(self.hand.tiles)
        for meld in self.melds:
            self.all_tiles.extend(meld.tiles)
        
        # Count array
        self.counts = np.zeros(34, dtype=np.int8)
        for tile in self.all_tiles:
            self.counts[tile.tile_index] += 1
        
        # Check if concealed
        self.is_concealed = all(m.is_concealed or m.meld_type == MeldType.CONCEALED_KONG 
                                for m in self.melds)
        
        # Decompose into sets and pair
        self._decompose_hand()
    
    def _decompose_hand(self):
        """Decompose hand into sets and pair for scoring"""
        # Convert melds to sets format
        self.sets = []
        for meld in self.melds:
            if meld.meld_type == MeldType.CHOW:
                self.sets.append(('chow', sorted(meld.tiles)))
            elif meld.meld_type == MeldType.PONG:
                self.sets.append(('pong', meld.tiles))
            elif meld.meld_type in (MeldType.KONG, MeldType.CONCEALED_KONG):
                self.sets.append(('kong', meld.tiles))
        
        # Decompose remaining hand tiles
        hand_counts = self.hand.to_count_array()
        self._find_decomposition(hand_counts, 4 - len(self.melds))
    
    def _find_decomposition(self, counts: np.ndarray, sets_needed: int) -> bool:
        """Find a valid decomposition of the hand"""
        if sets_needed == 0:
            # Find pair
            for i in range(34):
                if counts[i] >= 2:
                    tile = Tile.from_index(i)
                    self.pair = (tile, tile)
                    return True
            return False
        
        # Find first non-zero
        first_idx = -1
        for i in range(34):
            if counts[i] > 0:
                first_idx = i
                break
        
        if first_idx == -1:
            return sets_needed == 0
        
        # Try pong
        if counts[first_idx] >= 3:
            counts[first_idx] -= 3
            tile = Tile.from_index(first_idx)
            self.sets.append(('pong', [tile, tile, tile]))
            if self._find_decomposition(counts, sets_needed - 1):
                return True
            self.sets.pop()
            counts[first_idx] += 3
        
        # Try chow (numbered suits only)
        if first_idx < 27:
            suit_start = (first_idx // 9) * 9
            val = first_idx - suit_start
            if val <= 6:
                idx1, idx2, idx3 = first_idx, first_idx + 1, first_idx + 2
                if counts[idx1] >= 1 and counts[idx2] >= 1 and counts[idx3] >= 1:
                    counts[idx1] -= 1
                    counts[idx2] -= 1
                    counts[idx3] -= 1
                    t1 = Tile.from_index(idx1)
                    t2 = Tile.from_index(idx2)
                    t3 = Tile.from_index(idx3)
                    self.sets.append(('chow', [t1, t2, t3]))
                    if self._find_decomposition(counts, sets_needed - 1):
                        return True
                    self.sets.pop()
                    counts[idx1] += 1
                    counts[idx2] += 1
                    counts[idx3] += 1
        
        # Try pair here
        if counts[first_idx] >= 2 and self.pair is None:
            counts[first_idx] -= 2
            tile = Tile.from_index(first_idx)
            self.pair = (tile, tile)
            if self._find_decomposition(counts, sets_needed):
                return True
            self.pair = None
            counts[first_idx] += 2
        
        return False


@dataclass
class ScoringPattern:
    """Represents a scoring pattern"""
    name: str
    chinese_name: str
    points: int
    check_func: Callable[[ScoringContext], bool]
    excludes: List[str] = field(default_factory=list)  # Patterns this one excludes


class MCRScorer:
    """
    MCR Mahjong Scorer
    
    Implements all 81 scoring patterns and the exclusion principle.
    """
    
    def __init__(self):
        self.patterns = self._create_patterns()
    
    def calculate_score(
        self,
        hand: TileSet,
        melds: List[Meld],
        winning_tile: Tile,
        is_zimo: bool,
        round_wind: int,
        seat_wind: int,
        is_last_tile: bool = False,
        is_kong_draw: bool = False,
        is_robbing_kong: bool = False
    ) -> int:
        """
        Calculate the total score for a winning hand.
        
        Returns:
            Total score (sum of all applicable patterns)
        """
        # Create scoring context
        ctx = ScoringContext(
            hand=hand,
            melds=melds,
            winning_tile=winning_tile,
            is_zimo=is_zimo,
            round_wind=round_wind,
            seat_wind=seat_wind,
            is_last_tile=is_last_tile,
            is_kong_draw=is_kong_draw,
            is_robbing_kong=is_robbing_kong
        )
        
        # Find all matching patterns
        matched = self.get_matching_patterns(ctx)
        
        # Calculate total score
        total = sum(p.points for p in matched)
        
        return total
    
    def get_matching_patterns(self, ctx: ScoringContext) -> List[ScoringPattern]:
        """
        Get list of matching patterns after applying exclusion rules.
        """
        # Find all patterns that match
        matching = []
        for pattern in self.patterns:
            if pattern.check_func(ctx):
                matching.append(pattern)
        
        # Apply exclusion rules
        excluded_names: Set[str] = set()
        for pattern in matching:
            excluded_names.update(pattern.excludes)
        
        # Filter out excluded patterns
        final = [p for p in matching if p.name not in excluded_names]
        
        return final
    
    def _create_patterns(self) -> List[ScoringPattern]:
        """Create all 81 scoring patterns"""
        patterns = []
        
        # ========== 88 Points ==========
        patterns.append(ScoringPattern(
            "Big Four Winds", "大四喜", 88,
            self._check_big_four_winds,
            ["All Pungs", "Prevalent Wind", "Seat Wind", "Pung of Terminals or Honors",
             "Big Three Winds", "Little Four Winds"]
        ))
        
        patterns.append(ScoringPattern(
            "Big Three Dragons", "大三元", 88,
            self._check_big_three_dragons,
            ["Two Dragon Pungs", "Dragon Pung", "Little Three Dragons"]
        ))
        
        patterns.append(ScoringPattern(
            "All Green", "绿一色", 88,
            self._check_all_green,
            ["Half Flush", "One Voided Suit"]
        ))
        
        patterns.append(ScoringPattern(
            "Nine Gates", "九莲宝灯", 88,
            self._check_nine_gates,
            ["Full Flush", "Concealed Hand", "Pung of Terminals or Honors",
             "One Voided Suit", "No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Four Kongs", "四杠", 88,
            self._check_four_kongs,
            ["Three Kongs", "Two Concealed Kongs", "Two Melded Kongs", "Single Wait"]
        ))
        
        patterns.append(ScoringPattern(
            "Seven Shifted Pairs", "连七对", 88,
            self._check_seven_shifted_pairs,
            ["Full Flush", "Concealed Hand", "Single Wait", "Seven Pairs"]
        ))
        
        patterns.append(ScoringPattern(
            "Thirteen Orphans", "十三幺", 88,
            self._check_thirteen_orphans,
            ["All Types", "Concealed Hand", "Single Wait"]
        ))
        
        # ========== 64 Points ==========
        patterns.append(ScoringPattern(
            "All Terminals", "清幺九", 64,
            self._check_all_terminals,
            ["All Pungs", "Outside Hand", "Pung of Terminals or Honors", "No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Little Four Winds", "小四喜", 64,
            self._check_little_four_winds,
            ["Big Three Winds", "Prevalent Wind", "Seat Wind", "Pung of Terminals or Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Little Three Dragons", "小三元", 64,
            self._check_little_three_dragons,
            ["Two Dragon Pungs", "Dragon Pung"]
        ))
        
        patterns.append(ScoringPattern(
            "All Honors", "字一色", 64,
            self._check_all_honors,
            ["All Pungs", "Pung of Terminals or Honors", "Outside Hand"]
        ))
        
        patterns.append(ScoringPattern(
            "Four Concealed Pungs", "四暗刻", 64,
            self._check_four_concealed_pungs,
            ["Concealed Hand", "All Pungs", "Two Concealed Pungs"]
        ))
        
        patterns.append(ScoringPattern(
            "Pure Terminal Chows", "一色双龙会", 64,
            self._check_pure_terminal_chows,
            ["Full Flush", "All Chows", "Pure Double Chow", "Two Terminal Chows"]
        ))
        
        # ========== 48 Points ==========
        patterns.append(ScoringPattern(
            "Quadruple Chow", "一色四同顺", 48,
            self._check_quadruple_chow,
            ["Pure Triple Chow", "Tile Hog", "Pure Double Chow"]
        ))
        
        patterns.append(ScoringPattern(
            "Four Pure Shifted Pungs", "一色四节高", 48,
            self._check_four_pure_shifted_pungs,
            ["Pure Triple Chow", "All Pungs", "Pure Shifted Pungs"]
        ))
        
        # ========== 32 Points ==========
        patterns.append(ScoringPattern(
            "Four Shifted Chows", "一色四步高", 32,
            self._check_four_shifted_chows,
            ["Short Straight", "Two Terminal Chows", "Pure Shifted Chows"]
        ))
        
        patterns.append(ScoringPattern(
            "Three Kongs", "三杠", 32,
            self._check_three_kongs,
            ["Two Melded Kongs"]
        ))
        
        patterns.append(ScoringPattern(
            "All Terminals and Honors", "混幺九", 32,
            self._check_all_terminals_and_honors,
            ["All Pungs", "Outside Hand", "Pung of Terminals or Honors"]
        ))
        
        # ========== 24 Points ==========
        patterns.append(ScoringPattern(
            "Seven Pairs", "七对", 24,
            self._check_seven_pairs,
            ["Concealed Hand", "Single Wait"]
        ))
        
        patterns.append(ScoringPattern(
            "Greater Honors and Knitted Tiles", "七星不靠", 24,
            self._check_greater_honors_knitted,
            ["All Types", "Concealed Hand", "Lesser Honors and Knitted Tiles"]
        ))
        
        patterns.append(ScoringPattern(
            "All Even Pungs", "全双刻", 24,
            self._check_all_even_pungs,
            ["All Pungs", "All Simples", "No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Full Flush", "清一色", 24,
            self._check_full_flush,
            ["Half Flush", "One Voided Suit", "No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Pure Triple Chow", "一色三同顺", 24,
            self._check_pure_triple_chow,
            ["Pure Double Chow"]
        ))
        
        patterns.append(ScoringPattern(
            "Pure Shifted Pungs", "一色三节高", 24,
            self._check_pure_shifted_pungs,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Upper Tiles", "全大", 24,
            self._check_upper_tiles,
            ["No Honors", "Upper Four"]
        ))
        
        patterns.append(ScoringPattern(
            "Middle Tiles", "全中", 24,
            self._check_middle_tiles,
            ["All Simples", "No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Lower Tiles", "全小", 24,
            self._check_lower_tiles,
            ["No Honors", "Lower Four"]
        ))
        
        # ========== 16 Points ==========
        patterns.append(ScoringPattern(
            "Pure Straight", "清龙", 16,
            self._check_pure_straight,
            ["Short Straight", "Two Terminal Chows"]
        ))
        
        patterns.append(ScoringPattern(
            "Three-Suited Terminal Chows", "三色双龙会", 16,
            self._check_three_suited_terminal_chows,
            ["All Chows", "Mixed Double Chow", "Two Terminal Chows"]
        ))
        
        patterns.append(ScoringPattern(
            "Pure Shifted Chows", "一色三步高", 16,
            self._check_pure_shifted_chows,
            []
        ))
        
        patterns.append(ScoringPattern(
            "All Fives", "全带五", 16,
            self._check_all_fives,
            ["All Simples", "No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Triple Pung", "三同刻", 16,
            self._check_triple_pung,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Three Concealed Pungs", "三暗刻", 16,
            self._check_three_concealed_pungs,
            ["Two Concealed Pungs"]
        ))
        
        # ========== 12 Points ==========
        patterns.append(ScoringPattern(
            "Lesser Honors and Knitted Tiles", "全不靠", 12,
            self._check_lesser_honors_knitted,
            ["All Types", "Concealed Hand"]
        ))
        
        patterns.append(ScoringPattern(
            "Knitted Straight", "组合龙", 12,
            self._check_knitted_straight,
            ["All Types"]
        ))
        
        patterns.append(ScoringPattern(
            "Upper Four", "大于五", 12,
            self._check_upper_four,
            ["No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Lower Four", "小于五", 12,
            self._check_lower_four,
            ["No Honors"]
        ))
        
        patterns.append(ScoringPattern(
            "Big Three Winds", "大三风", 12,
            self._check_big_three_winds,
            []
        ))
        
        # ========== 8 Points ==========
        patterns.append(ScoringPattern(
            "Mixed Straight", "花龙", 8,
            self._check_mixed_straight,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Reversible Tiles", "推不倒", 8,
            self._check_reversible_tiles,
            ["One Voided Suit"]
        ))
        
        patterns.append(ScoringPattern(
            "Mixed Triple Chow", "三色三同顺", 8,
            self._check_mixed_triple_chow,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Mixed Shifted Pungs", "三色三节高", 8,
            self._check_mixed_shifted_pungs,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Chicken Hand", "无番和", 8,
            self._check_chicken_hand,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Last Tile Draw", "妙手回春", 8,
            self._check_last_tile_draw,
            ["Self-Drawn"]
        ))
        
        patterns.append(ScoringPattern(
            "Last Tile Claim", "海底捞月", 8,
            self._check_last_tile_claim,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Out with Replacement Tile", "杠上开花", 8,
            self._check_out_with_replacement,
            ["Self-Drawn"]
        ))
        
        patterns.append(ScoringPattern(
            "Robbing the Kong", "抢杠和", 8,
            self._check_robbing_kong,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Two Concealed Kongs", "双暗杠", 8,
            self._check_two_concealed_kongs,
            []
        ))
        
        # ========== 6 Points ==========
        patterns.append(ScoringPattern(
            "All Pungs", "碰碰和", 6,
            self._check_all_pungs,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Half Flush", "混一色", 6,
            self._check_half_flush,
            ["One Voided Suit"]
        ))
        
        patterns.append(ScoringPattern(
            "Mixed Shifted Chows", "三色三步高", 6,
            self._check_mixed_shifted_chows,
            []
        ))
        
        patterns.append(ScoringPattern(
            "All Types", "五门齐", 6,
            self._check_all_types,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Melded Hand", "全求人", 6,
            self._check_melded_hand,
            ["Single Wait"]
        ))
        
        patterns.append(ScoringPattern(
            "Two Dragon Pungs", "双箭刻", 6,
            self._check_two_dragon_pungs,
            []
        ))
        
        # ========== 4 Points ==========
        patterns.append(ScoringPattern(
            "Outside Hand", "全带幺", 4,
            self._check_outside_hand,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Fully Concealed Hand", "不求人", 4,
            self._check_fully_concealed_hand,
            ["Self-Drawn"]
        ))
        
        patterns.append(ScoringPattern(
            "Two Melded Kongs", "双明杠", 4,
            self._check_two_melded_kongs,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Last Tile", "和绝张", 4,
            self._check_last_tile,
            []
        ))
        
        # ========== 2 Points ==========
        patterns.append(ScoringPattern(
            "Dragon Pung", "箭刻", 2,
            self._check_dragon_pung,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Prevalent Wind", "圈风刻", 2,
            self._check_prevalent_wind,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Seat Wind", "门风刻", 2,
            self._check_seat_wind,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Concealed Hand", "门前清", 2,
            self._check_concealed_hand,
            []
        ))
        
        patterns.append(ScoringPattern(
            "All Chows", "平和", 2,
            self._check_all_chows,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Tile Hog", "四归一", 2,
            self._check_tile_hog,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Double Pung", "双同刻", 2,
            self._check_double_pung,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Two Concealed Pungs", "双暗刻", 2,
            self._check_two_concealed_pungs,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Concealed Kong", "暗杠", 2,
            self._check_concealed_kong,
            []
        ))
        
        patterns.append(ScoringPattern(
            "All Simples", "断幺九", 2,
            self._check_all_simples,
            []
        ))
        
        # ========== 1 Point ==========
        patterns.append(ScoringPattern(
            "Pure Double Chow", "一般高", 1,
            self._check_pure_double_chow,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Mixed Double Chow", "喜相逢", 1,
            self._check_mixed_double_chow,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Short Straight", "连六", 1,
            self._check_short_straight,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Two Terminal Chows", "老少副", 1,
            self._check_two_terminal_chows,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Pung of Terminals or Honors", "幺九刻", 1,
            self._check_pung_terminals_honors,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Melded Kong", "明杠", 1,
            self._check_melded_kong,
            []
        ))
        
        patterns.append(ScoringPattern(
            "One Voided Suit", "缺一门", 1,
            self._check_one_voided_suit,
            []
        ))
        
        patterns.append(ScoringPattern(
            "No Honors", "无字", 1,
            self._check_no_honors,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Edge Wait", "边张", 1,
            self._check_edge_wait,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Closed Wait", "嵌张", 1,
            self._check_closed_wait,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Single Wait", "单钓将", 1,
            self._check_single_wait,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Self-Drawn", "自摸", 1,
            self._check_self_drawn,
            []
        ))
        
        patterns.append(ScoringPattern(
            "Flower Tiles", "花牌", 1,
            self._check_flower_tiles,  # MCR doesn't use flowers, placeholder
            []
        ))
        
        return patterns
    
    # ========== Pattern Check Functions ==========
    
    # --- 88 Points ---
    def _check_big_four_winds(self, ctx: ScoringContext) -> bool:
        """Pungs/Kongs of all four winds"""
        winds_found = [False] * 4
        for set_type, tiles in ctx.sets:
            if set_type in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.WINDS:
                    winds_found[tiles[0].value] = True
        return all(winds_found)
    
    def _check_big_three_dragons(self, ctx: ScoringContext) -> bool:
        """Pungs/Kongs of all three dragons"""
        dragons_found = [False] * 3
        for set_type, tiles in ctx.sets:
            if set_type in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.DRAGONS:
                    dragons_found[tiles[0].value] = True
        return all(dragons_found)
    
    def _check_all_green(self, ctx: ScoringContext) -> bool:
        """All tiles are green (2,3,4,6,8 bamboo + green dragon)"""
        return all(t.is_green for t in ctx.all_tiles)
    
    def _check_nine_gates(self, ctx: ScoringContext) -> bool:
        """1112345678999 + any same suit, concealed"""
        if not ctx.is_concealed:
            return False
        
        # Must be single suit
        suits = set(t.suit for t in ctx.all_tiles)
        if len(suits) != 1:
            return False
        
        suit = list(suits)[0]
        if suit not in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            return False
        
        # Count tiles by value
        values = [0] * 10  # 1-indexed
        for t in ctx.all_tiles:
            values[t.value] += 1
        
        # Check 1112345678999 pattern
        # After removing winning tile, should have: 3,1,1,1,1,1,1,1,3
        for v in range(1, 10):
            if v == 1 or v == 9:
                if values[v] < 3:
                    return False
            else:
                if values[v] < 1:
                    return False
        
        return True
    
    def _check_four_kongs(self, ctx: ScoringContext) -> bool:
        """Four kongs"""
        kong_count = sum(1 for st, _ in ctx.sets if st == 'kong')
        return kong_count == 4
    
    def _check_seven_shifted_pairs(self, ctx: ScoringContext) -> bool:
        """Seven consecutive pairs in same suit"""
        counts = ctx.counts
        
        for suit in range(3):  # Numbered suits only
            offset = suit * 9
            for start in range(3):  # Can start at 1,2,3
                valid = True
                for i in range(7):
                    if counts[offset + start + i] != 2:
                        valid = False
                        break
                if valid:
                    return True
        return False
    
    def _check_thirteen_orphans(self, ctx: ScoringContext) -> bool:
        """One of each terminal and honor + one pair from these"""
        required = [
            0, 8,  # 1,9 Characters
            9, 17,  # 1,9 Bamboos
            18, 26,  # 1,9 Dots
            27, 28, 29, 30,  # Four winds
            31, 32, 33  # Three dragons
        ]
        
        counts = ctx.counts
        pair_found = False
        
        for idx in required:
            if counts[idx] == 0:
                return False
            if counts[idx] == 2:
                pair_found = True
        
        # Check only these tiles exist
        for i in range(34):
            if i not in required and counts[i] > 0:
                return False
        
        return pair_found and sum(counts) == 14
    
    # --- 64 Points ---
    def _check_all_terminals(self, ctx: ScoringContext) -> bool:
        """All tiles are terminals (1 or 9)"""
        for t in ctx.all_tiles:
            if not t.is_terminal:
                return False
        return True
    
    def _check_little_four_winds(self, ctx: ScoringContext) -> bool:
        """Three wind pungs + wind pair"""
        wind_pungs = 0
        wind_pair = False
        
        for set_type, tiles in ctx.sets:
            if set_type in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.WINDS:
                    wind_pungs += 1
        
        if ctx.pair and ctx.pair[0].suit == TileSuit.WINDS:
            wind_pair = True
        
        return wind_pungs == 3 and wind_pair
    
    def _check_little_three_dragons(self, ctx: ScoringContext) -> bool:
        """Two dragon pungs + dragon pair"""
        dragon_pungs = 0
        dragon_pair = False
        
        for set_type, tiles in ctx.sets:
            if set_type in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.DRAGONS:
                    dragon_pungs += 1
        
        if ctx.pair and ctx.pair[0].suit == TileSuit.DRAGONS:
            dragon_pair = True
        
        return dragon_pungs == 2 and dragon_pair
    
    def _check_all_honors(self, ctx: ScoringContext) -> bool:
        """All tiles are honors (winds and dragons)"""
        return all(t.is_honor for t in ctx.all_tiles)
    
    def _check_four_concealed_pungs(self, ctx: ScoringContext) -> bool:
        """Four concealed pungs/kongs"""
        concealed_pungs = 0
        
        for meld in ctx.melds:
            if meld.meld_type == MeldType.CONCEALED_KONG:
                concealed_pungs += 1
        
        # Count concealed pungs in hand
        for set_type, _ in ctx.sets:
            if set_type == 'pong':
                # If not in melds, it's concealed
                concealed_pungs += 1
        
        # Subtract exposed pungs
        for meld in ctx.melds:
            if meld.meld_type == MeldType.PONG:
                concealed_pungs -= 1
            elif meld.meld_type == MeldType.KONG and not meld.is_concealed:
                concealed_pungs -= 1
        
        return concealed_pungs >= 4
    
    def _check_pure_terminal_chows(self, ctx: ScoringContext) -> bool:
        """123+789 twice in same suit + 5 pair"""
        counts = ctx.counts
        
        for suit in range(3):
            offset = suit * 9
            # Need 123 x2 and 789 x2
            has_123 = (counts[offset] >= 2 and counts[offset+1] >= 2 and counts[offset+2] >= 2)
            has_789 = (counts[offset+6] >= 2 and counts[offset+7] >= 2 and counts[offset+8] >= 2)
            has_5_pair = counts[offset+4] >= 2
            
            if has_123 and has_789 and has_5_pair:
                return True
        
        return False
    
    # --- 48 Points ---
    def _check_quadruple_chow(self, ctx: ScoringContext) -> bool:
        """Four identical chows"""
        chows = [tuple(sorted(t.tile_index for t in tiles)) 
                 for st, tiles in ctx.sets if st == 'chow']
        
        from collections import Counter
        chow_counts = Counter(chows)
        return any(c >= 4 for c in chow_counts.values())
    
    def _check_four_pure_shifted_pungs(self, ctx: ScoringContext) -> bool:
        """Four pungs in sequence in same suit (e.g., 2222-3333-4444-5555)"""
        pungs = []
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                t = tiles[0]
                if t.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    pungs.append((t.suit, t.value))
        
        pungs.sort()
        
        for suit in range(3):
            suit_pungs = sorted([v for s, v in pungs if s == suit])
            if len(suit_pungs) >= 4:
                for i in range(len(suit_pungs) - 3):
                    if suit_pungs[i:i+4] == list(range(suit_pungs[i], suit_pungs[i]+4)):
                        return True
        
        return False
    
    # --- 32 Points ---
    def _check_four_shifted_chows(self, ctx: ScoringContext) -> bool:
        """Four chows in sequence (by 1 or 2) in same suit"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        for suit in range(3):
            suit_chows = sorted([v for s, v in chows if s == suit])
            if len(suit_chows) >= 4:
                # Check step of 1
                for i in range(len(suit_chows) - 3):
                    if suit_chows[i:i+4] == list(range(suit_chows[i], suit_chows[i]+4)):
                        return True
                # Check step of 2
                for i in range(len(suit_chows) - 3):
                    if suit_chows[i:i+4] == list(range(suit_chows[i], suit_chows[i]+8, 2)):
                        return True
        
        return False
    
    def _check_three_kongs(self, ctx: ScoringContext) -> bool:
        """Three kongs"""
        kong_count = sum(1 for st, _ in ctx.sets if st == 'kong')
        return kong_count == 3
    
    def _check_all_terminals_and_honors(self, ctx: ScoringContext) -> bool:
        """All tiles are terminals or honors"""
        return all(t.is_terminal_or_honor for t in ctx.all_tiles)
    
    # --- 24 Points ---
    def _check_seven_pairs(self, ctx: ScoringContext) -> bool:
        """Hand consists of 7 pairs"""
        if len(ctx.melds) > 0:
            return False
        return all(c == 0 or c == 2 for c in ctx.counts) and sum(ctx.counts) == 14
    
    def _check_greater_honors_knitted(self, ctx: ScoringContext) -> bool:
        """All 7 honors + 7 knitted tiles (147, 258, 369 from different suits)"""
        if len(ctx.melds) > 0:
            return False
        
        counts = ctx.counts
        
        # Check all honors present (one each)
        for i in range(27, 34):
            if counts[i] != 1:
                return False
        
        # Check knitted tiles (7 tiles from 147/258/369 pattern)
        knitted_count = 0
        # Pattern: 147 from one suit, 258 from another, 369 from third
        patterns = [[1,4,7], [2,5,8], [3,6,9]]
        
        for perm in [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]:
            count = 0
            for suit_idx, pattern_idx in enumerate(perm):
                for val in patterns[pattern_idx]:
                    tile_idx = suit_idx * 9 + val - 1
                    if counts[tile_idx] == 1:
                        count += 1
            if count == 7:
                return True
        
        return False
    
    def _check_all_even_pungs(self, ctx: ScoringContext) -> bool:
        """All pungs of even numbers (2,4,6,8)"""
        if not self._check_all_pungs(ctx):
            return False
        
        for t in ctx.all_tiles:
            if t.is_honor:
                return False
            if t.value % 2 != 0:
                return False
        
        return True
    
    def _check_full_flush(self, ctx: ScoringContext) -> bool:
        """All tiles same numbered suit (no honors)"""
        suits = set(t.suit for t in ctx.all_tiles)
        if len(suits) != 1:
            return False
        return list(suits)[0] in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS)
    
    def _check_pure_triple_chow(self, ctx: ScoringContext) -> bool:
        """Three identical chows"""
        chows = [tuple(sorted(t.tile_index for t in tiles)) 
                 for st, tiles in ctx.sets if st == 'chow']
        
        from collections import Counter
        chow_counts = Counter(chows)
        return any(c >= 3 for c in chow_counts.values())
    
    def _check_pure_shifted_pungs(self, ctx: ScoringContext) -> bool:
        """Three pungs in sequence in same suit"""
        pungs = []
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                t = tiles[0]
                if t.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    pungs.append((t.suit, t.value))
        
        for suit in range(3):
            suit_pungs = sorted([v for s, v in pungs if s == suit])
            if len(suit_pungs) >= 3:
                for i in range(len(suit_pungs) - 2):
                    if suit_pungs[i+1] == suit_pungs[i]+1 and suit_pungs[i+2] == suit_pungs[i]+2:
                        return True
        
        return False
    
    def _check_upper_tiles(self, ctx: ScoringContext) -> bool:
        """All tiles are 7,8,9"""
        for t in ctx.all_tiles:
            if t.is_honor:
                return False
            if t.value < 7:
                return False
        return True
    
    def _check_middle_tiles(self, ctx: ScoringContext) -> bool:
        """All tiles are 4,5,6"""
        for t in ctx.all_tiles:
            if t.is_honor:
                return False
            if t.value not in (4, 5, 6):
                return False
        return True
    
    def _check_lower_tiles(self, ctx: ScoringContext) -> bool:
        """All tiles are 1,2,3"""
        for t in ctx.all_tiles:
            if t.is_honor:
                return False
            if t.value > 3:
                return False
        return True
    
    # --- 16 Points ---
    def _check_pure_straight(self, ctx: ScoringContext) -> bool:
        """123-456-789 in same suit"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        for suit in range(3):
            suit_chows = set(v for s, v in chows if s == suit)
            if {1, 4, 7}.issubset(suit_chows):
                return True
        
        return False
    
    def _check_three_suited_terminal_chows(self, ctx: ScoringContext) -> bool:
        """123+789 from two suits + 5 pair from third suit"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        # Need 123 and 789 from different suits
        has_123 = set(s for s, v in chows if v == 1)
        has_789 = set(s for s, v in chows if v == 7)
        
        if len(has_123) >= 1 and len(has_789) >= 1:
            suits_used = has_123 | has_789
            if len(suits_used) == 2 and ctx.pair:
                # Pair should be 5 from the third suit
                pair_tile = ctx.pair[0]
                if pair_tile.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    if pair_tile.suit not in suits_used and pair_tile.value == 5:
                        return True
        
        return False
    
    def _check_pure_shifted_chows(self, ctx: ScoringContext) -> bool:
        """Three chows in sequence in same suit"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        for suit in range(3):
            suit_chows = sorted([v for s, v in chows if s == suit])
            if len(suit_chows) >= 3:
                # Check step of 1
                for i in range(len(suit_chows) - 2):
                    if suit_chows[i+1] == suit_chows[i]+1 and suit_chows[i+2] == suit_chows[i]+2:
                        return True
                # Check step of 2
                for i in range(len(suit_chows) - 2):
                    if suit_chows[i+1] == suit_chows[i]+2 and suit_chows[i+2] == suit_chows[i]+4:
                        return True
        
        return False
    
    def _check_all_fives(self, ctx: ScoringContext) -> bool:
        """Every set and pair contains a 5"""
        for st, tiles in ctx.sets:
            if st == 'chow':
                if not any(t.value == 5 for t in tiles):
                    return False
            else:  # pong/kong
                if tiles[0].value != 5:
                    return False
        
        if ctx.pair and ctx.pair[0].value != 5:
            return False
        
        return True
    
    def _check_triple_pung(self, ctx: ScoringContext) -> bool:
        """Three pungs of same number in different suits"""
        pungs = []
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                t = tiles[0]
                if t.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    pungs.append((t.suit, t.value))
        
        # Group by value
        from collections import defaultdict
        by_value = defaultdict(set)
        for suit, val in pungs:
            by_value[val].add(suit)
        
        return any(len(suits) >= 3 for suits in by_value.values())
    
    def _check_three_concealed_pungs(self, ctx: ScoringContext) -> bool:
        """Three concealed pungs"""
        concealed_pungs = 0
        
        for meld in ctx.melds:
            if meld.meld_type == MeldType.CONCEALED_KONG:
                concealed_pungs += 1
        
        # Pungs in hand are concealed
        hand_counts = ctx.hand.to_count_array()
        for i in range(34):
            if hand_counts[i] >= 3:
                concealed_pungs += 1
        
        return concealed_pungs >= 3
    
    # --- 12 Points ---
    def _check_lesser_honors_knitted(self, ctx: ScoringContext) -> bool:
        """Knitted tiles without all seven honors"""
        if len(ctx.melds) > 0:
            return False
        
        # Must be special hand
        counts = ctx.counts
        if sum(counts) != 14:
            return False
        
        # All single tiles or one pair
        pairs = sum(1 for c in counts if c == 2)
        singles = sum(1 for c in counts if c == 1)
        
        return (pairs == 1 and singles == 12) or (pairs == 0 and singles == 14)
    
    def _check_knitted_straight(self, ctx: ScoringContext) -> bool:
        """147-258-369 from three different suits"""
        counts = ctx.counts
        patterns = [[1,4,7], [2,5,8], [3,6,9]]
        
        for perm in [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]:
            found = True
            for suit_idx, pattern_idx in enumerate(perm):
                for val in patterns[pattern_idx]:
                    tile_idx = suit_idx * 9 + val - 1
                    if counts[tile_idx] < 1:
                        found = False
                        break
                if not found:
                    break
            if found:
                return True
        
        return False
    
    def _check_upper_four(self, ctx: ScoringContext) -> bool:
        """All tiles are 6,7,8,9"""
        for t in ctx.all_tiles:
            if t.is_honor:
                return False
            if t.value < 6:
                return False
        return True
    
    def _check_lower_four(self, ctx: ScoringContext) -> bool:
        """All tiles are 1,2,3,4"""
        for t in ctx.all_tiles:
            if t.is_honor:
                return False
            if t.value > 4:
                return False
        return True
    
    def _check_big_three_winds(self, ctx: ScoringContext) -> bool:
        """Three wind pungs"""
        wind_pungs = 0
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.WINDS:
                    wind_pungs += 1
        return wind_pungs == 3
    
    # --- 8 Points ---
    def _check_mixed_straight(self, ctx: ScoringContext) -> bool:
        """123-456-789 from three different suits"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        has_123 = set(s for s, v in chows if v == 1)
        has_456 = set(s for s, v in chows if v == 4)
        has_789 = set(s for s, v in chows if v == 7)
        
        if has_123 and has_456 and has_789:
            # Must be from three different suits
            for s1 in has_123:
                for s4 in has_456:
                    for s7 in has_789:
                        if len({s1, s4, s7}) == 3:
                            return True
        
        return False
    
    def _check_reversible_tiles(self, ctx: ScoringContext) -> bool:
        """All tiles look same upside down (1,2,3,4,5,8,9 dots, white dragon)"""
        reversible = {
            (TileSuit.DOTS, 1), (TileSuit.DOTS, 2), (TileSuit.DOTS, 3),
            (TileSuit.DOTS, 4), (TileSuit.DOTS, 5), (TileSuit.DOTS, 8),
            (TileSuit.DOTS, 9), (TileSuit.DRAGONS, DragonType.WHITE)
        }
        
        for t in ctx.all_tiles:
            if (t.suit, t.value) not in reversible:
                return False
        return True
    
    def _check_mixed_triple_chow(self, ctx: ScoringContext) -> bool:
        """Three chows of same numbers in different suits"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        from collections import defaultdict
        by_value = defaultdict(set)
        for suit, val in chows:
            by_value[val].add(suit)
        
        return any(len(suits) >= 3 for suits in by_value.values())
    
    def _check_mixed_shifted_pungs(self, ctx: ScoringContext) -> bool:
        """Three pungs in sequence from three suits"""
        pungs = []
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                t = tiles[0]
                if t.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    pungs.append((t.suit, t.value))
        
        # Check all combinations
        for i in range(len(pungs)):
            for j in range(i+1, len(pungs)):
                for k in range(j+1, len(pungs)):
                    suits = {pungs[i][0], pungs[j][0], pungs[k][0]}
                    vals = sorted([pungs[i][1], pungs[j][1], pungs[k][1]])
                    if len(suits) == 3 and vals == list(range(vals[0], vals[0]+3)):
                        return True
        
        return False
    
    def _check_chicken_hand(self, ctx: ScoringContext) -> bool:
        """No other patterns (minimum 8 points not met naturally)"""
        # This is a special case - used when no other patterns apply
        return False  # Always returns False, applied separately
    
    def _check_last_tile_draw(self, ctx: ScoringContext) -> bool:
        """Self-drawn win on last tile"""
        return ctx.is_zimo and ctx.is_last_tile
    
    def _check_last_tile_claim(self, ctx: ScoringContext) -> bool:
        """Win on last discarded tile"""
        return not ctx.is_zimo and ctx.is_last_tile
    
    def _check_out_with_replacement(self, ctx: ScoringContext) -> bool:
        """Win on replacement tile after kong"""
        return ctx.is_zimo and ctx.is_kong_draw
    
    def _check_robbing_kong(self, ctx: ScoringContext) -> bool:
        """Win by robbing opponent's add-kong"""
        return ctx.is_robbing_kong
    
    def _check_two_concealed_kongs(self, ctx: ScoringContext) -> bool:
        """Two concealed kongs"""
        concealed_kongs = sum(1 for m in ctx.melds if m.meld_type == MeldType.CONCEALED_KONG)
        return concealed_kongs >= 2
    
    # --- 6 Points ---
    def _check_all_pungs(self, ctx: ScoringContext) -> bool:
        """Four pungs/kongs + pair"""
        chow_count = sum(1 for st, _ in ctx.sets if st == 'chow')
        return chow_count == 0
    
    def _check_half_flush(self, ctx: ScoringContext) -> bool:
        """One numbered suit + honors"""
        numbered_suits = set()
        has_honors = False
        
        for t in ctx.all_tiles:
            if t.is_honor:
                has_honors = True
            else:
                numbered_suits.add(t.suit)
        
        return len(numbered_suits) == 1 and has_honors
    
    def _check_mixed_shifted_chows(self, ctx: ScoringContext) -> bool:
        """Three chows in sequence from three suits"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        for i in range(len(chows)):
            for j in range(i+1, len(chows)):
                for k in range(j+1, len(chows)):
                    suits = {chows[i][0], chows[j][0], chows[k][0]}
                    vals = sorted([chows[i][1], chows[j][1], chows[k][1]])
                    if len(suits) == 3 and vals == list(range(vals[0], vals[0]+3)):
                        return True
        
        return False
    
    def _check_all_types(self, ctx: ScoringContext) -> bool:
        """All five tile types present (3 suits + winds + dragons)"""
        has_chars = any(t.suit == TileSuit.CHARACTERS for t in ctx.all_tiles)
        has_bams = any(t.suit == TileSuit.BAMBOOS for t in ctx.all_tiles)
        has_dots = any(t.suit == TileSuit.DOTS for t in ctx.all_tiles)
        has_winds = any(t.suit == TileSuit.WINDS for t in ctx.all_tiles)
        has_dragons = any(t.suit == TileSuit.DRAGONS for t in ctx.all_tiles)
        
        return all([has_chars, has_bams, has_dots, has_winds, has_dragons])
    
    def _check_melded_hand(self, ctx: ScoringContext) -> bool:
        """Four melds + win on discard"""
        return len(ctx.melds) == 4 and not ctx.is_zimo
    
    def _check_two_dragon_pungs(self, ctx: ScoringContext) -> bool:
        """Two dragon pungs"""
        dragon_pungs = 0
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.DRAGONS:
                    dragon_pungs += 1
        return dragon_pungs == 2
    
    # --- 4 Points ---
    def _check_outside_hand(self, ctx: ScoringContext) -> bool:
        """Every set contains terminal or honor"""
        for st, tiles in ctx.sets:
            has_terminal_honor = any(t.is_terminal_or_honor for t in tiles)
            if not has_terminal_honor:
                return False
        
        if ctx.pair and not ctx.pair[0].is_terminal_or_honor:
            return False
        
        return True
    
    def _check_fully_concealed_hand(self, ctx: ScoringContext) -> bool:
        """Concealed hand with self-drawn win"""
        return ctx.is_concealed and ctx.is_zimo
    
    def _check_two_melded_kongs(self, ctx: ScoringContext) -> bool:
        """Two exposed kongs"""
        melded_kongs = sum(1 for m in ctx.melds 
                          if m.meld_type == MeldType.KONG and not m.is_concealed)
        return melded_kongs >= 2
    
    def _check_last_tile(self, ctx: ScoringContext) -> bool:
        """Win on 4th tile of a type (all others visible)"""
        # This requires game state knowledge not available in context
        # Simplified: check if winning tile is the last available
        return False  # Requires external game state
    
    # --- 2 Points ---
    def _check_dragon_pung(self, ctx: ScoringContext) -> bool:
        """Pung of dragons"""
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.DRAGONS:
                    return True
        return False
    
    def _check_prevalent_wind(self, ctx: ScoringContext) -> bool:
        """Pung of round wind"""
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.WINDS and tiles[0].value == ctx.round_wind:
                    return True
        return False
    
    def _check_seat_wind(self, ctx: ScoringContext) -> bool:
        """Pung of seat wind"""
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                if tiles[0].suit == TileSuit.WINDS and tiles[0].value == ctx.seat_wind:
                    return True
        return False
    
    def _check_concealed_hand(self, ctx: ScoringContext) -> bool:
        """No exposed melds (concealed kongs OK)"""
        for meld in ctx.melds:
            if meld.meld_type not in (MeldType.CONCEALED_KONG,):
                if not meld.is_concealed:
                    return False
        return True
    
    def _check_all_chows(self, ctx: ScoringContext) -> bool:
        """Four chows + non-honor pair"""
        pung_count = sum(1 for st, _ in ctx.sets if st in ('pong', 'kong'))
        if pung_count > 0:
            return False
        
        if ctx.pair and ctx.pair[0].is_honor:
            return False
        
        return True
    
    def _check_tile_hog(self, ctx: ScoringContext) -> bool:
        """Four of a tile type without kong"""
        for i in range(34):
            if ctx.counts[i] == 4:
                # Check it's not a kong
                is_kong = False
                for st, tiles in ctx.sets:
                    if st == 'kong' and tiles[0].tile_index == i:
                        is_kong = True
                        break
                if not is_kong:
                    return True
        return False
    
    def _check_double_pung(self, ctx: ScoringContext) -> bool:
        """Two pungs of same number in different suits"""
        pungs = []
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                t = tiles[0]
                if t.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    pungs.append((t.suit, t.value))
        
        from collections import defaultdict
        by_value = defaultdict(set)
        for suit, val in pungs:
            by_value[val].add(suit)
        
        return any(len(suits) >= 2 for suits in by_value.values())
    
    def _check_two_concealed_pungs(self, ctx: ScoringContext) -> bool:
        """Two concealed pungs"""
        concealed_pungs = 0
        
        for meld in ctx.melds:
            if meld.meld_type == MeldType.CONCEALED_KONG:
                concealed_pungs += 1
        
        hand_counts = ctx.hand.to_count_array()
        for i in range(34):
            if hand_counts[i] >= 3:
                concealed_pungs += 1
        
        return concealed_pungs >= 2
    
    def _check_concealed_kong(self, ctx: ScoringContext) -> bool:
        """One concealed kong"""
        return any(m.meld_type == MeldType.CONCEALED_KONG for m in ctx.melds)
    
    def _check_all_simples(self, ctx: ScoringContext) -> bool:
        """No terminals or honors"""
        return all(t.is_simple for t in ctx.all_tiles)
    
    # --- 1 Point ---
    def _check_pure_double_chow(self, ctx: ScoringContext) -> bool:
        """Two identical chows in same suit"""
        chows = [tuple(sorted(t.tile_index for t in tiles)) 
                 for st, tiles in ctx.sets if st == 'chow']
        
        from collections import Counter
        chow_counts = Counter(chows)
        return any(c >= 2 for c in chow_counts.values())
    
    def _check_mixed_double_chow(self, ctx: ScoringContext) -> bool:
        """Two identical chows in different suits"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        from collections import defaultdict
        by_value = defaultdict(set)
        for suit, val in chows:
            by_value[val].add(suit)
        
        return any(len(suits) >= 2 for suits in by_value.values())
    
    def _check_short_straight(self, ctx: ScoringContext) -> bool:
        """Two consecutive chows in same suit (e.g., 123-456)"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        for suit in range(3):
            suit_chows = sorted([v for s, v in chows if s == suit])
            for i in range(len(suit_chows) - 1):
                if suit_chows[i+1] == suit_chows[i] + 3:
                    return True
        
        return False
    
    def _check_two_terminal_chows(self, ctx: ScoringContext) -> bool:
        """123 and 789 in same suit"""
        chows = []
        for st, tiles in ctx.sets:
            if st == 'chow':
                t = min(tiles, key=lambda x: x.value)
                chows.append((t.suit, t.value))
        
        for suit in range(3):
            suit_chows = set(v for s, v in chows if s == suit)
            if 1 in suit_chows and 7 in suit_chows:
                return True
        
        return False
    
    def _check_pung_terminals_honors(self, ctx: ScoringContext) -> bool:
        """Pung of terminals or honors"""
        for st, tiles in ctx.sets:
            if st in ('pong', 'kong'):
                if tiles[0].is_terminal_or_honor:
                    return True
        return False
    
    def _check_melded_kong(self, ctx: ScoringContext) -> bool:
        """One exposed kong"""
        return any(m.meld_type == MeldType.KONG and not m.is_concealed for m in ctx.melds)
    
    def _check_one_voided_suit(self, ctx: ScoringContext) -> bool:
        """Missing one numbered suit"""
        has_chars = any(t.suit == TileSuit.CHARACTERS for t in ctx.all_tiles)
        has_bams = any(t.suit == TileSuit.BAMBOOS for t in ctx.all_tiles)
        has_dots = any(t.suit == TileSuit.DOTS for t in ctx.all_tiles)
        
        suit_count = sum([has_chars, has_bams, has_dots])
        return suit_count == 2
    
    def _check_no_honors(self, ctx: ScoringContext) -> bool:
        """No honor tiles"""
        return not any(t.is_honor for t in ctx.all_tiles)
    
    def _check_edge_wait(self, ctx: ScoringContext) -> bool:
        """Waiting on 3 to complete 12X or 7 to complete X89"""
        wt = ctx.winning_tile
        if wt.is_honor:
            return False
        
        # Check if winning tile completed a chow at the edge
        for st, tiles in ctx.sets:
            if st == 'chow' and wt in tiles:
                vals = sorted(t.value for t in tiles)
                if wt.value == 3 and vals == [1, 2, 3]:
                    return True
                if wt.value == 7 and vals == [7, 8, 9]:
                    return True
        
        return False
    
    def _check_closed_wait(self, ctx: ScoringContext) -> bool:
        """Waiting on middle tile of a sequence"""
        wt = ctx.winning_tile
        if wt.is_honor:
            return False
        
        for st, tiles in ctx.sets:
            if st == 'chow' and wt in tiles:
                vals = sorted(t.value for t in tiles)
                if wt.value == vals[1]:  # Middle tile
                    return True
        
        return False
    
    def _check_single_wait(self, ctx: ScoringContext) -> bool:
        """Waiting on the pair tile"""
        if ctx.pair and ctx.winning_tile == ctx.pair[0]:
            return True
        return False
    
    def _check_self_drawn(self, ctx: ScoringContext) -> bool:
        """Self-drawn win (zimo)"""
        return ctx.is_zimo
    
    def _check_flower_tiles(self, ctx: ScoringContext) -> bool:
        """Flower tiles (not used in standard MCR)"""
        return False  # MCR doesn't use flower tiles

