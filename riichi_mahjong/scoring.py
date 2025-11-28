"""
Riichi Mahjong Scoring System

Implements all Yaku (winning patterns) and Han/Fu calculation.
Supports both EMA and Tenhou scoring variations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict, Callable
from enum import IntEnum, auto
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcr_mahjong.tiles import Tile, TileSet, TileSuit, WindType, DragonType
from mcr_mahjong.player import Meld, MeldType

from .rules import RuleSet, TENHOU_RULES
from .dora import DoraSystem


class YakuType(IntEnum):
    """Categories of Yaku"""
    NORMAL = 0      # Regular yaku
    YAKUMAN = 1     # Limit hand (yakuman)
    DOUBLE_YAKUMAN = 2  # Double yakuman (optional rule)


@dataclass
class Yaku:
    """Represents a Yaku (winning pattern)"""
    name: str
    japanese_name: str
    han_closed: int        # Han value when closed
    han_open: int          # Han value when open (0 = not allowed open)
    yaku_type: YakuType = YakuType.NORMAL
    
    @property
    def is_yakuman(self) -> bool:
        return self.yaku_type in (YakuType.YAKUMAN, YakuType.DOUBLE_YAKUMAN)


@dataclass
class ScoringResult:
    """Result of scoring calculation"""
    han: int = 0
    fu: int = 0
    score: int = 0
    yaku_list: List[Yaku] = field(default_factory=list)
    dora_count: int = 0
    uradora_count: int = 0
    akadora_count: int = 0
    is_yakuman: bool = False
    yakuman_count: int = 0
    
    def add_yaku(self, yaku: Yaku, is_open: bool = False):
        """Add a yaku to the result"""
        self.yaku_list.append(yaku)
        if yaku.is_yakuman:
            self.is_yakuman = True
            self.yakuman_count += 1
        else:
            han = yaku.han_open if is_open else yaku.han_closed
            self.han += han


@dataclass
class HandAnalysis:
    """Analyzed hand structure for scoring"""
    tiles: List[Tile]
    melds: List[Meld]
    winning_tile: Tile
    is_tsumo: bool
    
    # Decomposition
    sets: List[Tuple[str, List[Tile]]] = field(default_factory=list)
    pair: Optional[List[Tile]] = None
    
    # State flags
    is_closed: bool = True
    is_pinfu: bool = False
    is_chiitoitsu: bool = False
    is_kokushi: bool = False
    
    # Win conditions
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False
    is_rinshan: bool = False
    is_chankan: bool = False
    is_haitei: bool = False
    is_houtei: bool = False
    is_tenhou: bool = False
    is_chihou: bool = False
    
    # Context
    round_wind: int = 0
    seat_wind: int = 0
    
    # Counts
    counts: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=np.int8))


class RiichiScorer:
    """
    Riichi Mahjong Scorer
    
    Calculates han, fu, and final score for winning hands.
    """
    
    def __init__(self, rules: Optional[RuleSet] = None):
        self.rules = rules or TENHOU_RULES
        self.yaku_checks = self._create_yaku_checks()
    
    def calculate_score(
        self,
        hand: TileSet,
        melds: List[Meld],
        winning_tile: Tile,
        is_tsumo: bool,
        is_riichi: bool = False,
        is_double_riichi: bool = False,
        is_ippatsu: bool = False,
        is_rinshan: bool = False,
        is_chankan: bool = False,
        is_haitei: bool = False,
        is_houtei: bool = False,
        is_tenhou: bool = False,
        is_chihou: bool = False,
        round_wind: int = 0,
        seat_wind: int = 0,
        dora_tiles: List[Tile] = None,
        uradora_tiles: List[Tile] = None,
        rules: Optional[RuleSet] = None,
    ) -> ScoringResult:
        """
        Calculate the score for a winning hand.
        
        Args:
            hand: Player's hand tiles (including winning tile)
            melds: Declared melds
            winning_tile: The tile that completed the hand
            is_tsumo: Self-drawn win
            is_riichi: Player declared riichi
            is_double_riichi: Player declared double riichi
            is_ippatsu: Win within one turn of riichi
            is_rinshan: Win on kan replacement tile
            is_chankan: Win by robbing a kan
            is_haitei: Win on last tile (tsumo)
            is_houtei: Win on last discard
            is_tenhou: Dealer wins on initial hand
            is_chihou: Non-dealer wins on first draw
            round_wind: Current round wind (0=E, 1=S, 2=W, 3=N)
            seat_wind: Player's seat wind
            dora_tiles: List of dora tiles
            uradora_tiles: List of uradora tiles (for riichi wins)
            rules: Rule set to use
            
        Returns:
            ScoringResult with han, fu, score, and yaku list
        """
        rules = rules or self.rules
        dora_tiles = dora_tiles or []
        uradora_tiles = uradora_tiles or []
        
        # Create hand analysis
        analysis = self._analyze_hand(
            hand, melds, winning_tile, is_tsumo,
            is_riichi, is_double_riichi, is_ippatsu,
            is_rinshan, is_chankan, is_haitei, is_houtei,
            is_tenhou, is_chihou, round_wind, seat_wind
        )
        
        result = ScoringResult()
        
        # Check for yakuman first
        yakuman_list = self._check_yakuman(analysis)
        if yakuman_list:
            for yaku in yakuman_list:
                result.add_yaku(yaku, not analysis.is_closed)
            result.score = self._calculate_yakuman_score(
                result.yakuman_count, is_tsumo, seat_wind == 0
            )
            return result
        
        # Check regular yaku
        is_open = not analysis.is_closed
        for yaku, check_func in self.yaku_checks:
            if check_func(analysis):
                if is_open and yaku.han_open == 0:
                    continue  # This yaku not allowed when open
                result.add_yaku(yaku, is_open)
        
        # Must have at least one yaku
        if not result.yaku_list:
            return ScoringResult()  # No yaku, invalid win
        
        # Count dora
        all_tiles = list(hand.tiles)
        for meld in melds:
            all_tiles.extend(meld.tiles)
        
        for tile in all_tiles:
            for dora in dora_tiles:
                if tile == dora:
                    result.dora_count += 1
        
        # Uradora (only for riichi wins)
        if is_riichi:
            for tile in all_tiles:
                for uradora in uradora_tiles:
                    if tile == uradora:
                        result.uradora_count += 1
        
        result.han += result.dora_count + result.uradora_count + result.akadora_count
        
        # Calculate fu
        result.fu = self._calculate_fu(analysis, is_tsumo, round_wind, seat_wind)
        
        # Calculate final score
        result.score = self._calculate_score(
            result.han, result.fu, is_tsumo, seat_wind == 0
        )
        
        return result
    
    def _analyze_hand(
        self,
        hand: TileSet,
        melds: List[Meld],
        winning_tile: Tile,
        is_tsumo: bool,
        is_riichi: bool,
        is_double_riichi: bool,
        is_ippatsu: bool,
        is_rinshan: bool,
        is_chankan: bool,
        is_haitei: bool,
        is_houtei: bool,
        is_tenhou: bool,
        is_chihou: bool,
        round_wind: int,
        seat_wind: int,
    ) -> HandAnalysis:
        """Analyze hand structure"""
        all_tiles = list(hand.tiles)
        
        analysis = HandAnalysis(
            tiles=all_tiles,
            melds=melds,
            winning_tile=winning_tile,
            is_tsumo=is_tsumo,
            is_riichi=is_riichi,
            is_double_riichi=is_double_riichi,
            is_ippatsu=is_ippatsu,
            is_rinshan=is_rinshan,
            is_chankan=is_chankan,
            is_haitei=is_haitei,
            is_houtei=is_houtei,
            is_tenhou=is_tenhou,
            is_chihou=is_chihou,
            round_wind=round_wind,
            seat_wind=seat_wind,
        )
        
        # Check if closed
        analysis.is_closed = all(
            m.is_concealed or m.meld_type == MeldType.CONCEALED_KONG
            for m in melds
        )
        
        # Build count array
        for tile in all_tiles:
            analysis.counts[tile.tile_index] += 1
        for meld in melds:
            for tile in meld.tiles:
                analysis.counts[tile.tile_index] += 1
        
        # Check for special hands
        if self._is_chiitoitsu(analysis.counts):
            analysis.is_chiitoitsu = True
        elif self._is_kokushi(analysis.counts):
            analysis.is_kokushi = True
        else:
            # Regular decomposition
            self._decompose_hand(analysis)
        
        return analysis
    
    def _decompose_hand(self, analysis: HandAnalysis) -> bool:
        """Decompose hand into sets and pair"""
        # Add melds to sets
        for meld in analysis.melds:
            if meld.meld_type == MeldType.CHOW:
                analysis.sets.append(('chow', sorted(meld.tiles, key=lambda t: t.value)))
            elif meld.meld_type == MeldType.PONG:
                analysis.sets.append(('pong', meld.tiles))
            elif meld.meld_type in (MeldType.KONG, MeldType.CONCEALED_KONG):
                analysis.sets.append(('kan', meld.tiles))
        
        # Decompose remaining hand
        hand_counts = np.zeros(34, dtype=np.int8)
        for tile in analysis.tiles:
            hand_counts[tile.tile_index] += 1
        
        return self._find_decomposition(analysis, hand_counts, 4 - len(analysis.melds))
    
    def _find_decomposition(
        self, 
        analysis: HandAnalysis, 
        counts: np.ndarray, 
        sets_needed: int
    ) -> bool:
        """Find valid decomposition"""
        if sets_needed == 0:
            # Find pair
            for i in range(34):
                if counts[i] >= 2:
                    tile = Tile.from_index(i)
                    analysis.pair = [tile, tile]
                    return True
            return False
        
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
            analysis.sets.append(('pong', [tile, tile, tile]))
            if self._find_decomposition(analysis, counts, sets_needed - 1):
                return True
            analysis.sets.pop()
            counts[first_idx] += 3
        
        # Try chow
        if first_idx < 27:
            suit_start = (first_idx // 9) * 9
            val = first_idx - suit_start
            if val <= 6:
                idx1, idx2, idx3 = first_idx, first_idx + 1, first_idx + 2
                if counts[idx1] >= 1 and counts[idx2] >= 1 and counts[idx3] >= 1:
                    counts[idx1] -= 1
                    counts[idx2] -= 1
                    counts[idx3] -= 1
                    t1, t2, t3 = Tile.from_index(idx1), Tile.from_index(idx2), Tile.from_index(idx3)
                    analysis.sets.append(('chow', [t1, t2, t3]))
                    if self._find_decomposition(analysis, counts, sets_needed - 1):
                        return True
                    analysis.sets.pop()
                    counts[idx1] += 1
                    counts[idx2] += 1
                    counts[idx3] += 1
        
        # Try pair
        if analysis.pair is None and counts[first_idx] >= 2:
            counts[first_idx] -= 2
            tile = Tile.from_index(first_idx)
            analysis.pair = [tile, tile]
            if self._find_decomposition(analysis, counts, sets_needed):
                return True
            analysis.pair = None
            counts[first_idx] += 2
        
        return False
    
    def _is_chiitoitsu(self, counts: np.ndarray) -> bool:
        """Check for 7 pairs hand"""
        pairs = sum(1 for c in counts if c == 2)
        singles = sum(1 for c in counts if c == 1)
        return pairs == 7 and singles == 0
    
    def _is_kokushi(self, counts: np.ndarray) -> bool:
        """Check for 13 orphans"""
        terminals_honors = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        has_pair = False
        
        for idx in terminals_honors:
            if counts[idx] == 0:
                return False
            if counts[idx] == 2:
                has_pair = True
        
        # Check only these tiles
        for i in range(34):
            if i not in terminals_honors and counts[i] > 0:
                return False
        
        return has_pair
    
    def _create_yaku_checks(self) -> List[Tuple[Yaku, Callable]]:
        """Create list of yaku with their check functions"""
        return [
            # 1 Han
            (Yaku("Riichi", "立直", 1, 0), self._check_riichi),
            (Yaku("Ippatsu", "一発", 1, 0), self._check_ippatsu),
            (Yaku("Menzen Tsumo", "門前清自摸和", 1, 0), self._check_menzen_tsumo),
            (Yaku("Tanyao", "断幺九", 1, 1), self._check_tanyao),
            (Yaku("Pinfu", "平和", 1, 0), self._check_pinfu),
            (Yaku("Iipeikou", "一盃口", 1, 0), self._check_iipeikou),
            (Yaku("Yakuhai (East)", "役牌 東", 1, 1), lambda a: self._check_yakuhai_wind(a, 0)),
            (Yaku("Yakuhai (South)", "役牌 南", 1, 1), lambda a: self._check_yakuhai_wind(a, 1)),
            (Yaku("Yakuhai (West)", "役牌 西", 1, 1), lambda a: self._check_yakuhai_wind(a, 2)),
            (Yaku("Yakuhai (North)", "役牌 北", 1, 1), lambda a: self._check_yakuhai_wind(a, 3)),
            (Yaku("Yakuhai (Haku)", "役牌 白", 1, 1), lambda a: self._check_yakuhai_dragon(a, 2)),
            (Yaku("Yakuhai (Hatsu)", "役牌 發", 1, 1), lambda a: self._check_yakuhai_dragon(a, 1)),
            (Yaku("Yakuhai (Chun)", "役牌 中", 1, 1), lambda a: self._check_yakuhai_dragon(a, 0)),
            (Yaku("Rinshan Kaihou", "嶺上開花", 1, 1), self._check_rinshan),
            (Yaku("Chankan", "槍槓", 1, 1), self._check_chankan),
            (Yaku("Haitei", "海底摸月", 1, 1), self._check_haitei),
            (Yaku("Houtei", "河底撈魚", 1, 1), self._check_houtei),
            
            # 2 Han
            (Yaku("Double Riichi", "両立直", 2, 0), self._check_double_riichi),
            (Yaku("Chiitoitsu", "七対子", 2, 0), self._check_chiitoitsu),
            (Yaku("Sanshoku Doujun", "三色同順", 2, 1), self._check_sanshoku_doujun),
            (Yaku("Ittsu", "一気通貫", 2, 1), self._check_ittsu),
            (Yaku("Toitoi", "対々和", 2, 2), self._check_toitoi),
            (Yaku("Sanankou", "三暗刻", 2, 2), self._check_sanankou),
            (Yaku("Sanshoku Doukou", "三色同刻", 2, 2), self._check_sanshoku_doukou),
            (Yaku("Sankantsu", "三槓子", 2, 2), self._check_sankantsu),
            (Yaku("Chanta", "混全帯幺九", 2, 1), self._check_chanta),
            (Yaku("Honroutou", "混老頭", 2, 2), self._check_honroutou),
            (Yaku("Shousangen", "小三元", 2, 2), self._check_shousangen),
            
            # 3 Han
            (Yaku("Honitsu", "混一色", 3, 2), self._check_honitsu),
            (Yaku("Junchan", "純全帯幺九", 3, 2), self._check_junchan),
            (Yaku("Ryanpeikou", "二盃口", 3, 0), self._check_ryanpeikou),
            
            # 6 Han
            (Yaku("Chinitsu", "清一色", 6, 5), self._check_chinitsu),
        ]
    
    def _check_yakuman(self, analysis: HandAnalysis) -> List[Yaku]:
        """Check for yakuman hands"""
        yakuman = []
        
        # Tenhou (dealer first draw win)
        if analysis.is_tenhou:
            yakuman.append(Yaku("Tenhou", "天和", 0, 0, YakuType.YAKUMAN))
        
        # Chihou (non-dealer first draw win)
        if analysis.is_chihou:
            yakuman.append(Yaku("Chihou", "地和", 0, 0, YakuType.YAKUMAN))
        
        # Kokushi Musou (13 orphans)
        if analysis.is_kokushi:
            yakuman.append(Yaku("Kokushi Musou", "国士無双", 0, 0, YakuType.YAKUMAN))
        
        # Suuankou (4 concealed triplets)
        if self._check_suuankou(analysis):
            yakuman.append(Yaku("Suuankou", "四暗刻", 0, 0, YakuType.YAKUMAN))
        
        # Daisangen (big 3 dragons)
        if self._check_daisangen(analysis):
            yakuman.append(Yaku("Daisangen", "大三元", 0, 0, YakuType.YAKUMAN))
        
        # Shousuushii (small 4 winds)
        if self._check_shousuushii(analysis):
            yakuman.append(Yaku("Shousuushii", "小四喜", 0, 0, YakuType.YAKUMAN))
        
        # Daisuushii (big 4 winds)
        if self._check_daisuushii(analysis):
            yakuman.append(Yaku("Daisuushii", "大四喜", 0, 0, YakuType.YAKUMAN))
        
        # Tsuuiisou (all honors)
        if self._check_tsuuiisou(analysis):
            yakuman.append(Yaku("Tsuuiisou", "字一色", 0, 0, YakuType.YAKUMAN))
        
        # Chinroutou (all terminals)
        if self._check_chinroutou(analysis):
            yakuman.append(Yaku("Chinroutou", "清老頭", 0, 0, YakuType.YAKUMAN))
        
        # Ryuuiisou (all green)
        if self._check_ryuuiisou(analysis):
            yakuman.append(Yaku("Ryuuiisou", "緑一色", 0, 0, YakuType.YAKUMAN))
        
        # Chuuren Poutou (9 gates)
        if self._check_chuuren(analysis):
            yakuman.append(Yaku("Chuuren Poutou", "九蓮宝燈", 0, 0, YakuType.YAKUMAN))
        
        # Suukantsu (4 kans)
        if self._check_suukantsu(analysis):
            yakuman.append(Yaku("Suukantsu", "四槓子", 0, 0, YakuType.YAKUMAN))
        
        return yakuman
    
    # === Yaku Check Functions ===
    
    def _check_riichi(self, a: HandAnalysis) -> bool:
        return a.is_riichi and not a.is_double_riichi
    
    def _check_ippatsu(self, a: HandAnalysis) -> bool:
        return a.is_ippatsu
    
    def _check_menzen_tsumo(self, a: HandAnalysis) -> bool:
        return a.is_closed and a.is_tsumo
    
    def _check_tanyao(self, a: HandAnalysis) -> bool:
        """All simples (no terminals/honors)"""
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if tile.is_terminal_or_honor:
                    return False
        return True
    
    def _check_pinfu(self, a: HandAnalysis) -> bool:
        """All sequences, valueless pair, two-sided wait"""
        if not a.is_closed:
            return False
        if a.is_chiitoitsu or a.is_kokushi:
            return False
        
        # All sets must be chows
        for set_type, _ in a.sets:
            if set_type != 'chow':
                return False
        
        # Pair must not be yakuhai
        if a.pair:
            pair_tile = a.pair[0]
            if pair_tile.suit == TileSuit.DRAGONS:
                return False
            if pair_tile.suit == TileSuit.WINDS:
                if pair_tile.value == a.round_wind or pair_tile.value == a.seat_wind:
                    return False
        
        return True
    
    def _check_iipeikou(self, a: HandAnalysis) -> bool:
        """Two identical sequences"""
        if not a.is_closed:
            return False
        
        chows = [tuple(t.tile_index for t in tiles) 
                 for st, tiles in a.sets if st == 'chow']
        
        from collections import Counter
        counts = Counter(chows)
        return any(c == 2 for c in counts.values())
    
    def _check_yakuhai_wind(self, a: HandAnalysis, wind: int) -> bool:
        """Check for wind pong/kan"""
        # Wind is yakuhai if it's round wind or seat wind
        is_round_wind = (wind == a.round_wind)
        is_seat_wind = (wind == a.seat_wind)
        
        if not (is_round_wind or is_seat_wind):
            return False
        
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                if tiles[0].suit == TileSuit.WINDS and tiles[0].value == wind:
                    return True
        return False
    
    def _check_yakuhai_dragon(self, a: HandAnalysis, dragon: int) -> bool:
        """Check for dragon pong/kan"""
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                if tiles[0].suit == TileSuit.DRAGONS and tiles[0].value == dragon:
                    return True
        return False
    
    def _check_rinshan(self, a: HandAnalysis) -> bool:
        return a.is_rinshan
    
    def _check_chankan(self, a: HandAnalysis) -> bool:
        return a.is_chankan
    
    def _check_haitei(self, a: HandAnalysis) -> bool:
        return a.is_haitei and a.is_tsumo
    
    def _check_houtei(self, a: HandAnalysis) -> bool:
        return a.is_houtei and not a.is_tsumo
    
    def _check_double_riichi(self, a: HandAnalysis) -> bool:
        return a.is_double_riichi
    
    def _check_chiitoitsu(self, a: HandAnalysis) -> bool:
        return a.is_chiitoitsu
    
    def _check_sanshoku_doujun(self, a: HandAnalysis) -> bool:
        """Three suits, same sequence"""
        chows_by_value = {}
        for st, tiles in a.sets:
            if st == 'chow':
                min_tile = min(tiles, key=lambda t: t.value)
                key = min_tile.value
                if key not in chows_by_value:
                    chows_by_value[key] = set()
                chows_by_value[key].add(min_tile.suit)
        
        return any(len(suits) >= 3 for suits in chows_by_value.values())
    
    def _check_ittsu(self, a: HandAnalysis) -> bool:
        """1-2-3, 4-5-6, 7-8-9 in same suit"""
        for suit in [TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS]:
            chow_starts = set()
            for st, tiles in a.sets:
                if st == 'chow':
                    min_tile = min(tiles, key=lambda t: t.value)
                    if min_tile.suit == suit:
                        chow_starts.add(min_tile.value)
            if {1, 4, 7}.issubset(chow_starts):
                return True
        return False
    
    def _check_toitoi(self, a: HandAnalysis) -> bool:
        """All triplets/quads"""
        for st, _ in a.sets:
            if st == 'chow':
                return False
        return True
    
    def _check_sanankou(self, a: HandAnalysis) -> bool:
        """Three concealed triplets"""
        concealed_pongs = 0
        
        for meld in a.melds:
            if meld.meld_type == MeldType.CONCEALED_KONG:
                concealed_pongs += 1
        
        # Count pongs in hand (not from melds)
        for st, tiles in a.sets:
            if st == 'pong':
                # Check if this pong was from hand (not a meld)
                is_meld = any(
                    m.meld_type == MeldType.PONG and m.tiles[0] == tiles[0]
                    for m in a.melds
                )
                if not is_meld:
                    concealed_pongs += 1
        
        return concealed_pongs >= 3
    
    def _check_sanshoku_doukou(self, a: HandAnalysis) -> bool:
        """Same triplet in three suits"""
        pongs_by_value = {}
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                t = tiles[0]
                if t.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
                    key = t.value
                    if key not in pongs_by_value:
                        pongs_by_value[key] = set()
                    pongs_by_value[key].add(t.suit)
        
        return any(len(suits) >= 3 for suits in pongs_by_value.values())
    
    def _check_sankantsu(self, a: HandAnalysis) -> bool:
        """Three kans"""
        kans = sum(1 for st, _ in a.sets if st == 'kan')
        return kans == 3
    
    def _check_chanta(self, a: HandAnalysis) -> bool:
        """All sets contain terminal or honor"""
        if a.is_chiitoitsu or a.is_kokushi:
            return False
        
        for st, tiles in a.sets:
            has_terminal_honor = any(t.is_terminal_or_honor for t in tiles)
            if not has_terminal_honor:
                return False
        
        if a.pair and not a.pair[0].is_terminal_or_honor:
            return False
        
        # Must have at least one sequence (otherwise it's honroutou)
        has_chow = any(st == 'chow' for st, _ in a.sets)
        return has_chow
    
    def _check_honroutou(self, a: HandAnalysis) -> bool:
        """All terminals and honors"""
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if not tile.is_terminal_or_honor:
                    return False
        return True
    
    def _check_shousangen(self, a: HandAnalysis) -> bool:
        """Small 3 dragons (2 pongs + pair)"""
        dragon_pongs = 0
        dragon_pair = False
        
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                if tiles[0].suit == TileSuit.DRAGONS:
                    dragon_pongs += 1
        
        if a.pair and a.pair[0].suit == TileSuit.DRAGONS:
            dragon_pair = True
        
        return dragon_pongs == 2 and dragon_pair
    
    def _check_honitsu(self, a: HandAnalysis) -> bool:
        """One suit + honors"""
        suits = set()
        has_honors = False
        
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if tile.is_honor:
                    has_honors = True
                else:
                    suits.add(tile.suit)
        
        return len(suits) == 1 and has_honors
    
    def _check_junchan(self, a: HandAnalysis) -> bool:
        """All sets contain terminal (no honors)"""
        if a.is_chiitoitsu or a.is_kokushi:
            return False
        
        has_honors = any(Tile.from_index(i).is_honor for i in range(34) if a.counts[i] > 0)
        if has_honors:
            return False
        
        for st, tiles in a.sets:
            has_terminal = any(t.is_terminal for t in tiles)
            if not has_terminal:
                return False
        
        if a.pair and not a.pair[0].is_terminal:
            return False
        
        return True
    
    def _check_ryanpeikou(self, a: HandAnalysis) -> bool:
        """Two sets of identical sequences"""
        if not a.is_closed:
            return False
        
        chows = [tuple(t.tile_index for t in tiles) 
                 for st, tiles in a.sets if st == 'chow']
        
        from collections import Counter
        counts = Counter(chows)
        pairs_of_chows = sum(1 for c in counts.values() if c >= 2)
        return pairs_of_chows >= 2
    
    def _check_chinitsu(self, a: HandAnalysis) -> bool:
        """Pure one suit (no honors)"""
        suits = set()
        
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if tile.is_honor:
                    return False
                suits.add(tile.suit)
        
        return len(suits) == 1
    
    # === Yakuman Checks ===
    
    def _check_suuankou(self, a: HandAnalysis) -> bool:
        """Four concealed triplets"""
        concealed_pongs = 0
        
        for meld in a.melds:
            if meld.meld_type == MeldType.CONCEALED_KONG:
                concealed_pongs += 1
            elif not meld.is_concealed:
                return False  # Has open meld
        
        for st, tiles in a.sets:
            if st == 'pong':
                is_meld = any(
                    m.meld_type == MeldType.PONG and m.tiles[0] == tiles[0]
                    for m in a.melds
                )
                if not is_meld:
                    concealed_pongs += 1
        
        return concealed_pongs >= 4
    
    def _check_daisangen(self, a: HandAnalysis) -> bool:
        """Big 3 dragons (3 dragon pongs)"""
        dragon_pongs = 0
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                if tiles[0].suit == TileSuit.DRAGONS:
                    dragon_pongs += 1
        return dragon_pongs == 3
    
    def _check_shousuushii(self, a: HandAnalysis) -> bool:
        """Small 4 winds (3 wind pongs + wind pair)"""
        wind_pongs = 0
        wind_pair = False
        
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                if tiles[0].suit == TileSuit.WINDS:
                    wind_pongs += 1
        
        if a.pair and a.pair[0].suit == TileSuit.WINDS:
            wind_pair = True
        
        return wind_pongs == 3 and wind_pair
    
    def _check_daisuushii(self, a: HandAnalysis) -> bool:
        """Big 4 winds (4 wind pongs)"""
        wind_pongs = 0
        for st, tiles in a.sets:
            if st in ('pong', 'kan'):
                if tiles[0].suit == TileSuit.WINDS:
                    wind_pongs += 1
        return wind_pongs == 4
    
    def _check_tsuuiisou(self, a: HandAnalysis) -> bool:
        """All honors"""
        for i in range(34):
            if a.counts[i] > 0:
                if not Tile.from_index(i).is_honor:
                    return False
        return True
    
    def _check_chinroutou(self, a: HandAnalysis) -> bool:
        """All terminals"""
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if not tile.is_terminal:
                    return False
        return True
    
    def _check_ryuuiisou(self, a: HandAnalysis) -> bool:
        """All green (2,3,4,6,8 bamboo + green dragon)"""
        for i in range(34):
            if a.counts[i] > 0:
                if not Tile.from_index(i).is_green:
                    return False
        return True
    
    def _check_chuuren(self, a: HandAnalysis) -> bool:
        """Nine gates (1112345678999 + any in same suit)"""
        if not a.is_closed:
            return False
        
        suits = set()
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if tile.is_honor:
                    return False
                suits.add(tile.suit)
        
        if len(suits) != 1:
            return False
        
        suit = list(suits)[0]
        suit_counts = [0] * 9
        for i in range(34):
            if a.counts[i] > 0:
                tile = Tile.from_index(i)
                if tile.suit == suit:
                    suit_counts[tile.value - 1] = a.counts[i]
        
        # Check 1112345678999 pattern
        required = [3, 1, 1, 1, 1, 1, 1, 1, 3]
        for i in range(9):
            if suit_counts[i] < required[i]:
                return False
        
        return True
    
    def _check_suukantsu(self, a: HandAnalysis) -> bool:
        """Four kans"""
        kans = sum(1 for st, _ in a.sets if st == 'kan')
        return kans == 4
    
    def _calculate_fu(
        self, 
        a: HandAnalysis, 
        is_tsumo: bool,
        round_wind: int,
        seat_wind: int
    ) -> int:
        """Calculate fu (minipoints)"""
        if a.is_chiitoitsu:
            return 25
        
        fu = 20  # Base fu
        
        # Tsumo (but not pinfu)
        if is_tsumo and not a.is_pinfu:
            fu += 2
        
        # Menzen ron
        if a.is_closed and not is_tsumo:
            fu += 10
        
        # Sets fu
        for st, tiles in a.sets:
            if st == 'pong':
                base = 4 if tiles[0].is_terminal_or_honor else 2
                # Double if concealed
                is_concealed = not any(
                    m.meld_type == MeldType.PONG and m.tiles[0] == tiles[0] and not m.is_concealed
                    for m in a.melds
                )
                fu += base * 2 if is_concealed else base
            elif st == 'kan':
                base = 16 if tiles[0].is_terminal_or_honor else 8
                is_concealed = any(
                    m.meld_type == MeldType.CONCEALED_KONG and m.tiles[0] == tiles[0]
                    for m in a.melds
                )
                fu += base * 2 if is_concealed else base
        
        # Pair fu
        if a.pair:
            pair_tile = a.pair[0]
            if pair_tile.suit == TileSuit.DRAGONS:
                fu += 2
            elif pair_tile.suit == TileSuit.WINDS:
                if pair_tile.value == round_wind:
                    fu += 2
                if pair_tile.value == seat_wind:
                    fu += 2
        
        # Wait fu
        # Simplified: single wait, edge wait, closed wait = 2 fu
        # This would require more complex wait analysis
        
        # Round up to nearest 10
        fu = ((fu + 9) // 10) * 10
        
        # Minimum 30 fu (except chiitoitsu)
        if fu < 30:
            fu = 30
        
        return fu
    
    def _calculate_score(
        self, 
        han: int, 
        fu: int, 
        is_tsumo: bool,
        is_dealer: bool
    ) -> int:
        """Calculate final score from han and fu"""
        # Limit hands
        if han >= 13:
            base = 8000  # Kazoe yakuman
        elif han >= 11:
            base = 6000  # Sanbaiman
        elif han >= 8:
            base = 4000  # Baiman
        elif han >= 6:
            base = 3000  # Haneman
        elif han >= 5:
            base = 2000  # Mangan
        else:
            # Calculate basic points
            base = fu * (2 ** (han + 2))
            if base > 2000:
                base = 2000  # Mangan cap
        
        if is_dealer:
            if is_tsumo:
                # Each player pays this
                return base * 2 * 3
            else:
                return base * 6
        else:
            if is_tsumo:
                # Dealer pays double, others pay single
                return base * 2 + base * 2
            else:
                return base * 4
    
    def _calculate_yakuman_score(
        self, 
        yakuman_count: int, 
        is_tsumo: bool,
        is_dealer: bool
    ) -> int:
        """Calculate yakuman score"""
        base = 8000 * yakuman_count
        
        if is_dealer:
            if is_tsumo:
                return base * 2 * 3
            else:
                return base * 6
        else:
            if is_tsumo:
                return base * 2 + base * 2
            else:
                return base * 4

