"""
Riichi Mahjong Rule Sets

Defines rule configurations for different Riichi Mahjong variants:
- EMA (European Mahjong Association)
- Tenhou (Japanese online platform)
- WRC (World Riichi Championship)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RuleSet:
    """
    Rule configuration for Riichi Mahjong.
    
    Different organizations and platforms use slightly different rules.
    This class encapsulates those differences.
    """
    
    name: str = "Default"
    
    # Starting points
    starting_points: int = 25000
    goal_points: int = 30000  # Target to end game
    
    # Uma (placement bonus/penalty)
    # Format: (1st, 2nd, 3rd, 4th) adjustments in thousands
    uma: Tuple[int, int, int, int] = (15, 5, -5, -15)
    
    # Oka (ante bonus to winner)
    oka: int = 0  # Points given from each player to 1st place
    
    # Red dora (akadora)
    red_fives: int = 0  # Number of red 5s (0, 3, or 4)
    
    # Kuitan (open tanyao)
    allow_kuitan: bool = True
    
    # Atozuke (winning without guaranteed yaku until final tile)
    allow_atozuke: bool = True
    
    # Multiple ron (double/triple ron)
    allow_multiple_ron: bool = True
    
    # Kuikae (swap calling)
    allow_kuikae: bool = False
    
    # Kyuushu kyuuhai (9 terminals redraw)
    allow_kyuushu: bool = True
    
    # Nagashi mangan
    allow_nagashi_mangan: bool = True
    
    # Pao (responsibility) rules
    apply_pao: bool = True
    
    # Renchan (dealer repeat) conditions
    renchan_on_tenpai: bool = True  # Dealer repeats if tenpai on draw
    
    # Agari yame (winner can end game if they're ahead)
    allow_agari_yame: bool = False
    
    # Number of rounds
    # 1 = East only (tonpuusen)
    # 2 = East-South (hanchan)
    # 3 = East-South-West (full game)
    num_rounds: int = 2
    
    # Sudden death if all below starting points
    sudden_death_below_zero: bool = True
    
    # Minimum points to call riichi
    min_riichi_points: int = 1000
    
    # Chombo (penalty for invalid win declaration)
    chombo_payment: int = 8000  # Mangan equivalent
    
    # Yakitori (penalty for not winning any hand)
    apply_yakitori: bool = False
    yakitori_penalty: int = 10000
    
    # Kan rules
    max_kans_per_round: int = 4  # Abortive draw if 4+ kans by different players
    
    # Furiten rules
    temporary_furiten: bool = True
    
    # Dora rules  
    kandora_immediate: bool = True  # Kandora revealed immediately (Tenhou style)
    uradora_on_riichi_win: bool = True
    
    def __repr__(self) -> str:
        return f"RuleSet({self.name})"


# EMA (European Mahjong Association) Rules
EMA_RULES = RuleSet(
    name="EMA",
    starting_points=30000,
    goal_points=30000,
    uma=(15, 5, -5, -15),
    oka=0,
    red_fives=0,  # No red dora in EMA
    allow_kuitan=True,
    allow_atozuke=True,
    allow_multiple_ron=True,
    allow_kuikae=False,
    allow_kyuushu=True,
    allow_nagashi_mangan=True,
    apply_pao=True,
    renchan_on_tenpai=True,
    allow_agari_yame=False,
    num_rounds=2,  # Hanchan
    sudden_death_below_zero=True,
    min_riichi_points=1000,
    chombo_payment=8000,
    apply_yakitori=False,
    max_kans_per_round=4,
    temporary_furiten=True,
    kandora_immediate=True,
    uradora_on_riichi_win=True,
)


# Tenhou Rules (Japanese online platform)
TENHOU_RULES = RuleSet(
    name="Tenhou",
    starting_points=25000,
    goal_points=30000,
    uma=(20, 10, -10, -20),  # 10-30 uma style
    oka=20000,  # 5000 from each player
    red_fives=3,  # One red 5 in each suit
    allow_kuitan=True,
    allow_atozuke=True,
    allow_multiple_ron=False,  # Head bump (atamahane)
    allow_kuikae=False,
    allow_kyuushu=True,
    allow_nagashi_mangan=True,
    apply_pao=True,
    renchan_on_tenpai=True,
    allow_agari_yame=True,
    num_rounds=2,  # Hanchan
    sudden_death_below_zero=True,
    min_riichi_points=1000,
    chombo_payment=8000,  # Mangan to all
    apply_yakitori=False,
    max_kans_per_round=4,
    temporary_furiten=True,
    kandora_immediate=True,  # Tenhou reveals kandora immediately
    uradora_on_riichi_win=True,
)


# WRC (World Riichi Championship) Rules
WRC_RULES = RuleSet(
    name="WRC",
    starting_points=30000,
    goal_points=30000,
    uma=(15, 5, -5, -15),
    oka=0,
    red_fives=0,
    allow_kuitan=True,
    allow_atozuke=True,
    allow_multiple_ron=True,
    allow_kuikae=False,
    allow_kyuushu=True,
    allow_nagashi_mangan=True,
    apply_pao=True,
    renchan_on_tenpai=True,
    allow_agari_yame=False,
    num_rounds=2,
    sudden_death_below_zero=True,
    min_riichi_points=1000,
    chombo_payment=8000,
    apply_yakitori=False,
    max_kans_per_round=4,
    temporary_furiten=True,
    kandora_immediate=True,
    uradora_on_riichi_win=True,
)


# Tonpuusen (East only) variant
TONPUUSEN_RULES = RuleSet(
    name="Tonpuusen",
    starting_points=25000,
    goal_points=30000,
    uma=(10, 5, -5, -10),
    oka=0,
    red_fives=3,
    allow_kuitan=True,
    allow_atozuke=True,
    allow_multiple_ron=False,
    allow_kuikae=False,
    allow_kyuushu=True,
    allow_nagashi_mangan=True,
    apply_pao=True,
    renchan_on_tenpai=True,
    allow_agari_yame=True,
    num_rounds=1,  # East only
    sudden_death_below_zero=True,
    min_riichi_points=1000,
    chombo_payment=8000,
    apply_yakitori=False,
    max_kans_per_round=4,
    temporary_furiten=True,
    kandora_immediate=True,
    uradora_on_riichi_win=True,
)

