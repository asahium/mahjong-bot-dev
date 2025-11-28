"""
Riichi Mahjong Game Engine

Main game logic for Japanese Mahjong (Riichi).
Supports both EMA and Tenhou rule variations.
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Any
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcr_mahjong.tiles import Tile, TileSet, TileSuit
from mcr_mahjong.player import Meld, MeldType
from mcr_mahjong.wall import Wall

from .player import RiichiPlayer, PlayerState, FuritenState
from .rules import RuleSet, TENHOU_RULES


class RiichiPhase(IntEnum):
    """Phases of a Riichi Mahjong game"""
    NOT_STARTED = 0
    DRAWING = 1          # Current player draws
    DISCARDING = 2       # Current player must discard
    CLAIMING = 3         # Other players can claim
    AFTER_KAN = 4        # After kan, check for chankan/rinshan
    GAME_OVER = 5


class RiichiActionType(IntEnum):
    """Types of actions in Riichi Mahjong"""
    DRAW = 0            # Draw a tile
    DISCARD = 1         # Discard a tile
    RIICHI = 2          # Declare riichi (with discard)
    CHII = 3            # Call chii (sequence)
    PON = 4             # Call pon (triplet)
    KAN = 5             # Call kan (from discard)
    ANKAN = 6           # Declare concealed kan
    SHOUMINKAN = 7      # Add to pon (kakan)
    TSUMO = 8           # Declare tsumo (self-draw win)
    RON = 9             # Declare ron (win from discard)
    PASS = 10           # Pass on claiming
    KYUUSHU = 11        # Nine terminals draw


@dataclass
class RiichiAction:
    """
    Represents a player action in Riichi Mahjong.
    """
    action_type: RiichiActionType
    player_idx: int
    tile: Optional[Tile] = None
    meld_tiles: Optional[List[Tile]] = None
    
    def __repr__(self) -> str:
        return f"RiichiAction({self.action_type.name}, P{self.player_idx}, {self.tile})"


@dataclass 
class RoundResult:
    """Result of a round"""
    winner: Optional[int] = None
    loser: Optional[int] = None  # For ron
    win_type: str = ""  # "tsumo", "ron", "draw"
    han: int = 0
    fu: int = 0
    score: int = 0
    yaku_list: List[str] = field(default_factory=list)
    dora_count: int = 0
    uradora_count: int = 0


class RiichiGame:
    """
    Riichi Mahjong Game Engine.
    
    Manages the full game state for 4-player Riichi Mahjong.
    """
    
    NUM_PLAYERS = 4
    INITIAL_SCORE = 25000
    
    def __init__(
        self, 
        rules: Optional[RuleSet] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize a new game.
        
        Args:
            rules: Rule set to use (default: Tenhou rules)
            seed: Random seed for reproducibility
        """
        self.rules = rules or TENHOU_RULES
        self.seed = seed
        
        # Wall (includes dead wall handling)
        self.wall = Wall()
        self.dead_wall: List[Tile] = []
        self.dora_indicators: List[Tile] = []
        self.uradora_indicators: List[Tile] = []
        self.kan_count = 0
        
        # Players
        self.players: List[RiichiPlayer] = [
            RiichiPlayer(i, score=self.rules.starting_points)
            for i in range(self.NUM_PLAYERS)
        ]
        
        # Game state
        self.phase = RiichiPhase.NOT_STARTED
        self.current_player = 0
        self.dealer = 0  # Oya (East)
        self.round_wind = 0  # 0=East, 1=South, 2=West, 3=North
        self.honba = 0  # Bonus counters
        self.riichi_sticks = 0  # Accumulated riichi bets
        self.turn_count = 0
        
        # Turn tracking
        self.last_discard: Optional[Tile] = None
        self.last_discard_player: Optional[int] = None
        self.last_draw: Optional[Tile] = None
        self.is_first_uninterrupted_go_around = True
        self.is_rinshan = False
        
        # Claiming state
        self.pending_claims: Dict[int, RiichiAction] = {}
        
        # History
        self.action_history: List[RiichiAction] = []
        
        # Scorer (set externally)
        self.scorer = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset for a new round"""
        if seed is not None:
            self.seed = seed
        
        self.wall.reset(self.seed)
        self._setup_dead_wall()
        
        for player in self.players:
            player.reset_round()
        
        self.phase = RiichiPhase.NOT_STARTED
        self.current_player = self.dealer
        self.turn_count = 0
        self.last_discard = None
        self.last_discard_player = None
        self.last_draw = None
        self.is_first_uninterrupted_go_around = True
        self.is_rinshan = False
        self.pending_claims = {}
        self.action_history = []
        self.kan_count = 0
    
    def _setup_dead_wall(self) -> None:
        """
        Set up the dead wall (14 tiles including dora indicators).
        The dead wall contains:
        - 4 dora indicator tiles (top row)
        - 4 uradora indicator tiles (bottom row)
        - 4 replacement tiles for kans
        - 2 extra tiles
        """
        # Draw 14 tiles for dead wall
        self.dead_wall = self.wall.draw_many(14)
        
        # First dora indicator is revealed
        self.dora_indicators = [self.dead_wall[0]]
        
        # Uradora indicators (revealed at end if riichi wins)
        self.uradora_indicators = [self.dead_wall[1]]
        
        # Track available kan replacement tiles
        self.kan_replacement_tiles = self.dead_wall[4:8]
    
    def get_dora_tiles(self) -> List[Tile]:
        """
        Get actual dora tiles (next tile after indicators).
        """
        dora_tiles = []
        for indicator in self.dora_indicators:
            dora = self._get_dora_from_indicator(indicator)
            dora_tiles.append(dora)
        return dora_tiles
    
    def _get_dora_from_indicator(self, indicator: Tile) -> Tile:
        """
        Get the dora tile from an indicator.
        Dora is the next tile in sequence.
        """
        if indicator.suit in (TileSuit.CHARACTERS, TileSuit.BAMBOOS, TileSuit.DOTS):
            # Numbered suits: wrap 9->1
            next_val = indicator.value % 9 + 1
            return Tile(indicator.suit, next_val)
        elif indicator.suit == TileSuit.WINDS:
            # Winds: E->S->W->N->E
            next_val = (indicator.value + 1) % 4
            return Tile(TileSuit.WINDS, next_val)
        else:  # Dragons
            # Dragons: Red->Green->White->Red
            next_val = (indicator.value + 1) % 3
            return Tile(TileSuit.DRAGONS, next_val)
    
    def start_round(self) -> None:
        """Start a new round by dealing hands"""
        if self.phase != RiichiPhase.NOT_STARTED:
            raise RuntimeError("Round already started")
        
        # Deal hands (13 tiles each)
        for i in range(self.NUM_PLAYERS):
            player_idx = (self.dealer + i) % self.NUM_PLAYERS
            tiles = self.wall.draw_many(13)
            for tile in tiles:
                self.players[player_idx].add_tile(tile)
        
        # Dealer goes first
        self.current_player = self.dealer
        self.phase = RiichiPhase.DRAWING
    
    def step(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """
        Execute an action and advance game state.
        
        Returns:
            Tuple of (round_over, result)
        """
        if self.phase == RiichiPhase.GAME_OVER:
            raise RuntimeError("Round is already over")
        
        self.action_history.append(action)
        
        handlers = {
            RiichiActionType.DRAW: self._handle_draw,
            RiichiActionType.DISCARD: self._handle_discard,
            RiichiActionType.RIICHI: self._handle_riichi,
            RiichiActionType.CHII: self._handle_chii,
            RiichiActionType.PON: self._handle_pon,
            RiichiActionType.KAN: self._handle_kan,
            RiichiActionType.ANKAN: self._handle_ankan,
            RiichiActionType.SHOUMINKAN: self._handle_shouminkan,
            RiichiActionType.TSUMO: self._handle_tsumo,
            RiichiActionType.RON: self._handle_ron,
            RiichiActionType.PASS: self._handle_pass,
            RiichiActionType.KYUUSHU: self._handle_kyuushu,
        }
        
        handler = handlers.get(action.action_type)
        if handler:
            return handler(action)
        
        return False, None
    
    def _handle_draw(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle drawing a tile"""
        # Allow draw in DRAWING phase or AFTER_KAN phase (for rinshan)
        is_rinshan = (self.phase == RiichiPhase.AFTER_KAN)
        
        if self.phase not in (RiichiPhase.DRAWING, RiichiPhase.AFTER_KAN):
            raise ValueError(f"Cannot draw in phase {self.phase}")
        
        player = self.players[action.player_idx]
        
        if is_rinshan:
            # Draw from dead wall after kan
            if self.kan_replacement_tiles:
                tile = self.kan_replacement_tiles.pop(0)
            else:
                # Fallback to regular wall if no replacement tiles
                if self.wall.remaining <= 0:
                    return self._handle_exhaustive_draw()
                tile = self.wall.draw()
            self.is_rinshan = True
        else:
            # Check for exhaustive draw
            if self.wall.remaining <= 0:
                return self._handle_exhaustive_draw()
            tile = self.wall.draw()
            self.is_rinshan = False
        
        player.add_tile(tile)
        self.last_draw = tile
        self.turn_count += 1
        
        # Reset temporary furiten at start of turn
        player.furiten.reset_temporary()
        
        # Mark first draw completed (for double riichi check)
        if not player.first_draw_completed:
            player.first_draw_completed = True
        
        self.phase = RiichiPhase.DISCARDING
        return False, None
    
    def _handle_discard(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle discarding a tile"""
        if self.phase != RiichiPhase.DISCARDING:
            raise ValueError(f"Cannot discard in phase {self.phase}")
        
        player = self.players[action.player_idx]
        tile = action.tile
        
        is_tsumogiri = (tile == self.last_draw)
        
        if not player.discard_tile(tile, is_riichi=False, is_tsumogiri=is_tsumogiri):
            raise ValueError(f"Player {action.player_idx} doesn't have tile {tile}")
        
        self.last_discard = tile
        self.last_discard_player = action.player_idx
        self.last_draw = None
        
        # Cancel ippatsu for all players on any discard
        for p in self.players:
            if p.index != action.player_idx:
                p.cancel_ippatsu()
        
        # First uninterrupted go-around ends after all players discard once
        # (simplified: ends after any call)
        
        self.phase = RiichiPhase.CLAIMING
        self.pending_claims = {}
        
        return False, None
    
    def _handle_riichi(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle riichi declaration with discard"""
        if self.phase != RiichiPhase.DISCARDING:
            raise ValueError(f"Cannot riichi in phase {self.phase}")
        
        player = self.players[action.player_idx]
        tile = action.tile
        
        # Check if double riichi is valid (first turn, no calls)
        is_double = (not player.first_draw_completed or 
                     (self.turn_count <= self.NUM_PLAYERS and 
                      self.is_first_uninterrupted_go_around))
        
        if not player.declare_riichi(is_double=is_double):
            raise ValueError(f"Player {action.player_idx} cannot declare riichi")
        
        is_tsumogiri = (tile == self.last_draw)
        if not player.discard_tile(tile, is_riichi=True, is_tsumogiri=is_tsumogiri):
            raise ValueError(f"Player {action.player_idx} doesn't have tile {tile}")
        
        player.riichi_discard_index = len(player.discards) - 1
        self.riichi_sticks += 1
        
        self.last_discard = tile
        self.last_discard_player = action.player_idx
        self.last_draw = None
        
        self.phase = RiichiPhase.CLAIMING
        self.pending_claims = {}
        
        return False, None
    
    def _handle_chii(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle chii (sequence) call"""
        player = self.players[action.player_idx]
        
        # Chii can only be called from player to the left
        expected_source = (action.player_idx - 1) % self.NUM_PLAYERS
        if self.last_discard_player != expected_source:
            raise ValueError("Chii can only be called from player to your left")
        
        # Remove tiles from hand
        for tile in action.meld_tiles:
            if not player.remove_tile(tile):
                raise ValueError(f"Player doesn't have tile {tile} for chii")
        
        # Create meld
        meld_tiles = list(action.meld_tiles) + [self.last_discard]
        meld = Meld(
            meld_type=MeldType.CHOW,
            tiles=meld_tiles,
            is_concealed=False,
            source_player=self.last_discard_player,
            source_tile=self.last_discard
        )
        player.declare_meld(meld)
        
        # Calls break first go-around and cancel ippatsu
        self.is_first_uninterrupted_go_around = False
        for p in self.players:
            p.cancel_ippatsu()
        
        self.last_discard = None
        self.last_discard_player = None
        
        self.current_player = action.player_idx
        self.phase = RiichiPhase.DISCARDING
        
        return False, None
    
    def _handle_pon(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle pon (triplet) call"""
        player = self.players[action.player_idx]
        
        # Remove 2 tiles from hand
        for _ in range(2):
            if not player.remove_tile(self.last_discard):
                raise ValueError("Player doesn't have enough tiles for pon")
        
        meld_tiles = [self.last_discard] * 3
        meld = Meld(
            meld_type=MeldType.PONG,
            tiles=meld_tiles,
            is_concealed=False,
            source_player=self.last_discard_player,
            source_tile=self.last_discard
        )
        player.declare_meld(meld)
        
        self.is_first_uninterrupted_go_around = False
        for p in self.players:
            p.cancel_ippatsu()
        
        self.last_discard = None
        self.last_discard_player = None
        
        self.current_player = action.player_idx
        self.phase = RiichiPhase.DISCARDING
        
        return False, None
    
    def _handle_kan(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle open kan (from discard)"""
        player = self.players[action.player_idx]
        
        # Remove 3 tiles from hand
        for _ in range(3):
            if not player.remove_tile(self.last_discard):
                raise ValueError("Player doesn't have enough tiles for kan")
        
        meld_tiles = [self.last_discard] * 4
        meld = Meld(
            meld_type=MeldType.KONG,
            tiles=meld_tiles,
            is_concealed=False,
            source_player=self.last_discard_player,
            source_tile=self.last_discard
        )
        player.declare_meld(meld)
        
        self.kan_count += 1
        self._reveal_kandora()
        
        self.is_first_uninterrupted_go_around = False
        for p in self.players:
            p.cancel_ippatsu()
        
        self.last_discard = None
        self.last_discard_player = None
        
        # Draw replacement tile
        self.current_player = action.player_idx
        self.phase = RiichiPhase.AFTER_KAN
        
        return False, None
    
    def _handle_ankan(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle concealed kan (ankan)"""
        player = self.players[action.player_idx]
        tile = action.tile
        
        # Remove 4 tiles from hand
        for _ in range(4):
            if not player.remove_tile(tile):
                raise ValueError("Player doesn't have 4 tiles for ankan")
        
        meld_tiles = [tile] * 4
        meld = Meld(
            meld_type=MeldType.CONCEALED_KONG,
            tiles=meld_tiles,
            is_concealed=True
        )
        player.declare_meld(meld)
        
        self.kan_count += 1
        self._reveal_kandora()
        
        # Ankan in riichi doesn't break ippatsu for self
        # but breaks for others
        for p in self.players:
            if p.index != action.player_idx:
                p.cancel_ippatsu()
        
        # Draw replacement tile
        self.phase = RiichiPhase.AFTER_KAN
        
        return False, None
    
    def _handle_shouminkan(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle added kan (shouminkan/kakan)"""
        player = self.players[action.player_idx]
        tile = action.tile
        
        # Find and upgrade the pon
        for i, meld in enumerate(player.melds):
            if meld.meld_type == MeldType.PONG and meld.tiles[0] == tile:
                if not player.remove_tile(tile):
                    raise ValueError(f"Player doesn't have tile {tile}")
                
                new_meld = Meld(
                    meld_type=MeldType.KONG,
                    tiles=[tile] * 4,
                    is_concealed=False,
                    source_player=meld.source_player,
                    source_tile=meld.source_tile
                )
                player.melds[i] = new_meld
                break
        
        self.kan_count += 1
        self._reveal_kandora()
        
        # Note: Other players can ron on shouminkan (chankan)
        # This is handled in the claiming phase
        
        for p in self.players:
            p.cancel_ippatsu()
        
        self.phase = RiichiPhase.AFTER_KAN
        
        return False, None
    
    def _reveal_kandora(self) -> None:
        """Reveal a new kan dora indicator"""
        if self.kan_count <= 4 and len(self.dead_wall) > self.kan_count * 2:
            self.dora_indicators.append(self.dead_wall[self.kan_count * 2])
            self.uradora_indicators.append(self.dead_wall[self.kan_count * 2 + 1])
    
    def _handle_tsumo(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle tsumo (self-draw win)"""
        player = self.players[action.player_idx]
        
        result = self._calculate_win(
            winner_idx=action.player_idx,
            loser_idx=None,
            winning_tile=self.last_draw,
            is_tsumo=True
        )
        
        # No yaku = not a valid tsumo, continue playing
        if result.score == 0:
            return False, None
        
        self._apply_tsumo_scores(action.player_idx, result)
        
        self.phase = RiichiPhase.GAME_OVER
        return True, result
    
    def _handle_ron(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle ron (win from discard)"""
        player = self.players[action.player_idx]
        
        # Check furiten - cannot ron if in furiten
        if player.furiten.is_furiten:
            # Furiten - treat as pass instead of error
            return self._handle_pass(RiichiAction(RiichiActionType.PASS, action.player_idx))
        
        result = self._calculate_win(
            winner_idx=action.player_idx,
            loser_idx=self.last_discard_player,
            winning_tile=self.last_discard,
            is_tsumo=False
        )
        
        # No yaku = not a valid ron, treat as pass
        if result.score == 0:
            return self._handle_pass(RiichiAction(RiichiActionType.PASS, action.player_idx))
        
        self._apply_ron_scores(action.player_idx, self.last_discard_player, result)
        
        self.phase = RiichiPhase.GAME_OVER
        return True, result
    
    def _handle_pass(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle passing on a claim"""
        player = self.players[action.player_idx]
        
        # If passing on a ron opportunity, set furiten
        if self._can_ron(action.player_idx, self.last_discard):
            player.furiten.set_passed_ron(player.is_riichi)
        
        self.pending_claims[action.player_idx] = action
        
        # Check if all players have responded
        claiming_players = set(range(self.NUM_PLAYERS)) - {self.last_discard_player}
        if set(self.pending_claims.keys()) >= claiming_players:
            # All passed, next player draws
            self.current_player = (self.last_discard_player + 1) % self.NUM_PLAYERS
            self.phase = RiichiPhase.DRAWING
            self.last_discard = None
            self.last_discard_player = None
        
        return False, None
    
    def _handle_kyuushu(self, action: RiichiAction) -> Tuple[bool, Optional[RoundResult]]:
        """Handle 9 terminals/honors draw (kyuushu kyuuhai)"""
        if not self.rules.allow_kyuushu:
            raise ValueError("Kyuushu kyuuhai not allowed in this ruleset")
        
        result = RoundResult(
            win_type="draw",
            yaku_list=["Kyuushu Kyuuhai"]
        )
        
        self.phase = RiichiPhase.GAME_OVER
        return True, result
    
    def _handle_exhaustive_draw(self) -> Tuple[bool, Optional[RoundResult]]:
        """Handle exhaustive draw (wall empty)"""
        # Check tenpai for each player
        tenpai_players = []
        for player in self.players:
            if self._is_tenpai(player):
                tenpai_players.append(player.index)
        
        # Distribute noten penalty (3000 points total)
        if len(tenpai_players) > 0 and len(tenpai_players) < 4:
            penalty_total = 3000
            noten_players = [i for i in range(4) if i not in tenpai_players]
            
            penalty_per_noten = penalty_total // len(noten_players)
            bonus_per_tenpai = penalty_total // len(tenpai_players)
            
            for i in noten_players:
                self.players[i].score -= penalty_per_noten
            for i in tenpai_players:
                self.players[i].score += bonus_per_tenpai
        
        result = RoundResult(
            win_type="draw",
            yaku_list=["Ryuukyoku (Exhaustive Draw)"]
        )
        
        self.phase = RiichiPhase.GAME_OVER
        return True, result
    
    def _calculate_win(
        self,
        winner_idx: int,
        loser_idx: Optional[int],
        winning_tile: Tile,
        is_tsumo: bool
    ) -> RoundResult:
        """Calculate win result using scorer"""
        if self.scorer is None:
            # Return dummy result if no scorer
            return RoundResult(
                winner=winner_idx,
                loser=loser_idx,
                win_type="tsumo" if is_tsumo else "ron",
                score=8000  # Default mangan
            )
        
        player = self.players[winner_idx]
        
        return self.scorer.calculate_score(
            hand=player.hand,
            melds=player.melds,
            winning_tile=winning_tile,
            is_tsumo=is_tsumo,
            is_riichi=player.is_riichi,
            is_double_riichi=player.is_double_riichi,
            is_ippatsu=player.ippatsu_valid,
            is_rinshan=self.phase == RiichiPhase.AFTER_KAN,
            is_chankan=False,  # TODO: implement
            is_haitei=self.wall.remaining == 0,
            is_houtei=self.wall.remaining == 0 and not is_tsumo,
            is_tenhou=is_tsumo and self.turn_count == 1 and winner_idx == self.dealer,
            is_chihou=is_tsumo and self.turn_count <= 4 and winner_idx != self.dealer,
            round_wind=self.round_wind,
            seat_wind=winner_idx,
            dora_tiles=self.get_dora_tiles(),
            uradora_tiles=[self._get_dora_from_indicator(i) for i in self.uradora_indicators] if player.is_riichi else [],
            rules=self.rules,
        )
    
    def _apply_tsumo_scores(self, winner_idx: int, result: RoundResult) -> None:
        """Apply score changes for tsumo"""
        winner = self.players[winner_idx]
        is_dealer = winner_idx == self.dealer
        
        if is_dealer:
            # Dealer tsumo: all pay equal
            payment = (result.score + self.honba * 300) // 3
            for i, player in enumerate(self.players):
                if i != winner_idx:
                    player.score -= payment
                    winner.score += payment
        else:
            # Non-dealer tsumo: dealer pays more
            dealer_payment = (result.score // 2) + self.honba * 100
            other_payment = (result.score // 4) + self.honba * 100
            
            for i, player in enumerate(self.players):
                if i == winner_idx:
                    continue
                if i == self.dealer:
                    player.score -= dealer_payment
                    winner.score += dealer_payment
                else:
                    player.score -= other_payment
                    winner.score += other_payment
        
        # Winner gets riichi sticks
        winner.score += self.riichi_sticks * 1000
        self.riichi_sticks = 0
    
    def _apply_ron_scores(self, winner_idx: int, loser_idx: int, result: RoundResult) -> None:
        """Apply score changes for ron"""
        winner = self.players[winner_idx]
        loser = self.players[loser_idx]
        
        payment = result.score + self.honba * 300
        loser.score -= payment
        winner.score += payment
        
        # Winner gets riichi sticks
        winner.score += self.riichi_sticks * 1000
        self.riichi_sticks = 0
    
    def _is_tenpai(self, player: RiichiPlayer) -> bool:
        """Check if player is in tenpai (one tile away from winning)"""
        hand_counts = player.get_hand_count_array()
        
        # Try adding each possible tile and check for win
        for tile_idx in range(34):
            test_counts = hand_counts.copy()
            test_counts[tile_idx] += 1
            
            if self._is_winning_hand_counts(test_counts, len(player.melds)):
                return True
        
        return False
    
    def _is_winning_hand_counts(self, counts: np.ndarray, num_melds: int) -> bool:
        """Check if tile counts form a winning hand"""
        sets_needed = 4 - num_melds
        return self._can_decompose(counts.copy(), sets_needed, need_pair=True)
    
    def _can_decompose(self, counts: np.ndarray, sets_needed: int, need_pair: bool) -> bool:
        """Recursively check if tiles can form winning structure"""
        if sets_needed == 0:
            if not need_pair:
                return np.sum(counts) == 0
            for i in range(34):
                if counts[i] >= 2:
                    counts[i] -= 2
                    if np.sum(counts) == 0:
                        return True
                    counts[i] += 2
            return False
        
        first_idx = -1
        for i in range(34):
            if counts[i] > 0:
                first_idx = i
                break
        
        if first_idx == -1:
            return sets_needed == 0 and not need_pair
        
        # Try triplet
        if counts[first_idx] >= 3:
            counts[first_idx] -= 3
            if self._can_decompose(counts, sets_needed - 1, need_pair):
                return True
            counts[first_idx] += 3
        
        # Try sequence (numbered suits only)
        if first_idx < 27:
            suit_start = (first_idx // 9) * 9
            val = first_idx - suit_start
            if val <= 6:
                idx1, idx2, idx3 = first_idx, first_idx + 1, first_idx + 2
                if counts[idx1] >= 1 and counts[idx2] >= 1 and counts[idx3] >= 1:
                    counts[idx1] -= 1
                    counts[idx2] -= 1
                    counts[idx3] -= 1
                    if self._can_decompose(counts, sets_needed - 1, need_pair):
                        return True
                    counts[idx1] += 1
                    counts[idx2] += 1
                    counts[idx3] += 1
        
        # Try pair
        if need_pair and counts[first_idx] >= 2:
            counts[first_idx] -= 2
            if self._can_decompose(counts, sets_needed, need_pair=False):
                return True
            counts[first_idx] += 2
        
        return False
    
    def _can_ron(self, player_idx: int, tile: Tile) -> bool:
        """Check if player can declare ron on the given tile"""
        player = self.players[player_idx]
        
        if player.furiten.is_furiten:
            return False
        
        # Add tile to hand temporarily
        test_hand = player.hand.copy()
        test_hand.add(tile)
        
        counts = test_hand.to_count_array()
        return self._is_winning_hand_counts(counts, len(player.melds))
    
    def get_valid_actions(self, player_idx: int) -> List[RiichiAction]:
        """Get all valid actions for a player"""
        valid = []
        player = self.players[player_idx]
        
        if self.phase == RiichiPhase.DRAWING:
            if player_idx == self.current_player:
                valid.append(RiichiAction(RiichiActionType.DRAW, player_idx))
        
        elif self.phase == RiichiPhase.DISCARDING:
            if player_idx == self.current_player:
                # In riichi, can only discard drawn tile (tsumogiri)
                if player.is_riichi:
                    if self.last_draw:
                        valid.append(RiichiAction(
                            RiichiActionType.DISCARD, player_idx, self.last_draw
                        ))
                else:
                    # Can discard any tile
                    for tile in player.hand.get_unique_tiles():
                        valid.append(RiichiAction(
                            RiichiActionType.DISCARD, player_idx, tile
                        ))
                    
                    # Can declare riichi if tenpai and menzen
                    if player.is_menzen and self._is_tenpai(player):
                        if player.score >= 1000:
                            for tile in player.hand.get_unique_tiles():
                                # Check if discarding this tile keeps tenpai
                                valid.append(RiichiAction(
                                    RiichiActionType.RIICHI, player_idx, tile
                                ))
                
                # Check for ankan
                for tile in player.can_ankan():
                    valid.append(RiichiAction(RiichiActionType.ANKAN, player_idx, tile))
                
                # Check for shouminkan (not in riichi)
                if not player.is_riichi:
                    for tile in player.can_shouminkan():
                        valid.append(RiichiAction(
                            RiichiActionType.SHOUMINKAN, player_idx, tile
                        ))
                
                # Check for tsumo
                if self.last_draw and self._can_tsumo(player_idx):
                    valid.append(RiichiAction(
                        RiichiActionType.TSUMO, player_idx, self.last_draw
                    ))
        
        elif self.phase == RiichiPhase.CLAIMING:
            if player_idx != self.last_discard_player:
                # Check for ron
                if self._can_ron(player_idx, self.last_discard):
                    valid.append(RiichiAction(
                        RiichiActionType.RON, player_idx, self.last_discard
                    ))
                
                # Check for kan (not in riichi)
                if not player.is_riichi and player.can_kan(self.last_discard):
                    valid.append(RiichiAction(
                        RiichiActionType.KAN, player_idx, self.last_discard
                    ))
                
                # Check for pon (not in riichi)
                if not player.is_riichi and player.can_pon(self.last_discard):
                    valid.append(RiichiAction(
                        RiichiActionType.PON, player_idx, self.last_discard
                    ))
                
                # Check for chii (only from left player, not in riichi)
                if not player.is_riichi:
                    if (self.last_discard_player + 1) % self.NUM_PLAYERS == player_idx:
                        for pair in player.can_chii(self.last_discard):
                            valid.append(RiichiAction(
                                RiichiActionType.CHII, player_idx,
                                self.last_discard, meld_tiles=list(pair)
                            ))
                
                # Can always pass
                valid.append(RiichiAction(RiichiActionType.PASS, player_idx))
        
        elif self.phase == RiichiPhase.AFTER_KAN:
            if player_idx == self.current_player:
                # Draw replacement tile from dead wall
                valid.append(RiichiAction(RiichiActionType.DRAW, player_idx))
        
        return valid
    
    def _can_tsumo(self, player_idx: int) -> bool:
        """Check if player can declare tsumo"""
        player = self.players[player_idx]
        
        if self.last_draw is None:
            return False
        
        counts = player.hand.to_count_array()
        if not self._is_winning_hand_counts(counts, len(player.melds)):
            return False
        
        # Must have at least one yaku (simplified check)
        # In full implementation, scorer would verify
        return True
    
    def get_observation(self, player_idx: int) -> Dict[str, Any]:
        """Get observation for a player"""
        player = self.players[player_idx]
        
        discards = []
        melds = []
        riichi_status = []
        scores = []
        
        for p in self.players:
            discards.append(p.discards)
            melds.append(p.melds)
            riichi_status.append(p.is_riichi)
            scores.append(p.score)
        
        return {
            "hand": player.get_hand_count_array(),
            "hand_tiles": player.get_hand_tiles(),
            "melds": melds,
            "own_melds": player.melds,
            "discards": discards,
            "dora_indicators": self.dora_indicators,
            "riichi_status": riichi_status,
            "furiten": player.furiten.is_furiten,
            "scores": scores,
            "last_discard": self.last_discard,
            "last_discard_player": self.last_discard_player,
            "last_draw": self.last_draw if player_idx == self.current_player else None,
            "current_player": self.current_player,
            "phase": self.phase,
            "round_wind": self.round_wind,
            "seat_wind": player_idx,
            "dealer": self.dealer,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "wall_remaining": self.wall.remaining,
            "turn_count": self.turn_count,
            "valid_actions": self.get_valid_actions(player_idx),
        }
    
    def copy(self) -> 'RiichiGame':
        """Create a deep copy of the game"""
        new_game = RiichiGame.__new__(RiichiGame)
        new_game.rules = self.rules
        new_game.seed = self.seed
        new_game.wall = self.wall.copy()
        new_game.dead_wall = list(self.dead_wall)
        new_game.dora_indicators = list(self.dora_indicators)
        new_game.uradora_indicators = list(self.uradora_indicators)
        new_game.kan_count = self.kan_count
        new_game.players = [p.copy() for p in self.players]
        new_game.phase = self.phase
        new_game.current_player = self.current_player
        new_game.dealer = self.dealer
        new_game.round_wind = self.round_wind
        new_game.honba = self.honba
        new_game.riichi_sticks = self.riichi_sticks
        new_game.turn_count = self.turn_count
        new_game.last_discard = self.last_discard
        new_game.last_discard_player = self.last_discard_player
        new_game.last_draw = self.last_draw
        new_game.is_first_uninterrupted_go_around = self.is_first_uninterrupted_go_around
        new_game.is_rinshan = self.is_rinshan
        new_game.pending_claims = dict(self.pending_claims)
        new_game.action_history = list(self.action_history)
        new_game.scorer = self.scorer
        return new_game
    
    def __repr__(self) -> str:
        return f"RiichiGame(phase={self.phase.name}, player={self.current_player}, wall={self.wall.remaining})"

