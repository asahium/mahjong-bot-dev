"""
MCR Mahjong Game Engine

Main game logic for Chinese Official Mahjong (MCR).
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Any
import numpy as np

from .tiles import Tile, TileSet, TileSuit
from .player import Player, Meld, MeldType
from .wall import Wall


class GamePhase(IntEnum):
    """Phases of the game"""
    NOT_STARTED = 0
    DRAWING = 1      # Current player draws a tile
    DISCARDING = 2   # Current player must discard
    CLAIMING = 3     # Other players can claim the discard
    GAME_OVER = 4


class ActionType(IntEnum):
    """Types of actions a player can take"""
    DRAW = 0         # Draw a tile (automatic)
    DISCARD = 1      # Discard a tile
    CHOW = 2         # Claim for a sequence (顺子)
    PONG = 3         # Claim for a triplet (刻子)
    KONG = 4         # Claim for a quad (杠) - exposed
    CONCEALED_KONG = 5  # Declare concealed kong (暗杠)
    ADD_KONG = 6     # Add to existing pong to make kong (加杠)
    HU = 7           # Declare win (胡)
    PASS = 8         # Pass on claiming


@dataclass
class Action:
    """
    Represents a player action.
    
    Attributes:
        action_type: Type of action
        player_idx: Index of player taking the action
        tile: Primary tile involved (discard tile, claimed tile, etc.)
        meld_tiles: Additional tiles for Chow (the two tiles from hand)
    """
    action_type: ActionType
    player_idx: int
    tile: Optional[Tile] = None
    meld_tiles: Optional[List[Tile]] = None
    
    def __repr__(self) -> str:
        return f"Action({self.action_type.name}, P{self.player_idx}, {self.tile})"


@dataclass
class GameState:
    """Serializable game state for observation"""
    current_player: int
    phase: GamePhase
    round_wind: int  # 0=East, 1=South, 2=West, 3=North
    dealer: int
    turn_count: int
    wall_remaining: int
    last_discard: Optional[Tile]
    last_discard_player: Optional[int]


class Game:
    """
    MCR Mahjong Game Engine.
    
    Manages the full game state and rules for 4-player MCR Mahjong.
    """
    
    MIN_WINNING_SCORE = 8  # MCR requires minimum 8 points to win
    NUM_PLAYERS = 4
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a new game.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.wall = Wall()
        self.players: List[Player] = [Player(i) for i in range(self.NUM_PLAYERS)]
        
        # Game state
        self.phase = GamePhase.NOT_STARTED
        self.current_player = 0
        self.dealer = 0  # East seat
        self.round_wind = 0  # East round
        self.turn_count = 0
        
        # Last action tracking
        self.last_discard: Optional[Tile] = None
        self.last_discard_player: Optional[int] = None
        self.last_draw: Optional[Tile] = None
        
        # Claiming state
        self.pending_claims: Dict[int, Action] = {}  # player_idx -> Action
        
        # History
        self.action_history: List[Action] = []
        
        # Scoring module reference (set externally)
        self.scorer = None
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the game to initial state"""
        if seed is not None:
            self.seed = seed
        
        self.wall.reset(self.seed)
        for player in self.players:
            player.reset()
        
        self.phase = GamePhase.NOT_STARTED
        self.current_player = self.dealer
        self.turn_count = 0
        self.last_discard = None
        self.last_discard_player = None
        self.last_draw = None
        self.pending_claims = {}
        self.action_history = []
    
    def start_game(self) -> None:
        """Start the game by dealing hands"""
        if self.phase != GamePhase.NOT_STARTED:
            raise RuntimeError("Game already started")
        
        # Deal hands
        hands = self.wall.deal_hands(self.NUM_PLAYERS)
        for i, hand in enumerate(hands):
            for tile in hand:
                self.players[i].add_tile(tile)
        
        # Dealer (East) draws first tile
        self.current_player = self.dealer
        self.phase = GamePhase.DRAWING
    
    def step(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Execute an action and advance game state.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (game_over, winner_idx, winning_score)
        """
        if self.phase == GamePhase.GAME_OVER:
            raise RuntimeError("Game is already over")
        
        self.action_history.append(action)
        
        if action.action_type == ActionType.DRAW:
            return self._handle_draw(action)
        elif action.action_type == ActionType.DISCARD:
            return self._handle_discard(action)
        elif action.action_type == ActionType.CHOW:
            return self._handle_chow(action)
        elif action.action_type == ActionType.PONG:
            return self._handle_pong(action)
        elif action.action_type == ActionType.KONG:
            return self._handle_kong(action)
        elif action.action_type == ActionType.CONCEALED_KONG:
            return self._handle_concealed_kong(action)
        elif action.action_type == ActionType.ADD_KONG:
            return self._handle_add_kong(action)
        elif action.action_type == ActionType.HU:
            return self._handle_hu(action)
        elif action.action_type == ActionType.PASS:
            return self._handle_pass(action)
        
        return False, None, None
    
    def _handle_draw(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle drawing a tile"""
        if self.phase != GamePhase.DRAWING:
            raise ValueError(f"Cannot draw in phase {self.phase}")
        
        player = self.players[action.player_idx]
        
        # Check for exhaustive draw
        if self.wall.is_empty:
            self.phase = GamePhase.GAME_OVER
            return True, None, None  # Draw game
        
        tile = self.wall.draw()
        player.add_tile(tile)
        self.last_draw = tile
        self.turn_count += 1
        
        self.phase = GamePhase.DISCARDING
        return False, None, None
    
    def _handle_discard(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle discarding a tile"""
        if self.phase != GamePhase.DISCARDING:
            raise ValueError(f"Cannot discard in phase {self.phase}")
        
        player = self.players[action.player_idx]
        tile = action.tile
        
        if not player.discard_tile(tile):
            raise ValueError(f"Player {action.player_idx} doesn't have tile {tile}")
        
        self.last_discard = tile
        self.last_discard_player = action.player_idx
        self.last_draw = None
        
        self.phase = GamePhase.CLAIMING
        self.pending_claims = {}
        
        return False, None, None
    
    def _handle_chow(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle claiming a Chow"""
        player = self.players[action.player_idx]
        
        # Chow can only be claimed from the player to your left
        expected_source = (action.player_idx - 1) % self.NUM_PLAYERS
        if self.last_discard_player != expected_source:
            raise ValueError("Chow can only be claimed from player to your left")
        
        # Remove tiles from hand
        for tile in action.meld_tiles:
            if not player.remove_tile(tile):
                raise ValueError(f"Player doesn't have tile {tile} for Chow")
        
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
        
        # Clear discard
        self.last_discard = None
        self.last_discard_player = None
        
        # Player must discard
        self.current_player = action.player_idx
        self.phase = GamePhase.DISCARDING
        
        return False, None, None
    
    def _handle_pong(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle claiming a Pong"""
        player = self.players[action.player_idx]
        
        # Remove 2 tiles from hand
        for _ in range(2):
            if not player.remove_tile(self.last_discard):
                raise ValueError(f"Player doesn't have enough tiles for Pong")
        
        # Create meld
        meld_tiles = [self.last_discard] * 3
        meld = Meld(
            meld_type=MeldType.PONG,
            tiles=meld_tiles,
            is_concealed=False,
            source_player=self.last_discard_player,
            source_tile=self.last_discard
        )
        player.declare_meld(meld)
        
        # Clear discard
        self.last_discard = None
        self.last_discard_player = None
        
        # Player must discard
        self.current_player = action.player_idx
        self.phase = GamePhase.DISCARDING
        
        return False, None, None
    
    def _handle_kong(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle claiming a Kong from discard"""
        player = self.players[action.player_idx]
        
        # Remove 3 tiles from hand
        for _ in range(3):
            if not player.remove_tile(self.last_discard):
                raise ValueError(f"Player doesn't have enough tiles for Kong")
        
        # Create meld
        meld_tiles = [self.last_discard] * 4
        meld = Meld(
            meld_type=MeldType.KONG,
            tiles=meld_tiles,
            is_concealed=False,
            source_player=self.last_discard_player,
            source_tile=self.last_discard
        )
        player.declare_meld(meld)
        
        # Clear discard
        self.last_discard = None
        self.last_discard_player = None
        
        # Player draws replacement tile (from back of wall)
        self.current_player = action.player_idx
        self.phase = GamePhase.DRAWING
        
        return False, None, None
    
    def _handle_concealed_kong(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle declaring a concealed Kong"""
        player = self.players[action.player_idx]
        tile = action.tile
        
        # Remove 4 tiles from hand
        for _ in range(4):
            if not player.remove_tile(tile):
                raise ValueError(f"Player doesn't have 4 tiles for concealed Kong")
        
        # Create meld
        meld_tiles = [tile] * 4
        meld = Meld(
            meld_type=MeldType.CONCEALED_KONG,
            tiles=meld_tiles,
            is_concealed=True
        )
        player.declare_meld(meld)
        
        # Player draws replacement tile
        self.phase = GamePhase.DRAWING
        
        return False, None, None
    
    def _handle_add_kong(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle adding to a Pong to make a Kong"""
        player = self.players[action.player_idx]
        tile = action.tile
        
        # Find the Pong to upgrade
        pong_meld = None
        for i, meld in enumerate(player.melds):
            if meld.meld_type == MeldType.PONG and meld.tiles[0] == tile:
                pong_meld = meld
                pong_idx = i
                break
        
        if pong_meld is None:
            raise ValueError(f"No Pong found for tile {tile}")
        
        # Remove tile from hand
        if not player.remove_tile(tile):
            raise ValueError(f"Player doesn't have tile {tile}")
        
        # Upgrade Pong to Kong
        new_meld = Meld(
            meld_type=MeldType.KONG,
            tiles=[tile] * 4,
            is_concealed=False,
            source_player=pong_meld.source_player,
            source_tile=pong_meld.source_tile
        )
        player.melds[pong_idx] = new_meld
        
        # Player draws replacement tile
        self.phase = GamePhase.DRAWING
        
        return False, None, None
    
    def _handle_hu(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle declaring a win"""
        player = self.players[action.player_idx]
        
        # Determine if it's self-drawn (Zimo) or from discard (Ronghu)
        is_zimo = self.last_draw is not None
        winning_tile = self.last_draw if is_zimo else self.last_discard
        
        # Calculate score
        score = 0
        if self.scorer is not None:
            score = self.scorer.calculate_score(
                hand=player.hand,
                melds=player.melds,
                winning_tile=winning_tile,
                is_zimo=is_zimo,
                round_wind=self.round_wind,
                seat_wind=action.player_idx,
                is_last_tile=self.wall.remaining == 0,
                is_kong_draw=(len(self.action_history) > 1 and 
                             self.action_history[-2].action_type in 
                             (ActionType.KONG, ActionType.CONCEALED_KONG, ActionType.ADD_KONG))
            )
        
        # Check minimum score requirement
        if score < self.MIN_WINNING_SCORE:
            raise ValueError(f"Score {score} is below minimum {self.MIN_WINNING_SCORE}")
        
        self.phase = GamePhase.GAME_OVER
        return True, action.player_idx, score
    
    def _handle_pass(self, action: Action) -> Tuple[bool, Optional[int], Optional[int]]:
        """Handle passing on a claim"""
        # Record that this player passed
        self.pending_claims[action.player_idx] = action
        
        # Check if all other players have responded
        claiming_players = set(range(self.NUM_PLAYERS)) - {self.last_discard_player}
        if set(self.pending_claims.keys()) == claiming_players:
            # All players passed, next player draws
            self.current_player = (self.last_discard_player + 1) % self.NUM_PLAYERS
            self.phase = GamePhase.DRAWING
            self.last_discard = None
            self.last_discard_player = None
        
        return False, None, None
    
    def get_valid_actions(self, player_idx: int) -> List[Action]:
        """
        Get all valid actions for a player in current state.
        
        Args:
            player_idx: Index of player
            
        Returns:
            List of valid Action objects
        """
        valid = []
        player = self.players[player_idx]
        
        if self.phase == GamePhase.DRAWING:
            if player_idx == self.current_player:
                valid.append(Action(ActionType.DRAW, player_idx))
        
        elif self.phase == GamePhase.DISCARDING:
            if player_idx == self.current_player:
                # Can discard any tile in hand
                for tile in player.hand.get_unique_tiles():
                    valid.append(Action(ActionType.DISCARD, player_idx, tile))
                
                # Check for concealed Kong
                for tile in player.can_concealed_kong():
                    valid.append(Action(ActionType.CONCEALED_KONG, player_idx, tile))
                
                # Check for adding to Kong
                for tile in player.can_add_to_kong():
                    valid.append(Action(ActionType.ADD_KONG, player_idx, tile))
                
                # Check for self-drawn win (Zimo)
                if self._can_win(player_idx, self.last_draw, is_zimo=True):
                    valid.append(Action(ActionType.HU, player_idx, self.last_draw))
        
        elif self.phase == GamePhase.CLAIMING:
            if player_idx != self.last_discard_player:
                # Check for win (Ron)
                if self._can_win(player_idx, self.last_discard, is_zimo=False):
                    valid.append(Action(ActionType.HU, player_idx, self.last_discard))
                
                # Check for Kong
                if player.can_kong(self.last_discard):
                    valid.append(Action(ActionType.KONG, player_idx, self.last_discard))
                
                # Check for Pong
                if player.can_pong(self.last_discard):
                    valid.append(Action(ActionType.PONG, player_idx, self.last_discard))
                
                # Check for Chow (only from player to your left)
                if (self.last_discard_player + 1) % self.NUM_PLAYERS == player_idx:
                    for pair in player.can_chow(self.last_discard):
                        valid.append(Action(
                            ActionType.CHOW, 
                            player_idx, 
                            self.last_discard,
                            meld_tiles=list(pair)
                        ))
                
                # Can always pass
                valid.append(Action(ActionType.PASS, player_idx))
        
        return valid
    
    def _can_win(self, player_idx: int, winning_tile: Optional[Tile], is_zimo: bool) -> bool:
        """
        Check if player can declare a win with the given tile.
        
        This performs both structure check and minimum score check.
        """
        if winning_tile is None:
            return False
        
        player = self.players[player_idx]
        
        # Add winning tile to hand temporarily for checking
        test_hand = player.hand.copy()
        test_hand.add(winning_tile)
        
        # Check if hand forms a valid winning structure
        if not self._is_valid_winning_hand(test_hand, player.melds):
            return False
        
        # Check minimum score
        if self.scorer is not None:
            score = self.scorer.calculate_score(
                hand=test_hand,
                melds=player.melds,
                winning_tile=winning_tile,
                is_zimo=is_zimo,
                round_wind=self.round_wind,
                seat_wind=player_idx,
                is_last_tile=self.wall.remaining == 0,
                is_kong_draw=False
            )
            return score >= self.MIN_WINNING_SCORE
        
        return True
    
    def _is_valid_winning_hand(self, hand: TileSet, melds: List[Meld]) -> bool:
        """
        Check if a hand forms a valid winning structure.
        
        Standard winning hand: 4 sets (Chow/Pong) + 1 pair
        Special hands: Seven Pairs, Thirteen Orphans, etc.
        """
        counts = hand.to_count_array()
        num_sets_needed = 4 - len(melds)
        
        # Try to find valid decomposition
        return self._can_decompose(counts, num_sets_needed, need_pair=True)
    
    def _can_decompose(self, counts: np.ndarray, sets_needed: int, need_pair: bool) -> bool:
        """
        Recursively check if tiles can be decomposed into sets and a pair.
        
        Args:
            counts: 34-element array of tile counts
            sets_needed: Number of sets still needed
            need_pair: Whether we still need to find the pair
        """
        # Base case: all sets formed, need to find pair
        if sets_needed == 0:
            if not need_pair:
                return np.sum(counts) == 0
            # Find pair
            for i in range(34):
                if counts[i] >= 2:
                    counts[i] -= 2
                    if np.sum(counts) == 0:
                        counts[i] += 2  # Restore for other checks
                        return True
                    counts[i] += 2
            return False
        
        # Find first non-zero tile
        first_idx = -1
        for i in range(34):
            if counts[i] > 0:
                first_idx = i
                break
        
        if first_idx == -1:
            return sets_needed == 0 and not need_pair
        
        # Try Pong (triplet)
        if counts[first_idx] >= 3:
            counts[first_idx] -= 3
            if self._can_decompose(counts, sets_needed - 1, need_pair):
                counts[first_idx] += 3
                return True
            counts[first_idx] += 3
        
        # Try Chow (sequence) - only for numbered suits
        if first_idx < 27:  # Numbered suits
            suit_start = (first_idx // 9) * 9
            value_in_suit = first_idx - suit_start
            
            # Can only form sequence starting with 1-7
            if value_in_suit <= 6:
                idx1, idx2, idx3 = first_idx, first_idx + 1, first_idx + 2
                if counts[idx1] >= 1 and counts[idx2] >= 1 and counts[idx3] >= 1:
                    counts[idx1] -= 1
                    counts[idx2] -= 1
                    counts[idx3] -= 1
                    if self._can_decompose(counts, sets_needed - 1, need_pair):
                        counts[idx1] += 1
                        counts[idx2] += 1
                        counts[idx3] += 1
                        return True
                    counts[idx1] += 1
                    counts[idx2] += 1
                    counts[idx3] += 1
        
        # Try taking a pair here (if we still need one)
        if need_pair and counts[first_idx] >= 2:
            counts[first_idx] -= 2
            if self._can_decompose(counts, sets_needed, need_pair=False):
                counts[first_idx] += 2
                return True
            counts[first_idx] += 2
        
        return False
    
    def get_state(self) -> GameState:
        """Get current game state"""
        return GameState(
            current_player=self.current_player,
            phase=self.phase,
            round_wind=self.round_wind,
            dealer=self.dealer,
            turn_count=self.turn_count,
            wall_remaining=self.wall.remaining,
            last_discard=self.last_discard,
            last_discard_player=self.last_discard_player
        )
    
    def get_observation(self, player_idx: int) -> Dict[str, Any]:
        """
        Get observation for a player (what they can see).
        
        Args:
            player_idx: Index of player getting observation
            
        Returns:
            Dictionary containing all visible information
        """
        player = self.players[player_idx]
        
        # Visible discards from all players
        discards = []
        for p in self.players:
            discards.append(p.discards)
        
        # Visible melds from all players
        melds = []
        for p in self.players:
            melds.append(p.melds)
        
        return {
            "hand": player.hand.to_count_array(),
            "hand_tiles": player.get_hand_tiles(),
            "melds": melds,
            "own_melds": player.melds,
            "discards": discards,
            "last_discard": self.last_discard,
            "last_discard_player": self.last_discard_player,
            "current_player": self.current_player,
            "phase": self.phase,
            "round_wind": self.round_wind,
            "seat_wind": player_idx,
            "wall_remaining": self.wall.remaining,
            "turn_count": self.turn_count,
            "valid_actions": self.get_valid_actions(player_idx)
        }
    
    def copy(self) -> 'Game':
        """Create a deep copy of the game"""
        new_game = Game.__new__(Game)
        new_game.seed = self.seed
        new_game.wall = self.wall.copy()
        new_game.players = [p.copy() for p in self.players]
        new_game.phase = self.phase
        new_game.current_player = self.current_player
        new_game.dealer = self.dealer
        new_game.round_wind = self.round_wind
        new_game.turn_count = self.turn_count
        new_game.last_discard = self.last_discard
        new_game.last_discard_player = self.last_discard_player
        new_game.last_draw = self.last_draw
        new_game.pending_claims = dict(self.pending_claims)
        new_game.action_history = list(self.action_history)
        new_game.scorer = self.scorer
        return new_game
    
    def __repr__(self) -> str:
        return f"Game(phase={self.phase.name}, current={self.current_player}, wall={self.wall.remaining})"

