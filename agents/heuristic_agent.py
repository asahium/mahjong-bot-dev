"""
Heuristic Agent for Riichi Mahjong

A rule-based agent that uses tile efficiency and defensive strategies.
Much stronger than random, provides good baseline for training.

Features:
- Shanten-based tile efficiency
- Ukeire maximization for discards
- Defense mode when opponent riichi
- Basic yaku awareness
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from riichi_mahjong.game import RiichiGame, RiichiAction, RiichiActionType, RiichiPhase
from riichi_mahjong.player import RiichiPlayer
from riichi_mahjong.shanten import ShantenCalculator, calculate_shanten


class HeuristicAgent:
    """
    Heuristic-based Riichi Mahjong agent.
    
    Uses tile efficiency (shanten + ukeire) for offense
    and genbutsu/suji for defense.
    """
    
    # Safety ratings for tiles (lower = safer)
    TERMINAL_SAFETY = 0.3   # 1, 9 tiles
    HONOR_SAFETY = 0.2      # Wind/Dragon tiles
    SIMPLE_2_8_SAFETY = 0.8  # Middle tiles (most dangerous)
    SIMPLE_3_7_SAFETY = 0.6  # Semi-middle tiles
    
    def __init__(self, defense_threshold: float = 0.7):
        """
        Initialize heuristic agent.
        
        Args:
            defense_threshold: Score ratio below which to play defensively
        """
        self.shanten_calc = ShantenCalculator()
        self.defense_threshold = defense_threshold
    
    def get_action(
        self, 
        game: RiichiGame, 
        player_idx: int,
        valid_actions: List[RiichiAction]
    ) -> RiichiAction:
        """
        Select best action based on heuristics.
        
        Args:
            game: Current game state
            player_idx: Index of this player
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        if not valid_actions:
            return RiichiAction(RiichiActionType.PASS, player_idx)
        
        player = game.players[player_idx]
        
        # Check for winning actions first
        for action in valid_actions:
            if action.action_type in (RiichiActionType.TSUMO, RiichiActionType.RON):
                return action
        
        # Check if any opponent is in riichi
        opponents_riichi = any(
            p.is_riichi for i, p in enumerate(game.players) if i != player_idx
        )
        
        # Decide between offense and defense
        should_defend = self._should_defend(game, player_idx, opponents_riichi)
        
        if should_defend:
            return self._defensive_action(game, player_idx, valid_actions)
        else:
            return self._offensive_action(game, player_idx, valid_actions)
    
    def _should_defend(
        self, 
        game: RiichiGame, 
        player_idx: int,
        opponents_riichi: bool
    ) -> bool:
        """Decide whether to play defensively."""
        player = game.players[player_idx]
        
        # Always defend if far from tenpai and opponent riichi
        if opponents_riichi:
            hand_counts = player.get_hand_count_array()
            shanten = calculate_shanten(hand_counts, len(player.melds))
            
            # If 3+ shanten away, focus on defense
            if shanten >= 3:
                return True
            
            # If 2 shanten and late game, also defend
            if shanten >= 2 and game.wall.remaining < 30:
                return True
        
        return False
    
    def _offensive_action(
        self, 
        game: RiichiGame, 
        player_idx: int,
        valid_actions: List[RiichiAction]
    ) -> RiichiAction:
        """Select best offensive action."""
        player = game.players[player_idx]
        hand_counts = player.get_hand_count_array()
        num_melds = len(player.melds)
        
        # Calculate current shanten
        current_result = self.shanten_calc.calculate(hand_counts, num_melds)
        current_shanten = current_result.shanten
        
        # Priority 1: Riichi if tenpai and good wait
        riichi_actions = [a for a in valid_actions if a.action_type == RiichiActionType.RIICHI]
        if riichi_actions and current_shanten == 0:
            # Check ukeire of the wait
            if current_result.ukeire >= 4:  # Reasonable wait
                # Pick best riichi discard
                return self._select_best_riichi_discard(
                    hand_counts, num_melds, riichi_actions
                )
        
        # Priority 2: Kan if it doesn't hurt hand
        for action in valid_actions:
            if action.action_type == RiichiActionType.ANKAN:
                # Ankan is usually good if we have 4 of a tile
                return action
        
        # Priority 3: Consider calls (pon/chii)
        call_action = self._evaluate_calls(
            game, player_idx, valid_actions, hand_counts, num_melds, current_shanten
        )
        if call_action:
            return call_action
        
        # Priority 4: Draw if available
        draw_actions = [a for a in valid_actions if a.action_type == RiichiActionType.DRAW]
        if draw_actions:
            return draw_actions[0]
        
        # Priority 5: Discard for maximum ukeire
        discard_actions = [a for a in valid_actions if a.action_type == RiichiActionType.DISCARD]
        if discard_actions:
            return self._select_best_discard(hand_counts, num_melds, discard_actions)
        
        # Default: pass
        pass_actions = [a for a in valid_actions if a.action_type == RiichiActionType.PASS]
        if pass_actions:
            return pass_actions[0]
        
        return valid_actions[0]
    
    def _defensive_action(
        self, 
        game: RiichiGame, 
        player_idx: int,
        valid_actions: List[RiichiAction]
    ) -> RiichiAction:
        """Select safest defensive action."""
        player = game.players[player_idx]
        
        # Never call in defense mode
        non_call_actions = [
            a for a in valid_actions 
            if a.action_type not in (
                RiichiActionType.CHII, RiichiActionType.PON, 
                RiichiActionType.KAN, RiichiActionType.ANKAN,
                RiichiActionType.SHOUMINKAN
            )
        ]
        
        if not non_call_actions:
            # Must pass on calls
            pass_actions = [a for a in valid_actions if a.action_type == RiichiActionType.PASS]
            if pass_actions:
                return pass_actions[0]
            return valid_actions[0]
        
        # Draw if available
        draw_actions = [a for a in non_call_actions if a.action_type == RiichiActionType.DRAW]
        if draw_actions:
            return draw_actions[0]
        
        # Discard safest tile
        discard_actions = [a for a in non_call_actions if a.action_type == RiichiActionType.DISCARD]
        if discard_actions:
            return self._select_safest_discard(game, player_idx, discard_actions)
        
        # Pass
        pass_actions = [a for a in non_call_actions if a.action_type == RiichiActionType.PASS]
        if pass_actions:
            return pass_actions[0]
        
        return valid_actions[0]
    
    def _select_best_discard(
        self, 
        hand_counts: np.ndarray,
        num_melds: int,
        discard_actions: List[RiichiAction]
    ) -> RiichiAction:
        """Select discard that maximizes ukeire."""
        best_action = discard_actions[0]
        best_score = -1
        
        for action in discard_actions:
            tile_idx = action.tile.tile_index
            
            # Simulate discard
            hand_counts[tile_idx] -= 1
            result = self.shanten_calc.calculate(hand_counts, num_melds)
            hand_counts[tile_idx] += 1
            
            # Score: prioritize lower shanten, then higher ukeire
            score = -result.shanten * 1000 + result.ukeire
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _select_best_riichi_discard(
        self, 
        hand_counts: np.ndarray,
        num_melds: int,
        riichi_actions: List[RiichiAction]
    ) -> RiichiAction:
        """Select best tile to discard with riichi."""
        best_action = riichi_actions[0]
        best_ukeire = -1
        
        for action in riichi_actions:
            tile_idx = action.tile.tile_index
            
            # Simulate discard
            hand_counts[tile_idx] -= 1
            result = self.shanten_calc.calculate(hand_counts, num_melds)
            hand_counts[tile_idx] += 1
            
            # Only consider discards that keep tenpai
            if result.shanten == 0 and result.ukeire > best_ukeire:
                best_ukeire = result.ukeire
                best_action = action
        
        return best_action
    
    def _select_safest_discard(
        self, 
        game: RiichiGame,
        player_idx: int,
        discard_actions: List[RiichiAction]
    ) -> RiichiAction:
        """Select safest tile to discard against riichi."""
        best_action = discard_actions[0]
        best_safety = -999
        
        # Collect genbutsu (100% safe tiles) from riichi players
        genbutsu = set()
        for i, p in enumerate(game.players):
            if i != player_idx and p.is_riichi:
                for tile in p.discards:
                    genbutsu.add(tile.tile_index)
        
        for action in discard_actions:
            tile_idx = action.tile.tile_index
            
            # Genbutsu is completely safe
            if tile_idx in genbutsu:
                return action
            
            # Calculate safety score
            safety = self._calculate_tile_safety(tile_idx, game, player_idx)
            
            if safety > best_safety:
                best_safety = safety
                best_action = action
        
        return best_action
    
    def _calculate_tile_safety(
        self, 
        tile_idx: int,
        game: RiichiGame,
        player_idx: int
    ) -> float:
        """
        Calculate safety score for a tile (higher = safer).
        
        Considers:
        - Tile type (honors/terminals safer)
        - Suji (safe based on discards)
        - Kabe (wall blocks)
        """
        safety = 0.0
        
        # Base safety by tile type
        if tile_idx >= 27:  # Honors
            safety += 2.0
        elif tile_idx % 9 in (0, 8):  # Terminals (1, 9)
            safety += 1.5
        elif tile_idx % 9 in (1, 7):  # 2, 8
            safety += 0.5
        elif tile_idx % 9 in (2, 6):  # 3, 7
            safety += 0.3
        # 4, 5, 6 are most dangerous (0 bonus)
        
        # Suji safety: if 1 is discarded, 4 is safer (can't be part of 123 wait)
        for i, p in enumerate(game.players):
            if i != player_idx and p.is_riichi:
                for discard in p.discards:
                    d_idx = discard.tile_index
                    
                    # Same suit check
                    if tile_idx < 27 and d_idx < 27:
                        if tile_idx // 9 == d_idx // 9:  # Same suit
                            d_val = d_idx % 9
                            t_val = tile_idx % 9
                            
                            # Suji relationships
                            if (d_val == 0 and t_val == 3) or \
                               (d_val == 3 and t_val == 0) or \
                               (d_val == 3 and t_val == 6) or \
                               (d_val == 6 and t_val == 3) or \
                               (d_val == 6 and t_val == 8) or \
                               (d_val == 8 and t_val == 6):
                                safety += 1.0
        
        # Kabe: if 4 of adjacent tiles visible, it's safer
        visible_counts = np.zeros(34, dtype=np.int8)
        for p in game.players:
            for tile in p.discards:
                visible_counts[tile.tile_index] += 1
            for meld in p.melds:
                for tile in meld.tiles:
                    visible_counts[tile.tile_index] += 1
        
        if tile_idx < 27:
            suit_start = (tile_idx // 9) * 9
            val = tile_idx % 9
            
            # Check for kabe (wall)
            if val > 0 and visible_counts[tile_idx - 1] >= 3:
                safety += 0.5
            if val < 8 and visible_counts[tile_idx + 1] >= 3:
                safety += 0.5
        
        return safety
    
    def _evaluate_calls(
        self, 
        game: RiichiGame,
        player_idx: int,
        valid_actions: List[RiichiAction],
        hand_counts: np.ndarray,
        num_melds: int,
        current_shanten: int
    ) -> Optional[RiichiAction]:
        """
        Evaluate whether to make a call (pon/chii).
        
        Returns action if call is beneficial, None otherwise.
        """
        player = game.players[player_idx]
        
        # Don't call if already in riichi
        if player.is_riichi:
            return None
        
        call_actions = [
            a for a in valid_actions 
            if a.action_type in (RiichiActionType.PON, RiichiActionType.CHII)
        ]
        
        if not call_actions:
            return None
        
        best_call = None
        best_improvement = 0
        
        for action in call_actions:
            # Simulate the call
            new_counts = hand_counts.copy()
            
            if action.action_type == RiichiActionType.PON:
                # Remove 2 tiles for pon
                tile_idx = game.last_discard.tile_index
                if new_counts[tile_idx] >= 2:
                    new_counts[tile_idx] -= 2
                else:
                    continue
            elif action.action_type == RiichiActionType.CHII:
                # Remove the meld tiles
                if action.meld_tiles:
                    for tile in action.meld_tiles:
                        if new_counts[tile.tile_index] > 0:
                            new_counts[tile.tile_index] -= 1
                        else:
                            continue
            
            # Calculate new shanten
            new_result = self.shanten_calc.calculate(new_counts, num_melds + 1)
            new_shanten = new_result.shanten
            
            # Only call if it improves shanten
            improvement = current_shanten - new_shanten
            
            if improvement > 0 and improvement > best_improvement:
                # Additional checks
                # Don't call for honors unless yakuhai
                if action.action_type == RiichiActionType.PON:
                    tile_idx = game.last_discard.tile_index
                    if tile_idx >= 27:  # Honor
                        # Only call for yakuhai (value tiles)
                        if not self._is_yakuhai(tile_idx, game, player_idx):
                            continue
                
                best_improvement = improvement
                best_call = action
        
        return best_call
    
    def _is_yakuhai(self, tile_idx: int, game: RiichiGame, player_idx: int) -> bool:
        """Check if honor tile is yakuhai (gives value)."""
        if tile_idx < 27:
            return False
        
        # Dragons are always yakuhai
        if tile_idx >= 31:
            return True
        
        # Wind yakuhai
        wind_idx = tile_idx - 27  # 0=E, 1=S, 2=W, 3=N
        
        # Round wind
        if wind_idx == game.round_wind:
            return True
        
        # Seat wind
        if wind_idx == player_idx:
            return True
        
        return False


def create_heuristic_policy(defense_threshold: float = 0.7):
    """
    Create a heuristic policy function for use in environments.
    
    Returns a function that takes (game, player_idx, valid_actions) and returns an action.
    """
    agent = HeuristicAgent(defense_threshold=defense_threshold)
    
    def policy(game: RiichiGame, player_idx: int, valid_actions: List[RiichiAction]) -> RiichiAction:
        return agent.get_action(game, player_idx, valid_actions)
    
    return policy


if __name__ == "__main__":
    # Quick test
    from riichi_mahjong.game import RiichiGame
    from riichi_mahjong.rules import TENHOU_RULES
    
    print("Testing HeuristicAgent...")
    
    game = RiichiGame(rules=TENHOU_RULES, seed=42)
    game.reset()
    game.start_round()
    
    agent = HeuristicAgent()
    
    # Play a few turns
    for turn in range(20):
        current = game.current_player
        valid_actions = game.get_valid_actions(current)
        
        if not valid_actions:
            print(f"Turn {turn}: No valid actions for P{current}")
            break
        
        action = agent.get_action(game, current, valid_actions)
        print(f"Turn {turn}: P{current} -> {action.action_type.name}")
        
        over, result = game.step(action)
        if over:
            print(f"Game over! Result: {result}")
            break
    
    print("\nTest completed!")

