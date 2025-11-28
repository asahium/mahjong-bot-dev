"""
MCR Mahjong Gymnasium Environment

A Gymnasium-compatible environment for training RL agents to play MCR Mahjong.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcr_mahjong.tiles import Tile, TileSet, TileSuit
from mcr_mahjong.player import Player, Meld, MeldType
from mcr_mahjong.game import Game, GamePhase, Action, ActionType
from mcr_mahjong.scoring import MCRScorer


class MCRMahjongEnv(gym.Env):
    """
    MCR Mahjong Environment for Reinforcement Learning.
    
    This environment simulates a 4-player MCR Mahjong game where the agent
    controls one player and the other three are controlled by simple policies
    (random or rule-based).
    
    Observation Space:
        A dictionary containing:
        - hand: (34,) int8 - Count of each tile type in hand
        - melds: (4, 4, 34) int8 - Melds for each player (max 4 melds, 34 tile counts)
        - discards: (4, 34) int8 - Discard pile counts for each player
        - last_discard: (34,) int8 - One-hot encoding of last discarded tile
        - valid_actions: (235,) int8 - Binary mask of valid actions
        - game_info: (8,) float32 - [current_player, phase, round_wind, seat_wind, 
                                     wall_remaining, turn_count, is_my_turn, can_win]
    
    Action Space:
        Discrete(235):
        - 0-33: Discard tile type 0-33
        - 34-67: Chow with starting tile 0-33 (only valid tiles)
        - 68-101: Pong tile 0-33
        - 102-135: Kong tile 0-33
        - 136-169: Concealed Kong tile 0-33
        - 170-203: Add to Kong tile 0-33
        - 204: Declare Hu (win)
        - 205: Pass
        - 206-233: Draw (placeholder for action masking)
        - 234: No-op (for invalid states)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    # Action space constants
    NUM_TILE_TYPES = 34
    ACTION_DISCARD_START = 0
    ACTION_CHOW_START = 34
    ACTION_PONG_START = 68
    ACTION_KONG_START = 102
    ACTION_CONCEALED_KONG_START = 136
    ACTION_ADD_KONG_START = 170
    ACTION_HU = 204
    ACTION_PASS = 205
    ACTION_DRAW = 206
    ACTION_NOOP = 234
    NUM_ACTIONS = 235
    
    def __init__(
        self,
        player_idx: int = 0,
        opponent_policy: str = "random",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        reward_shaping: bool = True,
    ):
        """
        Initialize the MCR Mahjong environment.
        
        Args:
            player_idx: Index of the agent's player (0-3)
            opponent_policy: Policy for opponents ("random" or "greedy")
            seed: Random seed for reproducibility
            render_mode: Rendering mode ("human" or "ansi")
            reward_shaping: Whether to use reward shaping
        """
        super().__init__()
        
        self.player_idx = player_idx
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        
        # Create game and scorer
        self.game = Game(seed=seed)
        self.scorer = MCRScorer()
        self.game.scorer = self.scorer
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(low=0, high=4, shape=(34,), dtype=np.int8),
            "melds": spaces.Box(low=0, high=4, shape=(4, 4, 34), dtype=np.int8),
            "discards": spaces.Box(low=0, high=4, shape=(4, 34), dtype=np.int8),
            "last_discard": spaces.Box(low=0, high=1, shape=(34,), dtype=np.int8),
            "valid_actions": spaces.Box(low=0, high=1, shape=(self.NUM_ACTIONS,), dtype=np.int8),
            "game_info": spaces.Box(low=-1, high=200, shape=(8,), dtype=np.float32),
        })
        
        # Define action space
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        
        # Track episode stats
        self._episode_reward = 0
        self._episode_length = 0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset the environment to start a new game.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset game
        self.game.reset(seed=seed)
        self.game.start_game()
        
        # Reset episode tracking
        self._episode_reward = 0
        self._episode_length = 0
        
        # Run opponent turns until it's agent's turn
        self._run_opponents_until_agent_turn()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index from action space
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._episode_length += 1
        
        # Convert action index to game Action
        game_action = self._action_to_game_action(action)
        
        # Validate action
        valid_actions = self.game.get_valid_actions(self.player_idx)
        if not self._is_valid_action(game_action, valid_actions):
            # Invalid action - apply penalty and choose a random valid action
            reward = -1.0
            if valid_actions:
                game_action = np.random.choice(valid_actions)
            else:
                # No valid actions (shouldn't happen normally)
                obs = self._get_observation()
                return obs, reward, False, False, self._get_info()
        else:
            reward = 0.0
        
        # Execute action
        game_over, winner, score = self.game.step(game_action)
        
        # Calculate reward
        if game_over:
            if winner == self.player_idx:
                # Agent won
                reward += score / 10.0  # Scale reward
            elif winner is not None:
                # Another player won
                reward -= score / 20.0
            # Draw game: reward = 0
        elif self.reward_shaping:
            # Small reward shaping for good play
            reward += self._calculate_shaping_reward(game_action)
        
        self._episode_reward += reward
        
        # Run opponent turns
        if not game_over:
            opponent_game_over, opponent_winner, opponent_score = self._run_opponents_until_agent_turn()
            if opponent_game_over:
                game_over = True
                winner = opponent_winner
                if winner is not None and winner != self.player_idx:
                    reward -= opponent_score / 20.0
        
        terminated = game_over
        truncated = self._episode_length > 1000  # Safety limit
        
        obs = self._get_observation()
        info = self._get_info()
        
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
                "winner": winner,
            }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation for the agent."""
        player = self.game.players[self.player_idx]
        
        # Hand tiles (34 tile types)
        hand = player.get_hand_count_array()
        
        # Melds for all players (4 players x max 4 melds x 34 tiles)
        melds = np.zeros((4, 4, 34), dtype=np.int8)
        for p_idx, p in enumerate(self.game.players):
            for m_idx, meld in enumerate(p.melds[:4]):
                for tile in meld.tiles:
                    melds[p_idx, m_idx, tile.tile_index] += 1
        
        # Discards for all players (4 x 34)
        discards = np.zeros((4, 34), dtype=np.int8)
        for p_idx, p in enumerate(self.game.players):
            for tile in p.discards:
                discards[p_idx, tile.tile_index] += 1
        
        # Last discard (one-hot)
        last_discard = np.zeros(34, dtype=np.int8)
        if self.game.last_discard is not None:
            last_discard[self.game.last_discard.tile_index] = 1
        
        # Valid actions mask
        valid_actions = self._get_valid_actions_mask()
        
        # Game info
        can_win = any(a.action_type == ActionType.HU 
                      for a in self.game.get_valid_actions(self.player_idx))
        game_info = np.array([
            self.game.current_player,
            self.game.phase.value,
            self.game.round_wind,
            self.player_idx,  # seat wind
            self.game.wall.remaining,
            self.game.turn_count,
            1 if self.game.current_player == self.player_idx else 0,
            1 if can_win else 0,
        ], dtype=np.float32)
        
        return {
            "hand": hand,
            "melds": melds,
            "discards": discards,
            "last_discard": last_discard,
            "valid_actions": valid_actions,
            "game_info": game_info,
        }
    
    def _get_valid_actions_mask(self) -> np.ndarray:
        """Get binary mask of valid actions."""
        mask = np.zeros(self.NUM_ACTIONS, dtype=np.int8)
        
        valid_actions = self.game.get_valid_actions(self.player_idx)
        
        for action in valid_actions:
            action_idx = self._game_action_to_idx(action)
            if action_idx is not None and 0 <= action_idx < self.NUM_ACTIONS:
                mask[action_idx] = 1
        
        return mask
    
    def _game_action_to_idx(self, action: Action) -> Optional[int]:
        """Convert game Action to action index."""
        if action.action_type == ActionType.DRAW:
            return self.ACTION_DRAW
        elif action.action_type == ActionType.DISCARD:
            if action.tile:
                return self.ACTION_DISCARD_START + action.tile.tile_index
        elif action.action_type == ActionType.CHOW:
            if action.meld_tiles:
                # Use the smallest tile in the meld as the identifier
                min_tile = min(action.meld_tiles, key=lambda t: t.tile_index)
                return self.ACTION_CHOW_START + min_tile.tile_index
        elif action.action_type == ActionType.PONG:
            if action.tile:
                return self.ACTION_PONG_START + action.tile.tile_index
        elif action.action_type == ActionType.KONG:
            if action.tile:
                return self.ACTION_KONG_START + action.tile.tile_index
        elif action.action_type == ActionType.CONCEALED_KONG:
            if action.tile:
                return self.ACTION_CONCEALED_KONG_START + action.tile.tile_index
        elif action.action_type == ActionType.ADD_KONG:
            if action.tile:
                return self.ACTION_ADD_KONG_START + action.tile.tile_index
        elif action.action_type == ActionType.HU:
            return self.ACTION_HU
        elif action.action_type == ActionType.PASS:
            return self.ACTION_PASS
        
        return None
    
    def _action_to_game_action(self, action_idx: int) -> Action:
        """Convert action index to game Action."""
        player_idx = self.player_idx
        
        if self.ACTION_DISCARD_START <= action_idx < self.ACTION_CHOW_START:
            tile_idx = action_idx - self.ACTION_DISCARD_START
            tile = Tile.from_index(tile_idx)
            return Action(ActionType.DISCARD, player_idx, tile)
        
        elif self.ACTION_CHOW_START <= action_idx < self.ACTION_PONG_START:
            tile_idx = action_idx - self.ACTION_CHOW_START
            # Need to find valid chow options and match
            valid_actions = self.game.get_valid_actions(player_idx)
            for a in valid_actions:
                if a.action_type == ActionType.CHOW and a.meld_tiles:
                    min_tile = min(a.meld_tiles, key=lambda t: t.tile_index)
                    if min_tile.tile_index == tile_idx:
                        return a
            # Fallback to pass if chow not valid
            return Action(ActionType.PASS, player_idx)
        
        elif self.ACTION_PONG_START <= action_idx < self.ACTION_KONG_START:
            tile_idx = action_idx - self.ACTION_PONG_START
            tile = Tile.from_index(tile_idx)
            return Action(ActionType.PONG, player_idx, tile)
        
        elif self.ACTION_KONG_START <= action_idx < self.ACTION_CONCEALED_KONG_START:
            tile_idx = action_idx - self.ACTION_KONG_START
            tile = Tile.from_index(tile_idx)
            return Action(ActionType.KONG, player_idx, tile)
        
        elif self.ACTION_CONCEALED_KONG_START <= action_idx < self.ACTION_ADD_KONG_START:
            tile_idx = action_idx - self.ACTION_CONCEALED_KONG_START
            tile = Tile.from_index(tile_idx)
            return Action(ActionType.CONCEALED_KONG, player_idx, tile)
        
        elif self.ACTION_ADD_KONG_START <= action_idx < self.ACTION_HU:
            tile_idx = action_idx - self.ACTION_ADD_KONG_START
            tile = Tile.from_index(tile_idx)
            return Action(ActionType.ADD_KONG, player_idx, tile)
        
        elif action_idx == self.ACTION_HU:
            winning_tile = self.game.last_draw or self.game.last_discard
            return Action(ActionType.HU, player_idx, winning_tile)
        
        elif action_idx == self.ACTION_PASS:
            return Action(ActionType.PASS, player_idx)
        
        elif action_idx == self.ACTION_DRAW:
            return Action(ActionType.DRAW, player_idx)
        
        # Default to pass for invalid actions
        return Action(ActionType.PASS, player_idx)
    
    def _is_valid_action(self, action: Action, valid_actions: List[Action]) -> bool:
        """Check if an action is in the list of valid actions."""
        for va in valid_actions:
            if va.action_type == action.action_type:
                if action.action_type in (ActionType.DRAW, ActionType.PASS, ActionType.HU):
                    return True
                elif action.action_type == ActionType.CHOW:
                    if action.meld_tiles and va.meld_tiles:
                        # Compare meld tiles
                        return set(t.tile_index for t in action.meld_tiles) == \
                               set(t.tile_index for t in va.meld_tiles)
                elif action.tile and va.tile:
                    return action.tile.tile_index == va.tile.tile_index
        return False
    
    def _run_opponents_until_agent_turn(self) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Run opponent turns until it's the agent's turn.
        
        Returns:
            Tuple of (game_over, winner, score)
        """
        max_steps = 200  # Prevent infinite loops
        steps = 0
        
        while steps < max_steps:
            steps += 1
            
            if self.game.phase == GamePhase.GAME_OVER:
                return True, None, None
            
            # Check if it's agent's turn to act with meaningful actions
            agent_actions = self.game.get_valid_actions(self.player_idx)
            meaningful_agent_actions = [a for a in agent_actions 
                                        if a.action_type != ActionType.PASS]
            
            if meaningful_agent_actions:
                # Agent has meaningful actions to take
                return False, None, None
            
            # In claiming phase, handle all passes
            if self.game.phase == GamePhase.CLAIMING:
                # Agent passes if they can only pass
                if agent_actions and all(a.action_type == ActionType.PASS for a in agent_actions):
                    pass_action = Action(ActionType.PASS, self.player_idx)
                    self.game.step(pass_action)
                
                # All opponents pass
                for i in range(4):
                    if i != self.player_idx and i != self.game.last_discard_player:
                        opp_actions = self.game.get_valid_actions(i)
                        if opp_actions:
                            # Opponent chooses their action
                            action = self._get_opponent_action(i, opp_actions)
                            game_over, winner, score = self.game.step(action)
                            if game_over:
                                return True, winner, score
                
                continue
            
            # In drawing phase, if it's opponent's turn to draw
            if self.game.phase == GamePhase.DRAWING:
                if self.game.current_player != self.player_idx:
                    opp_idx = self.game.current_player
                    opp_actions = self.game.get_valid_actions(opp_idx)
                    if opp_actions:
                        action = self._get_opponent_action(opp_idx, opp_actions)
                        game_over, winner, score = self.game.step(action)
                        if game_over:
                            return True, winner, score
                        continue
                else:
                    # It's agent's turn to draw
                    return False, None, None
            
            # In discarding phase, if it's opponent's turn to discard
            if self.game.phase == GamePhase.DISCARDING:
                if self.game.current_player != self.player_idx:
                    opp_idx = self.game.current_player
                    opp_actions = self.game.get_valid_actions(opp_idx)
                    if opp_actions:
                        action = self._get_opponent_action(opp_idx, opp_actions)
                        game_over, winner, score = self.game.step(action)
                        if game_over:
                            return True, winner, score
                        continue
                else:
                    # It's agent's turn to discard
                    return False, None, None
            
            # Safety: if we get here with no action taken, break
            break
        
        return False, None, None
    
    def _get_opponent_action(self, player_idx: int, valid_actions: List[Action]) -> Action:
        """Get action for an opponent based on policy."""
        if self.opponent_policy == "random":
            return self._random_policy(valid_actions)
        elif self.opponent_policy == "greedy":
            return self._greedy_policy(player_idx, valid_actions)
        else:
            return self._random_policy(valid_actions)
    
    def _random_policy(self, valid_actions: List[Action]) -> Action:
        """Random action selection."""
        # Prioritize meaningful actions over pass
        meaningful = [a for a in valid_actions if a.action_type != ActionType.PASS]
        if meaningful:
            return np.random.choice(meaningful)
        return np.random.choice(valid_actions)
    
    def _greedy_policy(self, player_idx: int, valid_actions: List[Action]) -> Action:
        """Simple greedy policy - always claim, always win."""
        # Prioritize: Hu > Kong > Pong > Chow > Draw > Discard > Pass
        priority = {
            ActionType.HU: 0,
            ActionType.KONG: 1,
            ActionType.CONCEALED_KONG: 1,
            ActionType.ADD_KONG: 1,
            ActionType.PONG: 2,
            ActionType.CHOW: 3,
            ActionType.DRAW: 4,
            ActionType.DISCARD: 5,
            ActionType.PASS: 6,
        }
        
        sorted_actions = sorted(valid_actions, key=lambda a: priority.get(a.action_type, 10))
        
        # For discard, choose randomly from hand
        if sorted_actions[0].action_type == ActionType.DISCARD:
            discards = [a for a in valid_actions if a.action_type == ActionType.DISCARD]
            return np.random.choice(discards)
        
        return sorted_actions[0]
    
    def _calculate_shaping_reward(self, action: Action) -> float:
        """Calculate reward shaping bonus."""
        reward = 0.0
        
        if action.action_type == ActionType.PONG:
            reward += 0.05
        elif action.action_type == ActionType.KONG:
            reward += 0.1
        elif action.action_type == ActionType.CHOW:
            reward += 0.02
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the environment state."""
        return {
            "turn": self.game.turn_count,
            "phase": self.game.phase.name,
            "current_player": self.game.current_player,
            "wall_remaining": self.game.wall.remaining,
        }
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
        return None
    
    def _render_human(self):
        """Render to console."""
        print(self._render_ansi())
    
    def _render_ansi(self) -> str:
        """Render as ASCII string."""
        lines = []
        lines.append(f"=== MCR Mahjong - Turn {self.game.turn_count} ===")
        lines.append(f"Phase: {self.game.phase.name}")
        lines.append(f"Current Player: {self.game.current_player}")
        lines.append(f"Wall Remaining: {self.game.wall.remaining}")
        
        if self.game.last_discard:
            lines.append(f"Last Discard: {self.game.last_discard} (P{self.game.last_discard_player})")
        
        lines.append("")
        lines.append(f"--- Your Hand (Player {self.player_idx}) ---")
        player = self.game.players[self.player_idx]
        lines.append(f"Hand: {player.hand}")
        
        if player.melds:
            melds_str = " | ".join(str(m) for m in player.melds)
            lines.append(f"Melds: {melds_str}")
        
        lines.append("")
        lines.append("--- Valid Actions ---")
        valid = self.game.get_valid_actions(self.player_idx)
        for a in valid[:10]:  # Limit display
            lines.append(f"  {a}")
        if len(valid) > 10:
            lines.append(f"  ... and {len(valid) - 10} more")
        
        return "\n".join(lines)
    
    def close(self):
        """Clean up resources."""
        pass


# Register the environment
def register_envs():
    """Register MCR Mahjong environments with Gymnasium."""
    gym.register(
        id="MCRMahjong-v0",
        entry_point="envs.mcr_env:MCRMahjongEnv",
        max_episode_steps=1000,
    )


if __name__ == "__main__":
    # Quick test
    env = MCRMahjongEnv(render_mode="human")
    obs, info = env.reset()
    
    print("Observation keys:", obs.keys())
    print("Hand shape:", obs["hand"].shape)
    print("Valid actions:", np.sum(obs["valid_actions"]), "available")
    
    env.render()
    
    # Take a few random actions
    for _ in range(5):
        valid_mask = obs["valid_actions"]
        valid_indices = np.where(valid_mask == 1)[0]
        if len(valid_indices) > 0:
            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            if terminated or truncated:
                print("Game ended!")
                break
        else:
            print("No valid actions!")
            break
    
    env.close()

