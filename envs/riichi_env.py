"""
Riichi Mahjong Gymnasium Environment

A Gymnasium-compatible environment for training RL agents to play Riichi Mahjong.
Supports both EMA and Tenhou rule variations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcr_mahjong.tiles import Tile, TileSet, TileSuit
from mcr_mahjong.player import Meld, MeldType

from riichi_mahjong.game import RiichiGame, RiichiPhase, RiichiAction, RiichiActionType
from riichi_mahjong.player import RiichiPlayer, PlayerState
from riichi_mahjong.scoring import RiichiScorer
from riichi_mahjong.rules import RuleSet, TENHOU_RULES, EMA_RULES


class RiichiMahjongEnv(gym.Env):
    """
    Riichi Mahjong Environment for Reinforcement Learning.
    
    Observation Space:
        A dictionary containing:
        - hand: (34,) int8 - Count of each tile type in hand
        - melds: (4, 4, 34) int8 - Melds for each player
        - discards: (4, 34) int8 - Discard counts for each player
        - dora_indicators: (5, 34) int8 - Dora indicator tiles (one-hot)
        - riichi_status: (4,) int8 - Riichi status per player
        - furiten: (1,) int8 - Whether agent is in furiten
        - scores: (4,) float32 - Current scores
        - valid_actions: (250,) int8 - Binary mask of valid actions
        - game_info: (12,) float32 - Game state info
    
    Action Space:
        Discrete(250):
        - 0-33: Discard tile type
        - 34-67: Riichi + Discard tile type
        - 68-101: Chii (call sequence)
        - 102-135: Pon (call triplet)
        - 136-169: Kan (open)
        - 170-203: Ankan (concealed)
        - 204-237: Shouminkan (add to pon)
        - 238: Tsumo
        - 239: Ron
        - 240: Pass
        - 241: Draw
        - 242-249: Reserved
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    # Action space constants
    NUM_TILE_TYPES = 34
    ACTION_DISCARD_START = 0
    ACTION_RIICHI_START = 34
    ACTION_CHII_START = 68
    ACTION_PON_START = 102
    ACTION_KAN_START = 136
    ACTION_ANKAN_START = 170
    ACTION_SHOUMINKAN_START = 204
    ACTION_TSUMO = 238
    ACTION_RON = 239
    ACTION_PASS = 240
    ACTION_DRAW = 241
    NUM_ACTIONS = 250
    
    def __init__(
        self,
        player_idx: int = 0,
        rules: str = "tenhou",  # "tenhou" or "ema"
        opponent_policy: str = "random",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        reward_shaping: bool = True,
    ):
        """
        Initialize the Riichi Mahjong environment.
        
        Args:
            player_idx: Index of the agent's player (0-3)
            rules: Rule set to use ("tenhou" or "ema")
            opponent_policy: Policy for opponents ("random" or "greedy")
            seed: Random seed
            render_mode: Rendering mode
            reward_shaping: Whether to use reward shaping
        """
        super().__init__()
        
        self.player_idx = player_idx
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        
        # Select rule set
        self.rules = TENHOU_RULES if rules == "tenhou" else EMA_RULES
        
        # Create game and scorer
        self.game = RiichiGame(rules=self.rules, seed=seed)
        self.scorer = RiichiScorer(rules=self.rules)
        self.game.scorer = self.scorer
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "hand": spaces.Box(low=0, high=4, shape=(34,), dtype=np.int8),
            "melds": spaces.Box(low=0, high=4, shape=(4, 4, 34), dtype=np.int8),
            "discards": spaces.Box(low=0, high=4, shape=(4, 34), dtype=np.int8),
            "dora_indicators": spaces.Box(low=0, high=1, shape=(5, 34), dtype=np.int8),
            "riichi_status": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
            "furiten": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "scores": spaces.Box(low=-200000, high=200000, shape=(4,), dtype=np.float32),
            "valid_actions": spaces.Box(low=0, high=1, shape=(self.NUM_ACTIONS,), dtype=np.int8),
            "game_info": spaces.Box(low=-1, high=200, shape=(12,), dtype=np.float32),
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
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.game.reset(seed=seed)
        self.game.start_round()
        
        self._episode_reward = 0
        self._episode_length = 0
        
        # Run opponent turns until agent's turn
        self._run_opponents_until_agent_turn()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Take a step in the environment."""
        self._episode_length += 1
        
        # Convert action to game action
        game_action = self._action_to_game_action(action)
        
        # Validate action
        valid_actions = self.game.get_valid_actions(self.player_idx)
        if not self._is_valid_action(game_action, valid_actions):
            reward = -1.0
            if valid_actions:
                # Take a random valid action instead
                import random
                game_action = random.choice(valid_actions)
            else:
                obs = self._get_observation()
                return obs, reward, False, False, self._get_info()
        else:
            reward = 0.0
        
        # Execute action
        round_over, result = self.game.step(game_action)
        
        # Calculate reward
        if round_over and result and hasattr(result, 'winner'):
            if result.winner == self.player_idx:
                reward += getattr(result, 'score', 0) / 1000.0
            elif result.winner is not None:
                reward -= getattr(result, 'score', 0) / 2000.0
        elif self.reward_shaping:
            reward += self._calculate_shaping_reward(game_action)
        
        self._episode_reward += reward
        
        # Run opponent turns
        if not round_over:
            opp_over, opp_result = self._run_opponents_until_agent_turn()
            if opp_over:
                round_over = True
                result = opp_result
                # Check for winner attribute (RoundResult has it, ScoringResult doesn't)
                if result and hasattr(result, 'winner') and result.winner is not None:
                    if result.winner != self.player_idx:
                        reward -= getattr(result, 'score', 0) / 2000.0
        
        terminated = round_over
        truncated = self._episode_length > 1000
        
        obs = self._get_observation()
        info = self._get_info()
        
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
                "result": result,
            }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation for the agent."""
        player = self.game.players[self.player_idx]
        
        # Hand (34 tile types)
        hand = player.get_hand_count_array()
        
        # Melds (4 players x 4 melds x 34 tiles)
        melds = np.zeros((4, 4, 34), dtype=np.int8)
        for p_idx, p in enumerate(self.game.players):
            for m_idx, meld in enumerate(p.melds[:4]):
                for tile in meld.tiles:
                    melds[p_idx, m_idx, tile.tile_index] += 1
        
        # Discards (4 x 34)
        discards = np.zeros((4, 34), dtype=np.int8)
        for p_idx, p in enumerate(self.game.players):
            for tile in p.discards:
                discards[p_idx, tile.tile_index] += 1
        
        # Dora indicators (5 x 34, one-hot)
        dora_indicators = np.zeros((5, 34), dtype=np.int8)
        for i, ind in enumerate(self.game.dora_indicators[:5]):
            dora_indicators[i, ind.tile_index] = 1
        
        # Riichi status
        riichi_status = np.array(
            [1 if p.is_riichi else 0 for p in self.game.players],
            dtype=np.int8
        )
        
        # Furiten
        furiten = np.array([1 if player.furiten.is_furiten else 0], dtype=np.int8)
        
        # Scores
        scores = np.array([p.score for p in self.game.players], dtype=np.float32)
        
        # Valid actions
        valid_actions = self._get_valid_actions_mask()
        
        # Game info
        can_tsumo = any(a.action_type == RiichiActionType.TSUMO 
                       for a in self.game.get_valid_actions(self.player_idx))
        can_ron = any(a.action_type == RiichiActionType.RON
                     for a in self.game.get_valid_actions(self.player_idx))
        
        game_info = np.array([
            self.game.current_player,
            self.game.phase.value,
            self.game.round_wind,
            self.player_idx,
            self.game.dealer,
            self.game.honba,
            self.game.riichi_sticks,
            self.game.wall.remaining,
            self.game.turn_count,
            1 if self.game.current_player == self.player_idx else 0,
            1 if can_tsumo else 0,
            1 if can_ron else 0,
        ], dtype=np.float32)
        
        return {
            "hand": hand,
            "melds": melds,
            "discards": discards,
            "dora_indicators": dora_indicators,
            "riichi_status": riichi_status,
            "furiten": furiten,
            "scores": scores,
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
    
    def _game_action_to_idx(self, action: RiichiAction) -> Optional[int]:
        """Convert game action to action index."""
        if action.action_type == RiichiActionType.DRAW:
            return self.ACTION_DRAW
        elif action.action_type == RiichiActionType.DISCARD:
            if action.tile:
                return self.ACTION_DISCARD_START + action.tile.tile_index
        elif action.action_type == RiichiActionType.RIICHI:
            if action.tile:
                return self.ACTION_RIICHI_START + action.tile.tile_index
        elif action.action_type == RiichiActionType.CHII:
            if action.meld_tiles:
                min_tile = min(action.meld_tiles, key=lambda t: t.tile_index)
                return self.ACTION_CHII_START + min_tile.tile_index
        elif action.action_type == RiichiActionType.PON:
            if action.tile:
                return self.ACTION_PON_START + action.tile.tile_index
        elif action.action_type == RiichiActionType.KAN:
            if action.tile:
                return self.ACTION_KAN_START + action.tile.tile_index
        elif action.action_type == RiichiActionType.ANKAN:
            if action.tile:
                return self.ACTION_ANKAN_START + action.tile.tile_index
        elif action.action_type == RiichiActionType.SHOUMINKAN:
            if action.tile:
                return self.ACTION_SHOUMINKAN_START + action.tile.tile_index
        elif action.action_type == RiichiActionType.TSUMO:
            return self.ACTION_TSUMO
        elif action.action_type == RiichiActionType.RON:
            return self.ACTION_RON
        elif action.action_type == RiichiActionType.PASS:
            return self.ACTION_PASS
        
        return None
    
    def _action_to_game_action(self, action_idx: int) -> RiichiAction:
        """Convert action index to game action."""
        player_idx = self.player_idx
        
        if self.ACTION_DISCARD_START <= action_idx < self.ACTION_RIICHI_START:
            tile_idx = action_idx - self.ACTION_DISCARD_START
            tile = Tile.from_index(tile_idx)
            return RiichiAction(RiichiActionType.DISCARD, player_idx, tile)
        
        elif self.ACTION_RIICHI_START <= action_idx < self.ACTION_CHII_START:
            tile_idx = action_idx - self.ACTION_RIICHI_START
            tile = Tile.from_index(tile_idx)
            return RiichiAction(RiichiActionType.RIICHI, player_idx, tile)
        
        elif self.ACTION_CHII_START <= action_idx < self.ACTION_PON_START:
            tile_idx = action_idx - self.ACTION_CHII_START
            valid_actions = self.game.get_valid_actions(player_idx)
            for a in valid_actions:
                if a.action_type == RiichiActionType.CHII and a.meld_tiles:
                    min_tile = min(a.meld_tiles, key=lambda t: t.tile_index)
                    if min_tile.tile_index == tile_idx:
                        return a
            return RiichiAction(RiichiActionType.PASS, player_idx)
        
        elif self.ACTION_PON_START <= action_idx < self.ACTION_KAN_START:
            tile_idx = action_idx - self.ACTION_PON_START
            tile = Tile.from_index(tile_idx)
            return RiichiAction(RiichiActionType.PON, player_idx, tile)
        
        elif self.ACTION_KAN_START <= action_idx < self.ACTION_ANKAN_START:
            tile_idx = action_idx - self.ACTION_KAN_START
            tile = Tile.from_index(tile_idx)
            return RiichiAction(RiichiActionType.KAN, player_idx, tile)
        
        elif self.ACTION_ANKAN_START <= action_idx < self.ACTION_SHOUMINKAN_START:
            tile_idx = action_idx - self.ACTION_ANKAN_START
            tile = Tile.from_index(tile_idx)
            return RiichiAction(RiichiActionType.ANKAN, player_idx, tile)
        
        elif self.ACTION_SHOUMINKAN_START <= action_idx < self.ACTION_TSUMO:
            tile_idx = action_idx - self.ACTION_SHOUMINKAN_START
            tile = Tile.from_index(tile_idx)
            return RiichiAction(RiichiActionType.SHOUMINKAN, player_idx, tile)
        
        elif action_idx == self.ACTION_TSUMO:
            return RiichiAction(RiichiActionType.TSUMO, player_idx, self.game.last_draw)
        
        elif action_idx == self.ACTION_RON:
            return RiichiAction(RiichiActionType.RON, player_idx, self.game.last_discard)
        
        elif action_idx == self.ACTION_PASS:
            return RiichiAction(RiichiActionType.PASS, player_idx)
        
        elif action_idx == self.ACTION_DRAW:
            return RiichiAction(RiichiActionType.DRAW, player_idx)
        
        return RiichiAction(RiichiActionType.PASS, player_idx)
    
    def _is_valid_action(self, action: RiichiAction, valid_actions: List[RiichiAction]) -> bool:
        """Check if action is valid."""
        for va in valid_actions:
            if va.action_type == action.action_type:
                if action.action_type in (RiichiActionType.DRAW, RiichiActionType.PASS,
                                         RiichiActionType.TSUMO, RiichiActionType.RON):
                    return True
                elif action.action_type == RiichiActionType.CHII:
                    if action.meld_tiles and va.meld_tiles:
                        return set(t.tile_index for t in action.meld_tiles) == \
                               set(t.tile_index for t in va.meld_tiles)
                elif action.tile and va.tile:
                    return action.tile.tile_index == va.tile.tile_index
        return False
    
    def _run_opponents_until_agent_turn(self) -> Tuple[bool, Any]:
        """Run opponent turns until agent's turn."""
        max_steps = 200
        steps = 0
        
        while steps < max_steps:
            steps += 1
            
            if self.game.phase == RiichiPhase.GAME_OVER:
                return True, None
            
            # Check if agent has meaningful actions
            agent_actions = self.game.get_valid_actions(self.player_idx)
            meaningful = [a for a in agent_actions if a.action_type != RiichiActionType.PASS]
            
            if meaningful:
                return False, None
            
            # Handle claiming phase
            if self.game.phase == RiichiPhase.CLAIMING:
                if agent_actions and all(a.action_type == RiichiActionType.PASS for a in agent_actions):
                    pass_action = RiichiAction(RiichiActionType.PASS, self.player_idx)
                    self.game.step(pass_action)
                
                for i in range(4):
                    if i != self.player_idx and i != self.game.last_discard_player:
                        opp_actions = self.game.get_valid_actions(i)
                        if opp_actions:
                            action = self._get_opponent_action(i, opp_actions)
                            over, result = self.game.step(action)
                            if over:
                                return True, result
                continue
            
            # Handle drawing phase
            if self.game.phase == RiichiPhase.DRAWING:
                if self.game.current_player != self.player_idx:
                    opp_idx = self.game.current_player
                    opp_actions = self.game.get_valid_actions(opp_idx)
                    if opp_actions:
                        action = self._get_opponent_action(opp_idx, opp_actions)
                        over, result = self.game.step(action)
                        if over:
                            return True, result
                        continue
                else:
                    return False, None
            
            # Handle discarding phase
            if self.game.phase == RiichiPhase.DISCARDING:
                if self.game.current_player != self.player_idx:
                    opp_idx = self.game.current_player
                    opp_actions = self.game.get_valid_actions(opp_idx)
                    if opp_actions:
                        action = self._get_opponent_action(opp_idx, opp_actions)
                        over, result = self.game.step(action)
                        if over:
                            return True, result
                        continue
                else:
                    return False, None
            
            # After kan phase
            if self.game.phase == RiichiPhase.AFTER_KAN:
                if self.game.current_player != self.player_idx:
                    opp_idx = self.game.current_player
                    opp_actions = self.game.get_valid_actions(opp_idx)
                    if opp_actions:
                        action = self._get_opponent_action(opp_idx, opp_actions)
                        over, result = self.game.step(action)
                        if over:
                            return True, result
                        continue
                else:
                    return False, None
            
            break
        
        return False, None
    
    def _get_opponent_action(self, player_idx: int, valid_actions: List[RiichiAction]) -> RiichiAction:
        """Get action for opponent."""
        if self.opponent_policy == "random":
            return self._random_policy(valid_actions)
        elif self.opponent_policy == "greedy":
            return self._greedy_policy(player_idx, valid_actions)
        return self._random_policy(valid_actions)
    
    def _random_policy(self, valid_actions: List[RiichiAction]) -> RiichiAction:
        """Random action selection."""
        meaningful = [a for a in valid_actions if a.action_type != RiichiActionType.PASS]
        if meaningful:
            import random
            return random.choice(meaningful)
        import random
        return random.choice(valid_actions)
    
    def _greedy_policy(self, player_idx: int, valid_actions: List[RiichiAction]) -> RiichiAction:
        """Greedy action selection."""
        priority = {
            RiichiActionType.TSUMO: 0,
            RiichiActionType.RON: 0,
            RiichiActionType.RIICHI: 1,
            RiichiActionType.KAN: 2,
            RiichiActionType.ANKAN: 2,
            RiichiActionType.PON: 3,
            RiichiActionType.CHII: 4,
            RiichiActionType.DRAW: 5,
            RiichiActionType.DISCARD: 6,
            RiichiActionType.SHOUMINKAN: 7,
            RiichiActionType.PASS: 8,
        }
        
        sorted_actions = sorted(valid_actions, key=lambda a: priority.get(a.action_type, 10))
        
        if sorted_actions[0].action_type == RiichiActionType.DISCARD:
            discards = [a for a in valid_actions if a.action_type == RiichiActionType.DISCARD]
            import random
            return random.choice(discards)
        
        return sorted_actions[0]
    
    def _calculate_shaping_reward(self, action: RiichiAction) -> float:
        """Calculate reward shaping."""
        reward = 0.0
        
        if action.action_type == RiichiActionType.RIICHI:
            reward += 0.1
        elif action.action_type == RiichiActionType.PON:
            reward += 0.03
        elif action.action_type == RiichiActionType.KAN:
            reward += 0.05
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        return {
            "turn": self.game.turn_count,
            "phase": self.game.phase.name,
            "current_player": self.game.current_player,
            "wall_remaining": self.game.wall.remaining,
            "honba": self.game.honba,
            "riichi_sticks": self.game.riichi_sticks,
        }
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "human":
            print(self._render_ansi())
        elif self.render_mode == "ansi":
            return self._render_ansi()
        return None
    
    def _render_ansi(self) -> str:
        """Render as ASCII string."""
        lines = []
        lines.append(f"=== Riichi Mahjong ({self.rules.name}) - Turn {self.game.turn_count} ===")
        lines.append(f"Phase: {self.game.phase.name}")
        lines.append(f"Current: P{self.game.current_player} | Dealer: P{self.game.dealer}")
        lines.append(f"Wall: {self.game.wall.remaining} | Honba: {self.game.honba} | Riichi: {self.game.riichi_sticks}")
        
        # Dora
        dora_str = " ".join(str(d) for d in self.game.dora_indicators)
        lines.append(f"Dora indicators: {dora_str}")
        
        if self.game.last_discard:
            lines.append(f"Last discard: {self.game.last_discard} (P{self.game.last_discard_player})")
        
        lines.append("")
        lines.append(f"--- Your Hand (P{self.player_idx}) ---")
        player = self.game.players[self.player_idx]
        lines.append(f"Hand: {player.hand}")
        lines.append(f"Score: {player.score}")
        
        status = []
        if player.is_riichi:
            status.append("RIICHI")
        if player.furiten.is_furiten:
            status.append("FURITEN")
        if status:
            lines.append(f"Status: {', '.join(status)}")
        
        if player.melds:
            melds_str = " | ".join(str(m) for m in player.melds)
            lines.append(f"Melds: {melds_str}")
        
        lines.append("")
        lines.append("--- Valid Actions ---")
        valid = self.game.get_valid_actions(self.player_idx)
        for a in valid[:10]:
            lines.append(f"  {a}")
        if len(valid) > 10:
            lines.append(f"  ... and {len(valid) - 10} more")
        
        return "\n".join(lines)
    
    def close(self):
        """Clean up."""
        pass


def register_riichi_envs():
    """Register Riichi Mahjong environments."""
    gym.register(
        id="RiichiMahjong-v0",
        entry_point="envs.riichi_env:RiichiMahjongEnv",
        max_episode_steps=1000,
    )
    
    gym.register(
        id="RiichiMahjong-Tenhou-v0",
        entry_point="envs.riichi_env:RiichiMahjongEnv",
        kwargs={"rules": "tenhou"},
        max_episode_steps=1000,
    )
    
    gym.register(
        id="RiichiMahjong-EMA-v0",
        entry_point="envs.riichi_env:RiichiMahjongEnv",
        kwargs={"rules": "ema"},
        max_episode_steps=1000,
    )


if __name__ == "__main__":
    # Quick test
    env = RiichiMahjongEnv(render_mode="human", rules="tenhou")
    obs, info = env.reset()
    
    print("Observation keys:", obs.keys())
    print("Hand shape:", obs["hand"].shape)
    print("Valid actions:", np.sum(obs["valid_actions"]), "available")
    
    env.render()
    
    for _ in range(5):
        valid_mask = obs["valid_actions"]
        valid_indices = np.where(valid_mask == 1)[0]
        if len(valid_indices) > 0:
            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}")
            if terminated or truncated:
                print("Game ended!")
                break
        else:
            print("No valid actions!")
            break
    
    env.close()

