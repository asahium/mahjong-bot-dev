"""
Random Agent for MCR Mahjong

A simple baseline agent that takes random valid actions.
"""

import numpy as np
from typing import Dict, Any


class RandomAgent:
    """
    Random agent that selects uniformly from valid actions.
    
    This serves as a baseline for comparison with trained agents.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the random agent.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Select an action given the current observation.
        
        Args:
            observation: Dictionary observation from the environment
            
        Returns:
            Action index
        """
        valid_actions = observation["valid_actions"]
        valid_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_indices) == 0:
            # No valid actions, return PASS
            return 205  # ACTION_PASS
        
        return self.rng.choice(valid_indices)
    
    def predict(self, observation: Dict[str, np.ndarray], deterministic: bool = True):
        """
        Predict action (compatible with SB3 interface).
        
        Args:
            observation: Dictionary observation
            deterministic: Whether to be deterministic (ignored for random agent)
            
        Returns:
            Tuple of (action, state)
        """
        action = self.act(observation)
        return action, None
    
    def reset(self):
        """Reset the agent state (no-op for random agent)."""
        pass
    
    def __repr__(self) -> str:
        return "RandomAgent()"


class GreedyAgent:
    """
    Greedy agent that prioritizes claiming and winning.
    
    Simple heuristic-based agent for slightly better baseline.
    """
    
    # Action type ranges
    ACTION_DISCARD_START = 0
    ACTION_CHOW_START = 34
    ACTION_PONG_START = 68
    ACTION_KONG_START = 102
    ACTION_CONCEALED_KONG_START = 136
    ACTION_ADD_KONG_START = 170
    ACTION_HU = 204
    ACTION_PASS = 205
    ACTION_DRAW = 206
    
    def __init__(self, seed: int = None):
        """Initialize the greedy agent."""
        self.rng = np.random.default_rng(seed)
    
    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Select action using greedy heuristics.
        
        Priority: Hu > Kong > Pong > Chow > Draw > Discard (smart) > Pass
        """
        valid_actions = observation["valid_actions"]
        valid_indices = np.where(valid_actions == 1)[0]
        
        if len(valid_indices) == 0:
            return self.ACTION_PASS
        
        # Check for Hu (always win if possible)
        if valid_actions[self.ACTION_HU] == 1:
            return self.ACTION_HU
        
        # Check for Kong
        kong_actions = valid_indices[
            (valid_indices >= self.ACTION_KONG_START) & 
            (valid_indices < self.ACTION_CONCEALED_KONG_START)
        ]
        if len(kong_actions) > 0:
            return self.rng.choice(kong_actions)
        
        # Check for concealed Kong
        concealed_kong = valid_indices[
            (valid_indices >= self.ACTION_CONCEALED_KONG_START) & 
            (valid_indices < self.ACTION_ADD_KONG_START)
        ]
        if len(concealed_kong) > 0:
            return self.rng.choice(concealed_kong)
        
        # Check for Pong
        pong_actions = valid_indices[
            (valid_indices >= self.ACTION_PONG_START) & 
            (valid_indices < self.ACTION_KONG_START)
        ]
        if len(pong_actions) > 0:
            return self.rng.choice(pong_actions)
        
        # Check for Chow (less priority)
        chow_actions = valid_indices[
            (valid_indices >= self.ACTION_CHOW_START) & 
            (valid_indices < self.ACTION_PONG_START)
        ]
        if len(chow_actions) > 0:
            # 50% chance to take chow
            if self.rng.random() < 0.5:
                return self.rng.choice(chow_actions)
        
        # Check for Draw
        if valid_actions[self.ACTION_DRAW] == 1:
            return self.ACTION_DRAW
        
        # Discard (prefer isolated tiles)
        discard_actions = valid_indices[
            (valid_indices >= self.ACTION_DISCARD_START) & 
            (valid_indices < self.ACTION_CHOW_START)
        ]
        if len(discard_actions) > 0:
            return self._smart_discard(observation, discard_actions)
        
        # Pass
        if valid_actions[self.ACTION_PASS] == 1:
            return self.ACTION_PASS
        
        return self.rng.choice(valid_indices)
    
    def _smart_discard(
        self, 
        observation: Dict[str, np.ndarray], 
        discard_actions: np.ndarray
    ) -> int:
        """
        Choose which tile to discard using simple heuristics.
        
        Prefer discarding:
        1. Isolated honor tiles
        2. Isolated terminal tiles
        3. Tiles that don't contribute to sequences
        """
        hand = observation["hand"]
        
        # Score each possible discard (lower is better to discard)
        scores = []
        for action in discard_actions:
            tile_idx = action - self.ACTION_DISCARD_START
            score = self._tile_value(hand, tile_idx)
            scores.append((action, score))
        
        # Sort by score (ascending) and pick from worst tiles
        scores.sort(key=lambda x: x[1])
        worst_tiles = [s[0] for s in scores[:3]]  # Top 3 worst
        
        return self.rng.choice(worst_tiles)
    
    def _tile_value(self, hand: np.ndarray, tile_idx: int) -> float:
        """
        Calculate value of keeping a tile.
        Higher value = better to keep.
        """
        count = hand[tile_idx]
        value = count * 2.0  # Base value from count
        
        # Honor tiles (indices 27-33)
        if tile_idx >= 27:
            if count >= 2:
                value += 3.0  # Good for pong
            else:
                value -= 1.0  # Isolated honor
            return value
        
        # Numbered tiles - check for sequence potential
        suit = tile_idx // 9
        num = tile_idx % 9  # 0-8 for values 1-9
        
        # Terminal tiles (1 or 9)
        if num == 0 or num == 8:
            value -= 0.5
        
        # Check for neighbors
        suit_start = suit * 9
        
        # Left neighbor
        if num > 0 and hand[suit_start + num - 1] > 0:
            value += 1.5
        
        # Right neighbor
        if num < 8 and hand[suit_start + num + 1] > 0:
            value += 1.5
        
        # Two-gap neighbors (for potential sequences)
        if num > 1 and hand[suit_start + num - 2] > 0:
            value += 0.5
        
        if num < 7 and hand[suit_start + num + 2] > 0:
            value += 0.5
        
        return value
    
    def predict(self, observation: Dict[str, np.ndarray], deterministic: bool = True):
        """Predict action (SB3 interface compatible)."""
        action = self.act(observation)
        return action, None
    
    def reset(self):
        """Reset agent state."""
        pass
    
    def __repr__(self) -> str:
        return "GreedyAgent()"

