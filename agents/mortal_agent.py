"""
Mortal AI Interface for Riichi Mahjong

Interface to use Mortal AI (https://github.com/Equim-chan/Mortal) as an opponent.
Mortal is a strong Riichi Mahjong AI trained on millions of Tenhou games.

Requirements:
- onnxruntime (pip install onnxruntime)
- Mortal model file (mortal.onnx)

If Mortal is not available, falls back to heuristic agent.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from riichi_mahjong.game import RiichiGame, RiichiAction, RiichiActionType

# Try to import ONNX runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class MortalAgent:
    """
    Wrapper for Mortal AI model.
    
    Mortal uses a neural network trained on Tenhou game logs.
    It's one of the strongest open-source Riichi Mahjong AIs.
    
    If the model is not available, falls back to heuristic agent.
    """
    
    # Default model path
    DEFAULT_MODEL_PATH = "models/mortal/mortal.onnx"
    
    # Mortal's action encoding (simplified mapping)
    # Mortal uses a different action space, this is approximate
    MORTAL_ACTION_SIZE = 181
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize Mortal agent.
        
        Args:
            model_path: Path to mortal.onnx model file
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.device = device
        self.session = None
        self.available = False
        
        # Try to load model
        self._load_model()
        
        # Fallback agent if Mortal not available
        if not self.available:
            from agents.heuristic_agent import HeuristicAgent
            self.fallback_agent = HeuristicAgent()
        else:
            self.fallback_agent = None
    
    def _load_model(self) -> bool:
        """Load the ONNX model."""
        if not ONNX_AVAILABLE:
            print("Warning: onnxruntime not installed. Using fallback agent.")
            return False
        
        model_file = Path(self.model_path)
        if not model_file.exists():
            print(f"Warning: Mortal model not found at {self.model_path}. Using fallback agent.")
            print("Download Mortal model from: https://github.com/Equim-chan/Mortal")
            return False
        
        try:
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Select execution provider
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                str(model_file),
                sess_options,
                providers=providers
            )
            
            self.available = True
            print(f"Mortal model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load Mortal model: {e}")
            return False
    
    def get_action(
        self, 
        game: RiichiGame, 
        player_idx: int,
        valid_actions: List[RiichiAction]
    ) -> RiichiAction:
        """
        Get action from Mortal AI.
        
        Args:
            game: Current game state
            player_idx: Index of this player
            valid_actions: List of valid actions
            
        Returns:
            Selected action
        """
        if not self.available or self.session is None:
            # Use fallback
            return self.fallback_agent.get_action(game, player_idx, valid_actions)
        
        try:
            # Convert game state to Mortal input format
            obs = self._create_mortal_observation(game, player_idx)
            
            # Run inference
            inputs = {self.session.get_inputs()[0].name: obs}
            outputs = self.session.run(None, inputs)
            
            # Get action probabilities
            action_probs = outputs[0][0]  # Shape: (181,)
            
            # Convert Mortal action to our action format
            action = self._convert_mortal_action(
                action_probs, game, player_idx, valid_actions
            )
            
            return action
            
        except Exception as e:
            print(f"Warning: Mortal inference failed: {e}")
            return self.fallback_agent.get_action(game, player_idx, valid_actions)
    
    def _create_mortal_observation(
        self, 
        game: RiichiGame, 
        player_idx: int
    ) -> np.ndarray:
        """
        Create observation in Mortal's input format.
        
        Mortal uses a specific encoding of game state.
        This is a simplified approximation.
        """
        player = game.players[player_idx]
        
        # Mortal input shape varies by version
        # Using simplified encoding here
        obs = np.zeros((1, 93, 34), dtype=np.float32)
        
        # Encode hand (one-hot style)
        hand_counts = player.get_hand_count_array()
        for i in range(34):
            for j in range(min(hand_counts[i], 4)):
                obs[0, j, i] = 1.0
        
        # Encode discards for each player
        for p_idx, p in enumerate(game.players):
            offset = 4 + p_idx * 21  # Each player gets 21 rows for discards
            for d_idx, tile in enumerate(p.discards[:21]):
                obs[0, offset + d_idx, tile.tile_index] = 1.0
        
        # Encode dora indicators
        dora_offset = 88
        for i, dora in enumerate(game.dora_indicators[:5]):
            obs[0, dora_offset + i, dora.tile_index] = 1.0
        
        return obs
    
    def _convert_mortal_action(
        self, 
        action_probs: np.ndarray,
        game: RiichiGame,
        player_idx: int,
        valid_actions: List[RiichiAction]
    ) -> RiichiAction:
        """
        Convert Mortal's action output to our action format.
        
        Mortal action space (simplified):
        - 0-33: Discard tiles
        - 34-67: Riichi + discard
        - 68: Tsumo
        - 69: Ron  
        - 70-103: Chi
        - 104-137: Pon
        - 138-171: Kan
        - 172+: Pass, etc.
        """
        # Create mapping from our actions to indices
        action_scores = {}
        
        for action in valid_actions:
            # Map each valid action to Mortal's action space
            mortal_idx = self._action_to_mortal_idx(action)
            if mortal_idx is not None and mortal_idx < len(action_probs):
                action_scores[action] = action_probs[mortal_idx]
            else:
                action_scores[action] = -100  # Invalid
        
        # Select highest scoring valid action
        best_action = max(action_scores.keys(), key=lambda a: action_scores[a])
        
        return best_action
    
    def _action_to_mortal_idx(self, action: RiichiAction) -> Optional[int]:
        """Map our action to Mortal's action index."""
        if action.action_type == RiichiActionType.DISCARD:
            if action.tile:
                return action.tile.tile_index  # 0-33
        elif action.action_type == RiichiActionType.RIICHI:
            if action.tile:
                return 34 + action.tile.tile_index  # 34-67
        elif action.action_type == RiichiActionType.TSUMO:
            return 68
        elif action.action_type == RiichiActionType.RON:
            return 69
        elif action.action_type == RiichiActionType.CHII:
            if action.meld_tiles:
                min_tile = min(action.meld_tiles, key=lambda t: t.tile_index)
                return 70 + min_tile.tile_index
        elif action.action_type == RiichiActionType.PON:
            if action.tile:
                return 104 + action.tile.tile_index
        elif action.action_type == RiichiActionType.KAN:
            if action.tile:
                return 138 + action.tile.tile_index
        elif action.action_type == RiichiActionType.PASS:
            return 172
        elif action.action_type == RiichiActionType.DRAW:
            return 173
        
        return None


class MortalDownloader:
    """Helper to download Mortal model."""
    
    MORTAL_RELEASES_URL = "https://api.github.com/repos/Equim-chan/Mortal/releases/latest"
    
    @staticmethod
    def download_model(output_path: str = "models/mortal/mortal.onnx") -> bool:
        """
        Download Mortal model from GitHub releases.
        
        Note: Mortal model may need to be obtained separately
        due to licensing or size constraints.
        """
        try:
            import urllib.request
            import json
            
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print("Note: Mortal model download requires manual steps.")
            print("Please visit https://github.com/Equim-chan/Mortal")
            print("and follow instructions to obtain the model file.")
            print(f"Place the model at: {output_path}")
            
            return False
            
        except Exception as e:
            print(f"Download failed: {e}")
            return False


def create_mortal_policy(model_path: Optional[str] = None):
    """
    Create a Mortal policy function for use in environments.
    
    Returns a function that takes (game, player_idx, valid_actions) and returns an action.
    """
    agent = MortalAgent(model_path=model_path)
    
    def policy(game: RiichiGame, player_idx: int, valid_actions: List[RiichiAction]) -> RiichiAction:
        return agent.get_action(game, player_idx, valid_actions)
    
    return policy


if __name__ == "__main__":
    print("Mortal AI Interface")
    print("=" * 50)
    
    agent = MortalAgent()
    
    if agent.available:
        print("Mortal model loaded successfully!")
    else:
        print("Mortal not available, using heuristic fallback")
    
    # Quick test
    from riichi_mahjong.game import RiichiGame
    from riichi_mahjong.rules import TENHOU_RULES
    
    game = RiichiGame(rules=TENHOU_RULES, seed=42)
    game.reset()
    game.start_round()
    
    print("\nTesting agent...")
    for turn in range(10):
        current = game.current_player
        valid_actions = game.get_valid_actions(current)
        
        if valid_actions:
            action = agent.get_action(game, current, valid_actions)
            print(f"Turn {turn}: P{current} -> {action.action_type.name}")
            
            over, result = game.step(action)
            if over:
                print(f"Game over! {result}")
                break
    
    print("\nTest completed!")

