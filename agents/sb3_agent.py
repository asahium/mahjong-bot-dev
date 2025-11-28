"""
Stable Baselines3 Agent Wrapper for MCR Mahjong

Provides custom feature extractors and wrappers for SB3 algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Type, Optional, List

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor


class MahjongFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for MCR Mahjong observations.
    
    Processes the dictionary observation into a flat feature vector
    suitable for the policy network.
    """
    
    def __init__(
        self, 
        observation_space: spaces.Dict,
        features_dim: int = 512,
        hand_embedding_dim: int = 128,
        meld_embedding_dim: int = 64,
        discard_embedding_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)
        
        self.hand_embedding_dim = hand_embedding_dim
        self.meld_embedding_dim = meld_embedding_dim
        self.discard_embedding_dim = discard_embedding_dim
        
        # Hand encoder (34 tile counts -> embedding)
        self.hand_encoder = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(),
            nn.Linear(128, hand_embedding_dim),
            nn.ReLU(),
        )
        
        # Meld encoder (4 players x 4 melds x 34 tiles -> embedding per player)
        self.meld_encoder = nn.Sequential(
            nn.Linear(4 * 34, 128),
            nn.ReLU(),
            nn.Linear(128, meld_embedding_dim),
            nn.ReLU(),
        )
        
        # Discard encoder (34 tile counts per player -> embedding)
        self.discard_encoder = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, discard_embedding_dim),
            nn.ReLU(),
        )
        
        # Last discard encoder
        self.last_discard_encoder = nn.Sequential(
            nn.Linear(34, 32),
            nn.ReLU(),
        )
        
        # Game info encoder
        self.game_info_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
        )
        
        combined_dim = (
            hand_embedding_dim + 
            4 * meld_embedding_dim + 
            4 * discard_embedding_dim + 
            32 + 32
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = observations["hand"].shape[0]
        
        hand = observations["hand"].float()
        hand_features = self.hand_encoder(hand)
        
        melds = observations["melds"].float()
        meld_features_list = []
        for p in range(4):
            player_melds = melds[:, p].reshape(batch_size, -1)
            meld_feat = self.meld_encoder(player_melds)
            meld_features_list.append(meld_feat)
        meld_features = torch.cat(meld_features_list, dim=-1)
        
        discards = observations["discards"].float()
        discard_features_list = []
        for p in range(4):
            discard_feat = self.discard_encoder(discards[:, p])
            discard_features_list.append(discard_feat)
        discard_features = torch.cat(discard_features_list, dim=-1)
        
        last_discard = observations["last_discard"].float()
        last_discard_features = self.last_discard_encoder(last_discard)
        
        game_info = observations["game_info"].float()
        game_info_features = self.game_info_encoder(game_info)
        
        combined = torch.cat([
            hand_features,
            meld_features,
            discard_features,
            last_discard_features,
            game_info_features,
        ], dim=-1)
        
        features = self.combiner(combined)
        return features


class RiichiFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Riichi Mahjong observations.
    
    Handles Riichi-specific observation space including:
    - dora_indicators
    - riichi_status
    - furiten
    - scores
    """
    
    def __init__(
        self, 
        observation_space: spaces.Dict,
        features_dim: int = 512,
    ):
        super().__init__(observation_space, features_dim)
        
        # Hand encoder (34 tile counts)
        self.hand_encoder = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Meld encoder (4 players x 4 melds x 34 tiles)
        self.meld_encoder = nn.Sequential(
            nn.Linear(4 * 34, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Discard encoder (34 counts per player)
        self.discard_encoder = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Dora encoder (5 x 34 one-hot indicators)
        self.dora_encoder = nn.Sequential(
            nn.Linear(5 * 34, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Status encoder (riichi_status + furiten + scores)
        # riichi: 4, furiten: 1, scores: 4 = 9
        self.status_encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
        )
        
        # Game info encoder (12 values for Riichi)
        self.game_info_encoder = nn.Sequential(
            nn.Linear(12, 48),
            nn.ReLU(),
        )
        
        # Combined dimension:
        # hand(128) + 4*meld(64) + 4*discard(64) + dora(32) + status(32) + game_info(48)
        combined_dim = 128 + 4 * 64 + 4 * 64 + 32 + 32 + 48  # = 752
        
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, 384),
            nn.ReLU(),
            nn.Linear(384, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = observations["hand"].shape[0]
        
        # Hand
        hand = observations["hand"].float()
        hand_features = self.hand_encoder(hand)
        
        # Melds
        melds = observations["melds"].float()
        meld_features_list = []
        for p in range(4):
            player_melds = melds[:, p].reshape(batch_size, -1)
            meld_feat = self.meld_encoder(player_melds)
            meld_features_list.append(meld_feat)
        meld_features = torch.cat(meld_features_list, dim=-1)
        
        # Discards
        discards = observations["discards"].float()
        discard_features_list = []
        for p in range(4):
            discard_feat = self.discard_encoder(discards[:, p])
            discard_features_list.append(discard_feat)
        discard_features = torch.cat(discard_features_list, dim=-1)
        
        # Dora indicators (5 x 34 -> flatten)
        dora = observations["dora_indicators"].float().reshape(batch_size, -1)
        dora_features = self.dora_encoder(dora)
        
        # Status: riichi_status (4), furiten (1), scores (4) normalized
        riichi_status = observations["riichi_status"].float()
        furiten = observations["furiten"].float()
        scores = observations["scores"].float() / 100000.0  # Normalize scores
        status = torch.cat([riichi_status, furiten, scores], dim=-1)
        status_features = self.status_encoder(status)
        
        # Game info
        game_info = observations["game_info"].float()
        game_info_features = self.game_info_encoder(game_info)
        
        # Combine all
        combined = torch.cat([
            hand_features,
            meld_features,
            discard_features,
            dora_features,
            status_features,
            game_info_features,
        ], dim=-1)
        
        features = self.combiner(combined)
        return features


class ActionMaskCallback(BaseCallback):
    """
    Callback that logs action mask statistics during training.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.valid_action_counts = []
    
    def _on_step(self) -> bool:
        # Log average number of valid actions
        if hasattr(self.training_env, 'get_attr'):
            try:
                obs = self.training_env.get_attr('last_obs')
                if obs and 'valid_actions' in obs[0]:
                    valid_count = np.sum(obs[0]['valid_actions'])
                    self.valid_action_counts.append(valid_count)
            except Exception:
                pass
        return True
    
    def _on_training_end(self) -> None:
        if self.valid_action_counts:
            avg_valid = np.mean(self.valid_action_counts)
            if self.verbose > 0:
                print(f"Average valid actions: {avg_valid:.2f}")


class SB3Agent:
    """
    Wrapper class for Stable Baselines3 agents in MCR Mahjong.
    
    Provides convenient methods for training and evaluation.
    """
    
    def __init__(
        self,
        env: gym.Env,
        algorithm: str = "PPO",
        policy: str = "MultiInputPolicy",
        features_extractor_class: Type[BaseFeaturesExtractor] = MahjongFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        learning_rate: float = 3e-4,
        device: str = "auto",
        verbose: int = 1,
        **kwargs
    ):
        """
        Initialize the SB3 agent.
        
        Args:
            env: Gymnasium environment
            algorithm: Algorithm name ("PPO", "DQN", "A2C")
            policy: Policy type
            features_extractor_class: Custom feature extractor class
            features_extractor_kwargs: Arguments for feature extractor
            learning_rate: Learning rate
            device: Device to use ("auto", "cpu", "cuda")
            verbose: Verbosity level
            **kwargs: Additional arguments for the algorithm
        """
        self.env = env
        self.algorithm_name = algorithm
        
        # Default feature extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {"features_dim": 512}
        
        # Policy kwargs
        policy_kwargs = {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": features_extractor_kwargs,
        }
        
        # Merge with any additional policy kwargs
        if "policy_kwargs" in kwargs:
            policy_kwargs.update(kwargs.pop("policy_kwargs"))
        
        # Create the algorithm
        algorithm_class = self._get_algorithm_class(algorithm)
        
        self.model = algorithm_class(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
            **kwargs
        )
    
    def _get_algorithm_class(self, name: str):
        """Get algorithm class from name."""
        algorithms = {
            "PPO": PPO,
            "DQN": DQN,
            "A2C": A2C,
        }
        if name not in algorithms:
            raise ValueError(f"Unknown algorithm: {name}. Choose from {list(algorithms.keys())}")
        return algorithms[name]
    
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        save_path: Optional[str] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation (if different from training)
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of episodes per evaluation
            save_path: Path to save the best model
            callbacks: Additional callbacks
        """
        callback_list = callbacks or []
        
        # Add evaluation callback if eval_env provided
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callback_list.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list if callback_list else None,
        )
        
        # Save final model
        if save_path:
            self.model.save(f"{save_path}/final_model")
    
    def predict(
        self, 
        observation: Dict[str, np.ndarray], 
        deterministic: bool = True
    ):
        """
        Predict action from observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, state)
        """
        # Apply action masking
        action, state = self.model.predict(observation, deterministic=deterministic)
        
        # Verify action is valid
        valid_actions = observation.get("valid_actions", None)
        if valid_actions is not None and valid_actions[action] == 0:
            # Invalid action, choose random valid one
            valid_indices = np.where(valid_actions == 1)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
        
        return action, state
    
    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Get action from observation (simple interface).
        
        Args:
            observation: Environment observation
            
        Returns:
            Action index
        """
        action, _ = self.predict(observation)
        return action
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a saved model."""
        algorithm_class = self._get_algorithm_class(self.algorithm_name)
        self.model = algorithm_class.load(path, env=self.env)
    
    @classmethod
    def from_pretrained(cls, path: str, env: gym.Env, algorithm: str = "PPO"):
        """
        Load a pretrained agent.
        
        Args:
            path: Path to saved model
            env: Environment
            algorithm: Algorithm name
            
        Returns:
            SB3Agent instance
        """
        agent = cls.__new__(cls)
        agent.env = env
        agent.algorithm_name = algorithm
        agent.load(path)
        return agent


def create_mahjong_ppo(
    env: gym.Env,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    verbose: int = 1,
    **kwargs
) -> SB3Agent:
    """
    Create a PPO agent configured for MCR Mahjong.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Batch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clipping range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        verbose: Verbosity
        
    Returns:
        Configured SB3Agent
    """
    return SB3Agent(
        env=env,
        algorithm="PPO",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        **kwargs
    )


def create_mahjong_dqn(
    env: gym.Env,
    learning_rate: float = 1e-4,
    buffer_size: int = 100000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    verbose: int = 1,
    **kwargs
) -> SB3Agent:
    """
    Create a DQN agent configured for MCR Mahjong.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        learning_starts: Steps before learning starts
        batch_size: Batch size
        tau: Soft update coefficient
        gamma: Discount factor
        target_update_interval: Target network update interval
        exploration_fraction: Fraction of training for exploration decay
        exploration_initial_eps: Initial exploration epsilon
        exploration_final_eps: Final exploration epsilon
        verbose: Verbosity
        
    Returns:
        Configured SB3Agent
    """
    return SB3Agent(
        env=env,
        algorithm="DQN",
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=verbose,
        **kwargs
    )

