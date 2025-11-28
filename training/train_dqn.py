#!/usr/bin/env python3
"""
DQN Training Script for MCR Mahjong

Train a DQN agent to play MCR Mahjong against random/greedy opponents.
Supports Weights & Biases logging.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from envs.mcr_env import MCRMahjongEnv
from agents.sb3_agent import SB3Agent, create_mahjong_dqn, MahjongFeaturesExtractor


class WandbCallback(BaseCallback):
    """Custom callback for logging metrics to Weights & Biases."""
    
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.total_score = 0
        
    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_info = info["episode"]
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])
                    
                    result = ep_info.get("result")
                    if result:
                        if hasattr(result, 'winner'):
                            if result.winner == 0:
                                self.win_count += 1
                                if hasattr(result, 'score'):
                                    self.total_score += result.score
                            elif result.winner is not None:
                                self.loss_count += 1
                            else:
                                self.draw_count += 1
        
        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            import wandb
            
            total_games = self.win_count + self.loss_count + self.draw_count
            
            metrics = {
                "rollout/ep_rew_mean": np.mean(self.episode_rewards[-100:]),
                "rollout/ep_len_mean": np.mean(self.episode_lengths[-100:]),
                "game/total_episodes": len(self.episode_rewards),
                "game/win_count": self.win_count,
                "game/loss_count": self.loss_count,
            }
            
            if total_games > 0:
                metrics["game/win_rate"] = self.win_count / total_games
            
            if self.win_count > 0:
                metrics["game/avg_win_score"] = self.total_score / self.win_count
            
            wandb.log(metrics, step=self.num_timesteps)
        
        return True
    
    def _on_training_end(self) -> None:
        if self.episode_rewards:
            import wandb
            total_games = self.win_count + self.loss_count + self.draw_count
            wandb.run.summary["final/total_episodes"] = len(self.episode_rewards)
            wandb.run.summary["final/mean_reward"] = np.mean(self.episode_rewards)
            if total_games > 0:
                wandb.run.summary["final/win_rate"] = self.win_count / total_games


def make_env(
    player_idx: int = 0,
    opponent_policy: str = "random",
    seed: int = None,
    reward_shaping: bool = True,
    rank: int = 0,
):
    """Create a wrapped environment for training."""
    def _init():
        env_seed = seed + rank if seed is not None else None
        env = MCRMahjongEnv(
            player_idx=player_idx,
            opponent_policy=opponent_policy,
            seed=env_seed,
            reward_shaping=reward_shaping,
        )
        env = Monitor(env)
        return env
    return _init


def train_dqn(
    total_timesteps: int = 500_000,
    learning_rate: float = 1e-4,
    buffer_size: int = 100000,
    learning_starts: int = 10000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.2,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    opponent_policy: str = "random",
    reward_shaping: bool = True,
    save_dir: str = "models/dqn",
    eval_freq: int = 10000,
    seed: int = 42,
    device: str = "auto",
    verbose: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "mcr-mahjong-rl",
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    """Train a DQN agent for MCR Mahjong."""
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = wandb_run_name or f"dqn_{timestamp}"
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        import wandb
        
        config = {
            "algorithm": "DQN",
            "game": "MCR Mahjong",
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "target_update_interval": target_update_interval,
            "exploration_fraction": exploration_fraction,
            "opponent_policy": opponent_policy,
            "reward_shaping": reward_shaping,
            "seed": seed,
            "device": device,
        }
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config,
            sync_tensorboard=True,
        )
    
    print("=" * 60)
    print("MCR Mahjong DQN Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Buffer size: {buffer_size:,}")
    print(f"Opponent policy: {opponent_policy}")
    print(f"Device: {device}")
    print(f"WandB logging: {'enabled' if use_wandb else 'disabled'}")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create training environment
    env = DummyVecEnv([
        make_env(
            player_idx=0,
            opponent_policy=opponent_policy,
            seed=seed,
            reward_shaping=reward_shaping,
            rank=0,
        )
    ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(
            player_idx=0,
            opponent_policy=opponent_policy,
            seed=seed + 1000,
            reward_shaping=False,
            rank=0,
        )
    ])
    
    # Create agent
    agent = create_mahjong_dqn(
        env=env,
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
        tensorboard_log=str(save_path / "tensorboard") if use_wandb else None,
    )
    
    # Create callbacks
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(save_path / "checkpoints"),
        name_prefix="dqn_mahjong",
    )
    callbacks.append(checkpoint_callback)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    if use_wandb:
        wandb_callback = WandbCallback(verbose=verbose, log_freq=1000)
        callbacks.append(wandb_callback)
    
    # Train
    print("\nStarting training...")
    agent.model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = save_path / "final_model"
    agent.save(str(final_path))
    print(f"\nTraining complete! Model saved to {final_path}")
    
    if use_wandb:
        import wandb
        wandb.save(str(final_path) + ".zip")
        wandb.finish()
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return agent, str(save_path)


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for MCR Mahjong")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=500_000,
                       help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100000,
                       help="Replay buffer size")
    parser.add_argument("--learning-starts", type=int, default=10000,
                       help="Steps before learning starts")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Minibatch size")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--target-update", type=int, default=1000,
                       help="Target network update interval")
    parser.add_argument("--exploration-fraction", type=float, default=0.2,
                       help="Fraction of training for epsilon decay")
    parser.add_argument("--eps-start", type=float, default=1.0,
                       help="Initial exploration epsilon")
    parser.add_argument("--eps-end", type=float, default=0.05,
                       help="Final exploration epsilon")
    
    # Environment parameters
    parser.add_argument("--opponent", type=str, default="random",
                       choices=["random", "greedy"],
                       help="Opponent policy")
    parser.add_argument("--no-reward-shaping", action="store_true",
                       help="Disable reward shaping")
    
    # Saving parameters
    parser.add_argument("--save-dir", type=str, default="models/dqn",
                       help="Directory to save models")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    # WandB parameters
    parser.add_argument("--wandb", action="store_true",
                       help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="mcr-mahjong-rl",
                       help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="WandB entity/team name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                       help="WandB run name")
    
    args = parser.parse_args()
    
    train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        opponent_policy=args.opponent,
        reward_shaping=not args.no_reward_shaping,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
