#!/usr/bin/env python3
"""
PPO Training Script for Riichi Mahjong

Train a PPO agent to play Riichi Mahjong (EMA or Tenhou rules).
Supports Weights & Biases logging.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from envs.riichi_env import RiichiMahjongEnv
from agents.sb3_agent import SB3Agent, MahjongFeaturesExtractor, RiichiFeaturesExtractor


class WandbCallback(BaseCallback):
    """
    Custom callback for logging metrics to Weights & Biases.
    """
    
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        
    def _on_step(self) -> bool:
        # Log episode statistics when available
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_info = info["episode"]
                    self.episode_rewards.append(ep_info["r"])
                    self.episode_lengths.append(ep_info["l"])
                    
                    # Track game outcomes
                    result = ep_info.get("result")
                    if result:
                        if hasattr(result, 'winner'):
                            if result.winner == 0:  # Agent won
                                self.win_count += 1
                            elif result.winner is not None:
                                self.loss_count += 1
                            else:
                                self.draw_count += 1
        
        # Log to wandb periodically
        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            import wandb
            
            total_games = self.win_count + self.loss_count + self.draw_count
            
            metrics = {
                "rollout/ep_rew_mean": np.mean(self.episode_rewards[-100:]),
                "rollout/ep_len_mean": np.mean(self.episode_lengths[-100:]),
                "rollout/ep_rew_std": np.std(self.episode_rewards[-100:]),
                "game/total_episodes": len(self.episode_rewards),
                "game/win_count": self.win_count,
                "game/loss_count": self.loss_count,
                "game/draw_count": self.draw_count,
            }
            
            if total_games > 0:
                metrics["game/win_rate"] = self.win_count / total_games
                metrics["game/loss_rate"] = self.loss_count / total_games
            
            wandb.log(metrics, step=self.num_timesteps)
        
        return True
    
    def _on_training_end(self) -> None:
        """Log final statistics."""
        if self.episode_rewards:
            import wandb
            
            total_games = self.win_count + self.loss_count + self.draw_count
            
            wandb.run.summary["final/total_episodes"] = len(self.episode_rewards)
            wandb.run.summary["final/mean_reward"] = np.mean(self.episode_rewards)
            wandb.run.summary["final/win_count"] = self.win_count
            wandb.run.summary["final/loss_count"] = self.loss_count
            
            if total_games > 0:
                wandb.run.summary["final/win_rate"] = self.win_count / total_games


def make_env(
    player_idx: int = 0,
    rules: str = "tenhou",
    opponent_policy: str = "random",
    seed: int = None,
    reward_shaping: bool = True,
    rank: int = 0,
):
    """Create a wrapped Riichi Mahjong environment."""
    def _init():
        env_seed = seed + rank if seed is not None else None
        env = RiichiMahjongEnv(
            player_idx=player_idx,
            rules=rules,
            opponent_policy=opponent_policy,
            seed=env_seed,
            reward_shaping=reward_shaping,
        )
        env = Monitor(env)
        return env
    return _init


def train_riichi_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    rules: str = "tenhou",
    opponent_policy: str = "random",
    reward_shaping: bool = True,
    save_dir: str = "models/riichi_ppo",
    eval_freq: int = 10000,
    seed: int = 42,
    device: str = "auto",
    verbose: int = 1,
    use_wandb: bool = False,
    wandb_project: str = "riichi-mahjong-rl",
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    """
    Train a PPO agent for Riichi Mahjong.
    """
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = wandb_run_name or f"ppo_{rules}_{timestamp}"
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        import wandb
        import os
        
        # Fix for multiprocessing issues
        os.environ["WANDB_START_METHOD"] = "thread"
        
        config = {
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "rules": rules,
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
            save_code=True,
            settings=wandb.Settings(start_method="thread"),
        )
    
    print("=" * 60)
    print(f"Riichi Mahjong PPO Training ({rules.upper()} rules)")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Opponent policy: {opponent_policy}")
    print(f"Device: {device}")
    print(f"WandB logging: {'enabled' if use_wandb else 'disabled'}")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environments
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(
                player_idx=0,
                rules=rules,
                opponent_policy=opponent_policy,
                seed=seed,
                reward_shaping=reward_shaping,
                rank=i,
            )
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(
                player_idx=0,
                rules=rules,
                opponent_policy=opponent_policy,
                seed=seed,
                reward_shaping=reward_shaping,
                rank=0,
            )
        ])
    
    # Evaluation environment
    eval_env = DummyVecEnv([
        make_env(
            player_idx=0,
            rules=rules,
            opponent_policy=opponent_policy,
            seed=seed + 1000,
            reward_shaping=False,
            rank=0,
        )
    ])
    
    # Create agent with Riichi-specific feature extractor
    policy_kwargs = {
        "features_extractor_class": RiichiFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 512},
    }
    
    from stable_baselines3 import PPO
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
        tensorboard_log=str(save_path / "tensorboard") if use_wandb else None,
    )
    
    # Wrap in SB3Agent-like object for consistency
    class AgentWrapper:
        def __init__(self, model):
            self.model = model
        def save(self, path):
            self.model.save(path)
    
    agent = AgentWrapper(model)
    
    # Callbacks
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=str(save_path / "checkpoints"),
        name_prefix=f"riichi_ppo_{rules}",
    )
    callbacks.append(checkpoint_callback)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(save_path / "logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Add wandb callback if enabled
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
    
    # Log final model to wandb
    if use_wandb:
        import wandb
        wandb.save(str(final_path) + ".zip")
        wandb.finish()
    
    env.close()
    eval_env.close()
    
    return agent, str(save_path)


def main():
    parser = argparse.ArgumentParser(description="Train PPO for Riichi Mahjong")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    
    # Environment parameters
    parser.add_argument("--rules", type=str, default="tenhou", choices=["tenhou", "ema"])
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy"])
    parser.add_argument("--no-reward-shaping", action="store_true")
    
    # Saving parameters
    parser.add_argument("--save-dir", type=str, default="models/riichi_ppo")
    parser.add_argument("--eval-freq", type=int, default=10000)
    
    # General parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    
    # WandB parameters
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="riichi-mahjong-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    args = parser.parse_args()
    
    train_riichi_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        rules=args.rules,
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
