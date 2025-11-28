#!/usr/bin/env python3
"""
Play against your trained Riichi Mahjong bot.

Usage:
    python play_against_bot.py --model models/riichi_ppo/ppo_tenhou_*/best/best_model.zip
    python play_against_bot.py --model models/riichi_ppo/ppo_tenhou_*/final_model.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from envs.riichi_env import RiichiMahjongEnv
from riichi_mahjong.game import RiichiActionType


def get_action_name(action_idx: int) -> str:
    """Convert action index to human-readable name."""
    if 0 <= action_idx < 34:
        return f"Discard tile {action_idx}"
    elif 34 <= action_idx < 68:
        return f"Riichi + Discard tile {action_idx - 34}"
    elif 68 <= action_idx < 102:
        return f"Chii (tile {action_idx - 68})"
    elif 102 <= action_idx < 136:
        return f"Pon (tile {action_idx - 102})"
    elif 136 <= action_idx < 170:
        return f"Open Kan (tile {action_idx - 136})"
    elif 170 <= action_idx < 204:
        return f"Closed Kan (tile {action_idx - 170})"
    elif 204 <= action_idx < 238:
        return f"Add to Pon (tile {action_idx - 204})"
    elif action_idx == 238:
        return "Tsumo (self-draw win)"
    elif action_idx == 239:
        return "Ron (win from discard)"
    elif action_idx == 240:
        return "Pass"
    elif action_idx == 241:
        return "Draw"
    else:
        return f"Action {action_idx}"


def play_game(model_path: str, rules: str = "tenhou", human_player: int = 0):
    """
    Play a game against the trained bot.
    
    Args:
        model_path: Path to the trained model
        rules: Rule set ("tenhou" or "ema")
        human_player: Which player is human (0-3)
    """
    print("=" * 60)
    print("🀄 Riichi Mahjong - Play Against Bot")
    print("=" * 60)
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    print()
    
    # Create environment (human plays as player 0)
    env = RiichiMahjongEnv(
        player_idx=human_player,
        rules=rules,
        opponent_policy="model",  # Will be overridden
        render_mode="human",
    )
    
    obs, info = env.reset()
    total_reward = 0
    
    print("Game started! You are Player 0.")
    print("Type action number to play, or 'q' to quit.")
    print()
    
    while True:
        env.render()
        print()
        
        # Get valid actions
        valid_mask = obs["valid_actions"]
        valid_indices = np.where(valid_mask == 1)[0]
        
        if len(valid_indices) == 0:
            print("No valid actions! Game might be stuck.")
            break
        
        # Check whose turn it is
        is_human_turn = (env.game.current_player == human_player or 
                        env.game.phase.value == 3)  # Claiming phase
        
        if is_human_turn:
            # Human's turn
            print("Your valid actions:")
            for i, action_idx in enumerate(valid_indices):
                print(f"  [{i}] {get_action_name(action_idx)}")
            print()
            
            while True:
                try:
                    user_input = input("Enter action number (or 'q' to quit): ").strip()
                    if user_input.lower() == 'q':
                        print("Thanks for playing!")
                        return
                    
                    choice = int(user_input)
                    if 0 <= choice < len(valid_indices):
                        action = valid_indices[choice]
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(valid_indices) - 1}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            # Bot's turn - use the model
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Ensure action is valid
            if valid_mask[action] != 1:
                # Pick a random valid action
                action = np.random.choice(valid_indices)
            
            print(f"Bot plays: {get_action_name(action)}")
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print()
            print("=" * 60)
            print("GAME OVER!")
            print("=" * 60)
            env.render()
            
            if "episode" in info:
                ep_info = info["episode"]
                print(f"Total reward: {ep_info['r']:.2f}")
                print(f"Episode length: {ep_info['l']}")
                
                result = ep_info.get("result")
                if result and hasattr(result, 'winner'):
                    if result.winner == human_player:
                        print("🎉 YOU WIN!")
                    elif result.winner is not None:
                        print(f"Bot (Player {result.winner}) wins...")
                    else:
                        print("Draw!")
            
            play_again = input("\nPlay again? (y/n): ").strip().lower()
            if play_again == 'y':
                obs, info = env.reset()
                total_reward = 0
                print("\n" + "=" * 60)
                print("New game started!")
                print("=" * 60 + "\n")
            else:
                break
    
    env.close()
    print("Thanks for playing!")


def watch_bots(model_path: str, rules: str = "tenhou", num_games: int = 5):
    """Watch bots play against each other."""
    print("=" * 60)
    print("🀄 Watching Bots Play")
    print("=" * 60)
    
    model = PPO.load(model_path)
    
    env = RiichiMahjongEnv(
        player_idx=0,
        rules=rules,
        opponent_policy="random",
        render_mode="human",
    )
    
    wins = 0
    for game in range(num_games):
        print(f"\n--- Game {game + 1}/{num_games} ---\n")
        obs, info = env.reset()
        
        while True:
            # Bot action
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            valid_mask = obs["valid_actions"]
            if valid_mask[action] != 1:
                valid_indices = np.where(valid_mask == 1)[0]
                action = np.random.choice(valid_indices) if len(valid_indices) > 0 else 240
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                env.render()
                if "episode" in info:
                    result = info["episode"].get("result")
                    if result and hasattr(result, 'winner') and result.winner == 0:
                        wins += 1
                        print("✓ Bot wins!")
                    else:
                        print("✗ Bot loses")
                break
    
    print(f"\n=== Results: {wins}/{num_games} wins ({100*wins/num_games:.1f}%) ===")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Play against your Riichi Mahjong bot")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--rules", type=str, default="tenhou",
                       choices=["tenhou", "ema"])
    parser.add_argument("--watch", action="store_true",
                       help="Watch bots play instead of playing yourself")
    parser.add_argument("--games", type=int, default=5,
                       help="Number of games to watch (with --watch)")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_bots(args.model, args.rules, args.games)
    else:
        play_game(args.model, args.rules)


if __name__ == "__main__":
    main()

