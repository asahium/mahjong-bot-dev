# Mahjong RL Agent

A reinforcement learning environment and training framework for both **Chinese Official Mahjong (MCR)** and **Japanese Riichi Mahjong**.

## Overview

This project provides:

- **Complete MCR Game Engine**: Full implementation of MCR rules with all 81 scoring patterns
- **Complete Riichi Game Engine**: Full implementation of Japanese Mahjong with 40+ Yaku
- **Gymnasium Environments**: Compatible with Stable Baselines3 and other RL frameworks
- **Training Scripts**: Ready-to-use PPO and DQN training configurations
- **Tenhou Client Interface**: Abstraction for connecting trained agents to Tenhou (for future use)
- **Baseline Agents**: Random and greedy agents for benchmarking

## Project Structure

```
mahjong-bot-dev/
├── mcr_mahjong/              # MCR (Chinese Official) game engine
│   ├── tiles.py              # Tile definitions (136 tiles)
│   ├── player.py             # Player state and melds
│   ├── wall.py               # Wall and dealing
│   ├── game.py               # Game logic
│   └── scoring.py            # 81 MCR scoring patterns
├── riichi_mahjong/           # Riichi (Japanese) game engine
│   ├── game.py               # Riichi game logic
│   ├── player.py             # Player with Riichi/Furiten
│   ├── scoring.py            # 40+ Yaku, Han/Fu system
│   ├── dora.py               # Dora system
│   └── rules.py              # EMA/Tenhou rule configs
├── envs/                     # Gymnasium environments
│   ├── mcr_env.py            # MCRMahjongEnv
│   └── riichi_env.py         # RiichiMahjongEnv
├── agents/                   # Agent implementations
│   ├── random_agent.py       # Random/Greedy baselines
│   └── sb3_agent.py          # Stable Baselines3 wrapper
├── clients/                  # Online platform interfaces
│   └── tenhou_client.py      # Tenhou client interface
├── training/                 # Training scripts
│   ├── train_ppo.py          # MCR PPO training
│   ├── train_dqn.py          # MCR DQN training
│   ├── train_riichi_ppo.py   # Riichi PPO training
│   ├── train_riichi_dqn.py   # Riichi DQN training
│   └── train_selfplay.py     # Self-play training for Riichi
├── benchmark.py              # Bot evaluation on predefined hands
├── play_against_bot.py       # Human vs Bot gameplay
├── tests/                    # Unit tests
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mahjong-bot-dev.git
cd mahjong-bot-dev

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### MCR Mahjong

```python
from mcr_mahjong import Game, Action, ActionType

game = Game(seed=42)
game.start_game()

actions = game.get_valid_actions(0)
action = actions[0]
game.step(action)
```

### Riichi Mahjong

```python
from riichi_mahjong import RiichiGame
from riichi_mahjong.rules import TENHOU_RULES

game = RiichiGame(rules=TENHOU_RULES, seed=42)
game.reset()
game.start_round()

actions = game.get_valid_actions(0)
print(f"Valid actions: {actions}")
```

### Training Agents

#### PPO Training (Recommended)

```bash
# Train MCR agent with parallel environments
python training/train_ppo.py --timesteps 1000000 --n-envs 4

# Train Riichi agent (Tenhou rules)
python training/train_riichi_ppo.py --timesteps 1000000 --rules tenhou

# Train Riichi agent (EMA rules)
python training/train_riichi_ppo.py --timesteps 1000000 --rules ema
```

#### DQN Training

```bash
# Train Riichi agent with DQN
python training/train_riichi_dqn.py --timesteps 500000 --rules tenhou

# DQN with custom exploration
python training/train_riichi_dqn.py \
    --timesteps 1000000 \
    --buffer-size 100000 \
    --exploration-fraction 0.2 \
    --eps-start 1.0 \
    --eps-end 0.05
```

#### Parallel Training

Use multiple environments for faster PPO training:

```bash
# 8 parallel environments (recommended for PPO)
python training/train_riichi_ppo.py --timesteps 2000000 --n-envs 8

# 16 environments for faster training on multi-core systems
python training/train_riichi_ppo.py --timesteps 5000000 --n-envs 16
```

**Note:** Parallel environments use `SubprocVecEnv` for true multiprocessing. Each environment runs in a separate process, providing linear speedup with CPU cores.

#### GPU Acceleration

```bash
# Apple Silicon GPU (MPS)
python training/train_riichi_ppo.py --timesteps 1000000 --device mps

# NVIDIA GPU (CUDA)
python training/train_riichi_ppo.py --timesteps 1000000 --device cuda

# Automatic device selection
python training/train_riichi_ppo.py --timesteps 1000000 --device auto
```

#### Hyperparameter Tuning

```bash
# Full hyperparameter control for PPO
python training/train_riichi_ppo.py \
    --timesteps 2000000 \
    --n-envs 8 \
    --lr 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 10 \
    --gamma 0.99 \
    --ent-coef 0.01 \
    --opponent random \
    --rules tenhou

# With reward shaping disabled
python training/train_riichi_ppo.py --timesteps 1000000 --no-reward-shaping
```

**PPO Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 1,000,000 | Total training steps |
| `--n-envs` | 4 | Parallel environments |
| `--lr` | 3e-4 | Learning rate |
| `--n-steps` | 2048 | Steps per update per env |
| `--batch-size` | 64 | Minibatch size |
| `--n-epochs` | 10 | Epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--ent-coef` | 0.01 | Entropy coefficient |
| `--opponent` | random | Opponent (random/greedy) |
| `--rules` | tenhou | Rule set (tenhou/ema) |
| `--device` | auto | Device (auto/cpu/cuda/mps) |

### Logging with Weights & Biases

Enable wandb logging to track experiments:

```bash
# First, login to wandb
wandb login

# Train with wandb logging
python training/train_ppo.py --timesteps 1000000 --wandb --wandb-project my-mahjong-project

# Train Riichi with wandb
python training/train_riichi_ppo.py \
    --timesteps 1000000 \
    --rules tenhou \
    --wandb \
    --wandb-project riichi-experiments \
    --wandb-run-name ppo-tenhou-v1
```

**WandB Metrics:**
- `rollout/ep_rew_mean` - Average episode reward
- `game/win_rate` - Percentage of games won
- `game/win_count` / `game/loss_count` - Win/loss statistics
- `train/*` - PPO/DQN training metrics

### Self-Play Training

Train agents by playing against themselves for stronger opponents:

```bash
# Basic self-play training
python training/train_selfplay.py --timesteps 2000000 --n-envs 8

# With custom parameters
python training/train_selfplay.py \
    --timesteps 5000000 \
    --n-envs 16 \
    --lr 3e-4 \
    --batch-size 256 \
    --rules tenhou \
    --device auto
```

**Self-Play Features:**
- **SelfPlayEnv**: Agent plays all 4 seats in the game
- **Past-Self Play**: Opponent models updated periodically (every 50k steps)
- **Enhanced Reward Shaping**: Shanten-based rewards, riichi bonuses, survival rewards
- **Parallel Environments**: SubprocVecEnv for faster training

**Self-Play Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--timesteps` | 2,000,000 | Total training steps |
| `--n-envs` | 8 | Parallel environments |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | Minibatch size |
| `--rules` | tenhou | Rule set (tenhou/ema) |
| `--save-dir` | models/selfplay | Model save directory |
| `--seed` | 42 | Random seed |
| `--device` | auto | Device (auto/cpu/cuda/mps) |

### Using the Gymnasium Environments

```python
from envs.riichi_env import RiichiMahjongEnv
import numpy as np

# Create Riichi environment with Tenhou rules
env = RiichiMahjongEnv(rules="tenhou", render_mode="human")
obs, info = env.reset()

for _ in range(100):
    valid_mask = obs["valid_actions"]
    valid_indices = np.where(valid_mask == 1)[0]
    
    action = np.random.choice(valid_indices)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Riichi Mahjong Rules

### Key Differences from MCR

| Feature | MCR | Riichi |
|---------|-----|--------|
| Minimum to win | 8 points | 1 Yaku |
| Scoring | Pattern-based | Han/Fu system |
| Riichi | No | Yes (1000pt bet) |
| Dora | No | Yes |
| Furiten | No | Yes |

### Rule Variations

| Feature | EMA | Tenhou |
|---------|-----|--------|
| Red Dora | No | 3 (one per suit) |
| Starting Points | 30000 | 25000 |
| Uma | 15/5/-5/-15 | 20/10/-10/-20 |
| Multiple Ron | Yes | No (head bump) |

### Yaku (Winning Patterns)

**1 Han:**
- Riichi (立直) - Declare ready hand
- Tsumo (自摸) - Self-draw win
- Pinfu (平和) - All sequences, no-point hand
- Tanyao (断幺九) - All simples
- Yakuhai (役牌) - Value tiles

**2 Han:**
- Chiitoitsu (七対子) - Seven pairs
- Sanshoku (三色同順) - Three-colored straight
- Toitoi (対々和) - All triplets

**Yakuman (Limit Hands):**
- Kokushi Musou (国士無双) - Thirteen orphans
- Suuankou (四暗刻) - Four concealed triplets
- Daisangen (大三元) - Big three dragons
- Chuuren Poutou (九蓮宝燈) - Nine gates

### Dora System

Dora tiles add han without being yaku:
- **Dora**: Based on indicator tile
- **Uradora**: Revealed on riichi win
- **Kandora**: Revealed after kan
- **Akadora**: Red 5s (Tenhou rules)

## Future: Playing on Tenhou

The project includes a `TenhouClientInterface` for future integration:

```python
from clients.tenhou_client import TenhouClientInterface, MockTenhouClient

# For testing locally
client = MockTenhouClient()

# For real Tenhou connection (requires implementation)
# client = RealTenhouClient()

async def main():
    await client.connect(username="your_username")
    await client.join_game()
    
    while not client.game_over:
        state = await client.get_state()
        obs = state.to_observation()
        action, _ = agent.predict(obs)
        await client.send_action(action)
```

**Note**: Using bots on Tenhou may violate their terms of service.

## Observation Space (Riichi)

| Key | Shape | Description |
|-----|-------|-------------|
| `hand` | (34,) | Tile counts in hand |
| `melds` | (4, 4, 34) | Melds for all players |
| `discards` | (4, 34) | Discard counts |
| `dora_indicators` | (5, 34) | Dora indicators (one-hot) |
| `riichi_status` | (4,) | Riichi status per player |
| `furiten` | (1,) | Furiten status |
| `scores` | (4,) | Current scores |
| `valid_actions` | (250,) | Valid action mask |
| `game_info` | (12,) | Game state info |

## Action Space (Riichi)

| Range | Action Type |
|-------|-------------|
| 0-33 | Discard tile |
| 34-67 | Riichi + Discard |
| 68-101 | Chii |
| 102-135 | Pon |
| 136-169 | Kan (open) |
| 170-203 | Ankan (concealed) |
| 204-237 | Shouminkan |
| 238 | Tsumo |
| 239 | Ron |
| 240 | Pass |
| 241 | Draw |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=mcr_mahjong --cov=riichi_mahjong --cov=envs
```

## Benchmarking

Evaluate your trained model on predefined hands to assess decision quality:

```bash
# Run benchmark on a trained model
python benchmark.py --model models/riichi_ppo/ppo_tenhou_*/best/best_model.zip

# With verbose output
python benchmark.py --model models/riichi_ppo/ppo_tenhou_*/final_model.zip --verbose
```

**Benchmark Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Discard | 4 | Tile efficiency, tenpai preservation |
| Riichi | 2 | When to declare riichi |
| Defense | 2 | Safe tile selection against riichi |

**Scoring:**
- **PASS (100%)**: Model chose an expected optimal action
- **OKAY (50%)**: Model chose acceptable but not optimal action
- **FAIL (0%)**: Model made a clear mistake

**Example Output:**

```
RIICHI MAHJONG BOT BENCHMARK
============================================================
PASS Tenpai - Keep wait tiles
   Situation: discard
   Bot chose: Discard 1p
   Expected:  Discard 1p
...
============================================================
SUMMARY
Total tests: 8
Passed: 5
Failed: 1
Okay: 2

Overall Score: 75.0%
```

A score of 70%+ indicates the bot has learned basic mahjong strategy.

## Play Against Bot

Test your trained model by playing against it or watching it play:

### Interactive Play

```bash
# Play against your trained bot
python play_against_bot.py --model models/riichi_ppo/ppo_tenhou_*/best/best_model.zip

# Use EMA rules
python play_against_bot.py --model path/to/model.zip --rules ema
```

### Watch Mode

```bash
# Watch the bot play 5 games against random opponents
python play_against_bot.py --model path/to/model.zip --watch --games 5

# Watch 10 games
python play_against_bot.py --model path/to/model.zip --watch --games 10
```

**Controls:**
- Enter action number to play
- Type `q` to quit
- Type `y` to play again after game ends

## Training Tips

### Getting Started

1. **Start Simple**: Train against random opponents first to learn basic tile efficiency
2. **Reward Shaping**: Keep enabled (`--reward-shaping`) for faster initial learning
3. **Use PPO**: Generally more stable than DQN for this environment

### Parallel Training Recommendations

| CPU Cores | Recommended `--n-envs` | Notes |
|-----------|------------------------|-------|
| 4 | 4 | Good for laptops |
| 8 | 8 | Standard workstation |
| 16+ | 16 | Server/high-end desktop |

**Note:** PPO benefits significantly from parallel environments. More environments = faster wall-clock training time.

### Curriculum Learning

Progress through increasingly difficult opponents:

```bash
# Stage 1: Random opponents (500k-1M steps)
python training/train_riichi_ppo.py --timesteps 1000000 --opponent random

# Stage 2: Greedy opponents (500k-1M steps)
python training/train_riichi_ppo.py --timesteps 1000000 --opponent greedy

# Stage 3: Self-play (2M+ steps)
python training/train_selfplay.py --timesteps 2000000 --n-envs 8
```

### Monitoring Training

```bash
# TensorBoard (if using wandb)
tensorboard --logdir models/riichi_ppo/*/tensorboard

# Check model checkpoints
ls models/riichi_ppo/*/checkpoints/

# Evaluate with benchmark
python benchmark.py --model models/riichi_ppo/*/best/best_model.zip
```

### GPU vs CPU

- **CPU**: Sufficient for most training, especially with parallel envs
- **MPS (Apple Silicon)**: Good speedup on M1/M2/M3 Macs
- **CUDA**: Best for large batch sizes and long training runs

### Recommended Training Pipeline

```bash
# 1. Initial training against random (1M steps)
python training/train_riichi_ppo.py \
    --timesteps 1000000 \
    --n-envs 8 \
    --rules tenhou \
    --wandb

# 2. Benchmark the model
python benchmark.py --model models/riichi_ppo/*/best/best_model.zip

# 3. Self-play refinement (2M steps)
python training/train_selfplay.py \
    --timesteps 2000000 \
    --n-envs 16 \
    --rules tenhou

# 4. Final evaluation
python play_against_bot.py --model models/selfplay/*/final_model.zip --watch --games 10
```

## References

- [MCR Rules](http://mahjong-europe.org/rules.html)
- [Riichi Book I](http://riichi.wiki)
- [Suphx: Mastering Mahjong with Deep RL](https://arxiv.org/abs/2003.13590)
- [RLCard: A Toolkit for RL in Card Games](https://arxiv.org/abs/1910.04376)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## License

MIT License
