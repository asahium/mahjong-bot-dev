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
mcr-bot-dev/
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
│   └── train_riichi_dqn.py   # Riichi DQN training
├── tests/                    # Unit tests
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mcr-bot-dev.git
cd mcr-bot-dev

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

```bash
# Train MCR agent
python training/train_ppo.py --timesteps 1000000 --n-envs 4

# Train Riichi agent (Tenhou rules)
python training/train_riichi_ppo.py --timesteps 1000000 --rules tenhou

# Train Riichi agent (EMA rules)
python training/train_riichi_ppo.py --timesteps 1000000 --rules ema

# With Apple Silicon GPU (MPS)
python training/train_riichi_ppo.py --timesteps 1000000 --device mps
```

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

## Training Tips

1. **Start Simple**: Train against random opponents first
2. **Reward Shaping**: Enable for faster initial learning
3. **Parallel Environments**: Use `n_envs=4+` for PPO
4. **Monitor Progress**: `tensorboard --logdir models/riichi_ppo/logs`
5. **Curriculum Learning**: Progress from random → greedy → self-play

## References

- [MCR Rules](http://mahjong-europe.org/rules.html)
- [Riichi Book I](http://riichi.wiki)
- [Suphx: Mastering Mahjong with Deep RL](https://arxiv.org/abs/2003.13590)
- [RLCard: A Toolkit for RL in Card Games](https://arxiv.org/abs/1910.04376)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## License

MIT License
