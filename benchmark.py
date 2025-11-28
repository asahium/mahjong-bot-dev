#!/usr/bin/env python3
"""
Benchmark for Riichi Mahjong Bot

Tests the bot's decision-making on predefined hands to evaluate learning quality.

Scenarios tested:
1. Discard selection - which tile to discard from various hands
2. Riichi decision - when to declare riichi
3. Call decision - when to chi/pon/kan
4. Defense - safe tile selection when opponent in riichi

Usage:
    python benchmark.py --model models/riichi_ppo/*/best/best_model.zip
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from mcr_mahjong.tiles import Tile, TileSuit


@dataclass
class TestCase:
    """A benchmark test case."""
    name: str
    description: str
    hand: List[int]  # Tile indices (0-33)
    expected_actions: List[int]  # Expected good actions
    bad_actions: List[int]  # Actions that would be mistakes
    situation: str  # "discard", "riichi", "call", "defense"


# Tile index helpers
def man(n: int) -> int:
    """Characters (万) 1-9 -> indices 0-8"""
    return n - 1

def pin(n: int) -> int:
    """Dots (筒) 1-9 -> indices 9-17"""
    return 9 + n - 1

def sou(n: int) -> int:
    """Bamboo (索) 1-9 -> indices 18-26"""
    return 18 + n - 1

def wind(n: int) -> int:
    """Winds: East=0, South=1, West=2, North=3 -> indices 27-30"""
    return 27 + n

def dragon(n: int) -> int:
    """Dragons: White=0, Green=1, Red=2 -> indices 31-33"""
    return 31 + n


# Benchmark test cases
BENCHMARK_TESTS = [
    # =========================================
    # DISCARD SELECTION TESTS
    # =========================================
    TestCase(
        name="Tenpai - Keep wait tiles",
        description="Hand is tenpai waiting on 14m. Should discard isolated tile, not wait tiles.",
        hand=[man(1), man(1), man(1), man(2), man(3), man(4), man(5), man(6), 
              man(7), man(8), man(9), pin(1), pin(1)],  # Waiting 14m
        expected_actions=[pin(1)],  # Discard the pair that's not part of wait
        bad_actions=[man(1), man(4)],  # Don't break the wait
        situation="discard",
    ),
    
    TestCase(
        name="Iishanten - Efficient discard",
        description="One away from tenpai. Should keep useful tiles.",
        hand=[man(1), man(2), man(3), man(5), man(5), man(5), 
              pin(2), pin(3), pin(4), sou(6), sou(7), sou(9), wind(0)],
        expected_actions=[wind(0), sou(9)],  # Discard isolated tiles
        bad_actions=[man(5), pin(3)],  # Don't break groups
        situation="discard",
    ),
    
    TestCase(
        name="Isolated terminals",
        description="Should prefer discarding isolated terminals over simples.",
        hand=[man(1), man(4), man(5), man(6), pin(1), pin(2), pin(3),
              sou(3), sou(4), sou(5), sou(9), dragon(0), dragon(0)],
        expected_actions=[man(1), sou(9)],  # Isolated terminals
        bad_actions=[man(5), pin(2), sou(4)],  # Don't discard middle tiles
        situation="discard",
    ),
    
    TestCase(
        name="Pairs vs sequences",
        description="When choosing between pairs, keep the pair with more potential.",
        hand=[man(2), man(2), man(5), man(5), man(8), man(8),
              pin(3), pin(4), pin(5), sou(2), sou(3), sou(4), wind(3)],
        expected_actions=[wind(3), man(8), man(8)],  # Isolated wind or edge pair
        bad_actions=[man(5), pin(4)],  # Keep middle tiles
        situation="discard",
    ),
    
    # =========================================
    # RIICHI DECISION TESTS
    # =========================================
    TestCase(
        name="Good riichi - Multi-wait",
        description="Tenpai with good wait. Should declare riichi.",
        hand=[man(1), man(2), man(3), man(4), man(5), man(6),
              pin(2), pin(3), pin(4), sou(5), sou(5), sou(6), sou(7)],  # Wait 58s
        expected_actions=[34 + sou(5)],  # Riichi + discard
        bad_actions=[sou(5)],  # Just discard without riichi
        situation="riichi",
    ),
    
    TestCase(
        name="Bad riichi - Single wait",
        description="Tenpai but bad wait (tanki). Consider not riichi.",
        hand=[man(1), man(2), man(3), man(4), man(5), man(6),
              pin(2), pin(3), pin(4), sou(7), sou(8), sou(9), wind(0)],  # Wait East only
        expected_actions=[wind(0)],  # Maybe just discard
        bad_actions=[],  # Riichi is acceptable but not optimal
        situation="riichi",
    ),
    
    # =========================================
    # DEFENSE TESTS
    # =========================================
    TestCase(
        name="Defense - Safe tile selection",
        description="Opponent riichi. Should discard safe tiles.",
        hand=[man(1), man(2), man(3), man(5), man(6), man(7),
              pin(4), pin(5), pin(6), sou(1), sou(5), sou(9), wind(0)],
        expected_actions=[wind(0), sou(1), sou(9)],  # Safe tiles (terminals, honors)
        bad_actions=[man(5), pin(5), sou(5)],  # Dangerous middle tiles
        situation="defense",
    ),
    
    TestCase(
        name="Defense - Genbutsu priority",
        description="When defending, prefer tiles opponent already discarded.",
        hand=[man(3), man(4), man(5), pin(1), pin(2), pin(3),
              sou(4), sou(5), sou(6), sou(7), sou(8), wind(1), wind(1)],
        expected_actions=[wind(1)],  # Safe honor
        bad_actions=[sou(7)],  # Middle tile is dangerous
        situation="defense",
    ),
]


class BenchmarkRunner:
    """Run benchmark tests on a trained model."""
    
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)
        self.results: List[Dict] = []
    
    def create_observation(
        self, 
        hand: List[int],
        riichi_status: List[bool] = None,
    ) -> Dict[str, np.ndarray]:
        """Create observation from hand tiles."""
        # Hand counts
        hand_arr = np.zeros(34, dtype=np.int8)
        for tile_idx in hand:
            hand_arr[tile_idx] += 1
        
        # Empty melds
        melds = np.zeros((4, 4, 34), dtype=np.int8)
        
        # Empty discards
        discards = np.zeros((4, 34), dtype=np.int8)
        
        # Dora (random)
        dora = np.zeros((5, 34), dtype=np.int8)
        dora[0, 4] = 1  # 5m as dora indicator
        
        # Riichi status
        if riichi_status is None:
            riichi_status = [False, False, False, False]
        riichi = np.array([1 if r else 0 for r in riichi_status], dtype=np.int8)
        
        # Furiten
        furiten = np.array([0], dtype=np.int8)
        
        # Scores
        scores = np.array([25000, 25000, 25000, 25000], dtype=np.float32)
        
        # Valid actions (all discards valid for simplicity)
        valid = np.zeros(250, dtype=np.int8)
        for tile_idx in range(34):
            if hand_arr[tile_idx] > 0:
                valid[tile_idx] = 1  # Discard
                valid[34 + tile_idx] = 1  # Riichi + discard
        valid[240] = 1  # Pass always valid
        
        # Game info
        game_info = np.array([
            0,  # current_player
            2,  # phase (discarding)
            0,  # round_wind
            0,  # seat
            0,  # dealer
            0,  # honba
            0,  # riichi_sticks
            70, # wall remaining
            10, # turn count
            1,  # is my turn
            0,  # can tsumo
            0,  # can ron
        ], dtype=np.float32)
        
        return {
            "hand": hand_arr,
            "melds": melds,
            "discards": discards,
            "dora_indicators": dora,
            "riichi_status": riichi,
            "furiten": furiten,
            "scores": scores,
            "valid_actions": valid,
            "game_info": game_info,
        }
    
    def run_test(self, test: TestCase) -> Dict:
        """Run a single test case."""
        # Create observation
        riichi_status = [False, True, False, False] if test.situation == "defense" else None
        obs = self.create_observation(test.hand, riichi_status)
        
        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        
        # Evaluate
        is_expected = action in test.expected_actions
        is_bad = action in test.bad_actions
        
        # Score
        if is_expected:
            score = 1.0
            status = "✓ PASS"
        elif is_bad:
            score = 0.0
            status = "✗ FAIL"
        else:
            score = 0.5
            status = "~ OKAY"
        
        result = {
            "name": test.name,
            "situation": test.situation,
            "action": action,
            "expected": test.expected_actions,
            "bad": test.bad_actions,
            "score": score,
            "status": status,
        }
        
        self.results.append(result)
        return result
    
    def run_all(self) -> float:
        """Run all benchmark tests."""
        print("\n" + "=" * 70)
        print("🎯 RIICHI MAHJONG BOT BENCHMARK")
        print("=" * 70 + "\n")
        
        total_score = 0
        
        for test in BENCHMARK_TESTS:
            result = self.run_test(test)
            
            action_name = self._action_to_str(result["action"])
            expected_names = [self._action_to_str(a) for a in result["expected"]]
            
            print(f"{result['status']} {test.name}")
            print(f"   Situation: {test.situation}")
            print(f"   Bot chose: {action_name}")
            print(f"   Expected:  {', '.join(expected_names)}")
            print(f"   {test.description}")
            print()
            
            total_score += result["score"]
        
        # Summary
        avg_score = total_score / len(BENCHMARK_TESTS) if BENCHMARK_TESTS else 0
        
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total tests: {len(BENCHMARK_TESTS)}")
        print(f"Passed: {sum(1 for r in self.results if r['score'] == 1.0)}")
        print(f"Failed: {sum(1 for r in self.results if r['score'] == 0.0)}")
        print(f"Okay:   {sum(1 for r in self.results if r['score'] == 0.5)}")
        print(f"\nOverall Score: {avg_score * 100:.1f}%")
        print("=" * 70)
        
        # Per-situation breakdown
        print("\nBy Situation:")
        for situation in ["discard", "riichi", "defense"]:
            sit_results = [r for r in self.results if r["situation"] == situation]
            if sit_results:
                sit_score = sum(r["score"] for r in sit_results) / len(sit_results)
                print(f"  {situation}: {sit_score * 100:.1f}%")
        
        return avg_score
    
    def _action_to_str(self, action: int) -> str:
        """Convert action index to readable string."""
        if 0 <= action < 34:
            return f"Discard {self._tile_name(action)}"
        elif 34 <= action < 68:
            return f"Riichi + {self._tile_name(action - 34)}"
        elif action == 240:
            return "Pass"
        else:
            return f"Action {action}"
    
    def _tile_name(self, idx: int) -> str:
        """Convert tile index to name."""
        if idx < 9:
            return f"{idx + 1}m"
        elif idx < 18:
            return f"{idx - 8}p"
        elif idx < 27:
            return f"{idx - 17}s"
        elif idx < 31:
            winds = ["East", "South", "West", "North"]
            return winds[idx - 27]
        else:
            dragons = ["White", "Green", "Red"]
            return dragons[idx - 31]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Riichi Mahjong bot")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Expand glob if needed
    import glob
    models = glob.glob(args.model)
    if models:
        model_path = sorted(models)[-1]
    else:
        model_path = args.model
    
    print(f"Loading model: {model_path}")
    
    runner = BenchmarkRunner(model_path)
    score = runner.run_all()
    
    # Return exit code based on score
    if score >= 0.7:
        print("\n✓ Bot passed benchmark!")
        sys.exit(0)
    else:
        print("\n✗ Bot needs more training")
        sys.exit(1)


if __name__ == "__main__":
    main()

