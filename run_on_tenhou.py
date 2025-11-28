#!/usr/bin/env python3
"""
Run your trained bot on Tenhou.

Usage:
    # Test with mock client (local simulation)
    python run_on_tenhou.py --model models/riichi_ppo/*/best/best_model.zip --mock --games 5
    
    # Real Tenhou (requires websockets: pip install websockets)
    python run_on_tenhou.py --model models/riichi_ppo/*/best/best_model.zip --username ID12345678-abcd1234

⚠️ Warning: Using bots on Tenhou may violate their Terms of Service.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO

from clients.tenhou_client import (
    MockTenhouClient,
    TenhouGameType,
    TenhouLobby,
    run_agent_on_tenhou,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_with_mock(model_path: str, num_games: int = 5):
    """Test agent with mock Tenhou client."""
    logger.info(f"Loading model from: {model_path}")
    agent = PPO.load(model_path)
    
    logger.info("Creating mock Tenhou client (local simulation)...")
    client = MockTenhouClient()
    
    logger.info(f"Running {num_games} games...")
    results = await run_agent_on_tenhou(
        agent=agent,
        client=client,
        username="TestBot",
        num_games=num_games,
        game_type=TenhouGameType.HANCHAN,
    )
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Results: {len(results)} games completed")
    logger.info(f"{'='*50}")
    
    return results


async def run_on_real_tenhou(
    model_path: str,
    username: str,
    num_games: int = 1,
    game_type: str = "hanchan",
    lobby: str = "ippan",
):
    """Run bot on real Tenhou server."""
    try:
        from clients.tenhou_websocket import TenhouWebSocketClient, run_bot_on_tenhou
    except ImportError as e:
        logger.error(f"Failed to import Tenhou WebSocket client: {e}")
        logger.error("Install websockets: pip install websockets")
        return []
    
    gt = TenhouGameType.HANCHAN if game_type == "hanchan" else TenhouGameType.TONPUU
    
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Connecting to Tenhou as: {username}")
    logger.info(f"Game type: {game_type}, Games: {num_games}")
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    ⚠️  WARNING ⚠️                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Using bots on Tenhou may violate their Terms of Service!        ║
║                                                                  ║
║  - Your account may be banned                                    ║
║  - This is for educational/research purposes only                ║
║                                                                  ║
║  Press Ctrl+C within 5 seconds to cancel...                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    await asyncio.sleep(5)
    
    results = await run_bot_on_tenhou(
        model_path=model_path,
        username=username,
        num_games=num_games,
        game_type=gt,
    )
    
    logger.info(f"\n{'='*50}")
    logger.info("RESULTS")
    logger.info(f"{'='*50}")
    
    for r in results:
        game_num = r.get("game", 0)
        scores = r.get("final_scores", [])
        seat = r.get("seat", 0)
        
        if scores:
            my_score = scores[seat] if seat < len(scores) else 0
            ranking = sorted(scores, reverse=True).index(my_score) + 1
            logger.info(f"Game {game_num}: Score={my_score}, Rank={ranking}/4")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run trained Riichi Mahjong bot on Tenhou",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test locally with mock client
    python run_on_tenhou.py --model models/riichi_ppo/*/best/best_model.zip --mock --games 10
    
    # Run on real Tenhou
    python run_on_tenhou.py --model models/riichi_ppo/*/best/best_model.zip --username NoName
    
    # Run on real Tenhou with specific settings
    python run_on_tenhou.py \\
        --model models/riichi_ppo/*/best/best_model.zip \\
        --username ID12345678-abcd1234 \\
        --games 5 \\
        --game-type hanchan \\
        --lobby ippan
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock client (local simulation, no real Tenhou)")
    parser.add_argument("--username", type=str, default="NoName",
                       help="Tenhou username (for real Tenhou)")
    parser.add_argument("--games", type=int, default=1,
                       help="Number of games to play")
    parser.add_argument("--game-type", type=str, default="hanchan",
                       choices=["hanchan", "tonpuu"],
                       help="Game type (hanchan=East-South, tonpuu=East only)")
    parser.add_argument("--lobby", type=str, default="ippan",
                       choices=["ippan", "joukyuu", "tokujou", "houou"],
                       help="Tenhou lobby")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists() and '*' not in args.model:
        # Try to expand glob
        import glob
        matches = glob.glob(args.model)
        if matches:
            model_path = Path(matches[0])
        else:
            logger.error(f"Model not found: {args.model}")
            sys.exit(1)
    
    if args.mock:
        asyncio.run(test_with_mock(str(model_path), args.games))
    else:
        asyncio.run(run_on_real_tenhou(
            model_path=str(model_path),
            username=args.username,
            num_games=args.games,
            game_type=args.game_type,
            lobby=args.lobby,
        ))


if __name__ == "__main__":
    main()
