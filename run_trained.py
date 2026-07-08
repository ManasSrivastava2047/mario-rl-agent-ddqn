"""
run_trained.py

Load a trained Mario agent and watch it play Super Mario Bros 1-1.

Uses the checkpoint saved by main.py (default: checkpoints/latest/mario_latest.chkpt).
Pass --best to use the best-performing checkpoint instead.

Run:
    python run_trained.py
    python run_trained.py --checkpoint checkpoints/latest/mario_best.chkpt
    python run_trained.py --episodes 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame
import torch

from wrappers import make_env
from mario import Mario


LATEST_CHECKPOINT = Path("checkpoints") / "latest" / "mario_latest.chkpt"
BEST_CHECKPOINT = Path("checkpoints") / "latest" / "mario_best.chkpt"

SCALE = 3
FPS = 60


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated) or bool(truncated)
    else:
        obs, reward, done, info = result
    return obs, reward, done, info


def get_frame(env):
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    return np.asarray(frame)


def find_checkpoint(use_best: bool) -> Path:
    path = BEST_CHECKPOINT if use_best else LATEST_CHECKPOINT
    if path.exists():
        return path

    # Fall back: newest session folder
    checkpoints = sorted(Path("checkpoints").glob("*/mario_latest.chkpt"))
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        "No checkpoint found. Train first with: python main.py\n"
        f"Looked for: {path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run a trained Mario RL agent")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .chkpt file (default: checkpoints/latest/mario_latest.chkpt)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Use mario_best.chkpt instead of mario_latest.chkpt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3, use 0 for unlimited until Esc)",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_checkpoint(args.best)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint}")

    env = make_env()
    save_dir = checkpoint.parent
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    trained_episode = mario.load(checkpoint, exploit_only=True)
    print(f"Loaded model from episode {trained_episode} (exploit mode, epsilon=0)")

    obs = reset_env(env)
    frame = get_frame(env)
    h, w = frame.shape[0], frame.shape[1]

    pygame.init()
    pygame.display.set_caption("Mario RL - Trained Agent")
    screen = pygame.display.set_mode((w * SCALE, h * SCALE + 40))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    info = {"coins": 0, "life": 2, "score": 0, "time": 400, "world": 1, "stage": 1}
    episode = 0
    total_episodes = args.episodes
    done = False
    running = True
    last_reward = 0.0

    print("Press Esc or close the window to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not done:
            with torch.no_grad():
                action = mario.act(obs)
            obs, last_reward, done, info = step_env(env, action)

        frame = get_frame(env)
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (w * SCALE, h * SCALE))
        screen.blit(surface, (0, 0))

        pygame.draw.rect(screen, (0, 0, 0), (0, h * SCALE, w * SCALE, 40))
        status = (
            f"Ep {episode + 1}   "
            f"World {info.get('world', 1)}-{info.get('stage', 1)}   "
            f"X: {info.get('x_pos', 0)}   "
            f"Score: {info.get('score', 0)}   "
            f"Reward: {last_reward:.0f}"
        )
        if info.get("flag_get"):
            status += "   FLAG!"
        if done:
            status += "   -- DONE --"
        text_surface = font.render(status, True, (255, 255, 255))
        screen.blit(text_surface, (8, h * SCALE + 10))

        pygame.display.flip()
        clock.tick(FPS)

        if done:
            episode += 1
            if total_episodes > 0 and episode >= total_episodes:
                running = False
            else:
                obs = reset_env(env)
                done = False

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
