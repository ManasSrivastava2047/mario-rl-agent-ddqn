"""
main.py

Training entry point. Wires together:
    wrappers.make_env()  -> preprocessed Super Mario Bros environment
    network.MarioNet     -> the CNN (used internally by Mario)
    mario.Mario          -> the DDQN agent
    loggers.MetricLogger -> progress tracking / plots

Runs until you stop it (Ctrl+C). Saves checkpoints every SAVE_EVERY_EPISODES
episodes and writes progress.txt + best_result.txt.

Run:
    python main.py
"""

import datetime
import shutil
from pathlib import Path

import torch

from wrappers import make_env
from mario import Mario
from loggers import MetricLogger

# Save model + plots every N episodes
SAVE_EVERY_EPISODES = 20
RECORD_EVERY_EPISODES = 20

LATEST_DIR = Path("checkpoints") / "latest"


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


def save_checkpoint(mario, logger, episode, best=False):
    mario.save("mario_latest.chkpt", episode=episode)
    if best:
        shutil.copy2(mario.save_dir / "mario_latest.chkpt", mario.save_dir / "mario_best.chkpt")
        shutil.copy2(mario.save_dir / "mario_best.chkpt", LATEST_DIR / "mario_best.chkpt")
    shutil.copy2(mario.save_dir / "mario_latest.chkpt", LATEST_DIR / "mario_latest.chkpt")
    logger.mirror_to_latest(LATEST_DIR)


def main():
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print("Training runs until you press Ctrl+C.")
    print(f"Checkpoints saved every {SAVE_EVERY_EPISODES} episodes to checkpoints/latest/")

    env = make_env()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    logger = MetricLogger(save_dir)

    episode = 0

    try:
        while True:
            state = reset_env(env)
            ep_max_x = 0
            flag_get = False

            while True:
                action = mario.act(state)
                next_state, reward, done, info = step_env(env, action)
                mario.cache(state, next_state, action, reward, done)
                q, loss = mario.learn()
                logger.log_step(reward, loss, q)

                ep_max_x = max(ep_max_x, info.get("x_pos", 0))
                flag_get = flag_get or info.get("flag_get", False)

                state = next_state
                if done or info.get("flag_get", False):
                    break

            logger.log_episode()
            ep_reward = logger.ep_rewards[-1]
            ep_length = logger.ep_lengths[-1]
            ep_score = info.get("score", 0)

            logger.log_episode_progress(
                episode, ep_reward, ep_length, ep_max_x, flag_get, ep_score
            )
            is_best = logger.update_best(
                episode, ep_reward, ep_length, ep_max_x, flag_get, ep_score
            )

            if episode % RECORD_EVERY_EPISODES == 0:
                logger.record(episode=episode, epsilon=mario.exploration_rate, step=mario.curr_step)

            if episode % SAVE_EVERY_EPISODES == 0 or is_best:
                save_checkpoint(mario, logger, episode, best=is_best)

            episode += 1

    except KeyboardInterrupt:
        print("\nTraining stopped by user. Saving final checkpoint...")
        logger.record(episode=episode, epsilon=mario.exploration_rate, step=mario.curr_step)
        save_checkpoint(mario, logger, episode)
        print(f"Saved to {save_dir} and {LATEST_DIR}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
