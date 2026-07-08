"""
loggers.py

MetricLogger: tracks per-episode reward/length/loss/Q-value, writes a
plain-text log file, and saves moving-average plots (reward/length/loss/Q).
Also maintains progress.txt (every episode) and best_result.txt (new records).
"""

import time
import datetime
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.save_log = save_dir / "log"
        self.progress_file = save_dir / "progress.txt"
        self.best_result_file = save_dir / "best_result.txt"

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )

        with open(self.progress_file, "w") as f:
            f.write(
                f"{'Episode':>8}{'Reward':>10}{'Length':>8}{'MaxX':>8}"
                f"{'Flag':>6}{'Score':>10}{'Time':>20}\n"
            )

        self.best_result = {
            "episode": -1,
            "reward": float("-inf"),
            "length": 0,
            "max_x": 0,
            "flag_get": False,
            "score": 0,
        }
        self._write_best_result()
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages (updated on every call to record())
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current-episode running totals
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """Mark end of episode."""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

    def log_episode_progress(self, episode, reward, length, max_x, flag_get, score):
        """Append one line to progress.txt after each episode."""
        with open(self.progress_file, "a") as f:
            f.write(
                f"{episode:8d}{reward:10.1f}{length:8d}{max_x:8d}"
                f"{int(flag_get):6d}{score:10d}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

    def _is_better(self, reward, max_x, flag_get):
        if flag_get and not self.best_result["flag_get"]:
            return True
        if flag_get and self.best_result["flag_get"]:
            return reward > self.best_result["reward"]
        if max_x > self.best_result["max_x"]:
            return True
        if max_x == self.best_result["max_x"] and reward > self.best_result["reward"]:
            return True
        return False

    def update_best(self, episode, reward, length, max_x, flag_get, score):
        """Update best_result.txt if this episode beat the previous best."""
        if not self._is_better(reward, max_x, flag_get):
            return False

        self.best_result = {
            "episode": episode,
            "reward": reward,
            "length": length,
            "max_x": max_x,
            "flag_get": flag_get,
            "score": score,
        }
        self._write_best_result()
        print(
            f"New best! Episode {episode} - "
            f"reward {reward:.1f}, max_x {max_x}, flag {flag_get}, score {score}"
        )
        return True

    def _write_best_result(self):
        br = self.best_result
        with open(self.best_result_file, "w") as f:
            f.write("Best result so far\n")
            f.write("==================\n")
            f.write(f"Episode:   {br['episode']}\n")
            f.write(f"Reward:    {br['reward']:.1f}\n")
            f.write(f"Length:    {br['length']}\n")
            f.write(f"Max X:     {br['max_x']}\n")
            f.write(f"Flag:      {br['flag_get']}\n")
            f.write(f"Score:     {br['score']}\n")
            f.write(f"Updated:   {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\n")

    def mirror_to_latest(self, latest_dir: Path):
        """Copy progress/best/log files to checkpoints/latest for easy access."""
        latest_dir.mkdir(parents=True, exist_ok=True)
        for name in ("progress.txt", "best_result.txt", "log"):
            src = self.save_dir / name
            if src.exists():
                shutil.copy2(src, latest_dir / name)