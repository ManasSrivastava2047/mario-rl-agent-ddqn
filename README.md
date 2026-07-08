# Super Mario Bros RL Agent

This project trains a **Double Deep Q-Network (DDQN)** agent to play **Super Mario Bros 1-1** using PyTorch. It includes training, checkpoint loading for evaluation, manual gameplay, and simple logging so you can track how far the agent is getting over time.

The current setup is designed for a small action space and long-running training on a local machine, especially a Windows setup.

## What the project does

- Trains a Mario agent with **epsilon-greedy exploration**
- Uses **experience replay** and separate **online/target networks**
- Preprocesses frames to **84x84 grayscale stacked over 4 frames**
- Saves **checkpoints, logs, and best-run summaries**
- Lets you **watch a trained model play**
- Lets you **play manually** with keyboard controls

## Project files

| File | Purpose |
|------|---------|
| `main.py` | Main training script |
| `mario.py` | DDQN agent logic: act, replay buffer, learn, save/load |
| `network.py` | CNN used to estimate Q-values |
| `wrappers.py` | Mario environment setup and frame preprocessing |
| `loggers.py` | Episode logs, moving averages, best-result tracking |
| `run_trained.py` | Load a checkpoint and run the trained agent |
| `play_mario.py` | Play Mario manually |
| `requirements.txt` | Python dependencies |

## Setup

Recommended on Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. Training can run on CPU, and will use CUDA automatically if available.

## Training

Start training with:

```bash
python main.py
```

Training behavior:

- Runs continuously until you stop it with `Ctrl+C`
- Saves checkpoints every `20` episodes
- Writes per-episode progress and best-run summaries
- Mirrors the newest important files into `checkpoints/latest/`
- Saves one final checkpoint when training is interrupted

## Training outputs

Every training run creates a timestamped folder inside `checkpoints/`, for example:

```text
checkpoints/2026-07-07T23-09-02/
```

Typical generated files:

| File | Description |
|------|-------------|
| `progress.txt` | One line per episode with reward, length, max X, flag, score, and timestamp |
| `best_result.txt` | Best result seen so far in that run |
| `log` | Moving-average stats written every 20 episodes |
| `mario_latest.chkpt` | Most recent checkpoint |
| `mario_best.chkpt` | Best-performing checkpoint |
| `reward_plot.jpg` | Reward moving-average plot |
| `length_plot.jpg` | Episode length moving-average plot |
| `loss_plot.jpg` | Loss moving-average plot |
| `q_plot.jpg` | Q-value moving-average plot |

The folder `checkpoints/latest/` is used as a convenient mirror of the newest run artifacts.

## Training progress so far

Based on the current logs in `checkpoints/latest/`:

- Best recorded episode: `76`
- Best reward: `2893.0`
- Best episode length: `1141`
- Furthest distance reached (`Max X`): `3161`
- Level cleared: `True`
- Best score: `1700`

The current log also shows training has reached at least:

- Episode `315`
- Step `66267`
- Epsilon `0.984`

So the pipeline is working, checkpoints are being saved correctly, and the agent has already managed to clear the level at least once.

## Watching the trained agent

Run a saved model with:

```bash
python run_trained.py
python run_trained.py --best
python run_trained.py --episodes 5
python run_trained.py --checkpoint checkpoints/latest/mario_best.chkpt
```

Notes:

- By default it loads `checkpoints/latest/mario_latest.chkpt`
- `--best` loads `checkpoints/latest/mario_best.chkpt`
- Evaluation runs with exploration turned off, so the agent only exploits what it has learned
- A pygame window opens and shows the current game screen plus status text

## Playing manually

```bash
python play_mario.py
```

Controls:

| Key | Action |
|-----|--------|
| Right / Left arrows | Move |
| Space | Jump |
| Shift | Run / fireball |
| Down / Up arrows | Crouch / enter pipe |
| `R` | Reset level |
| `Esc` | Quit |

## RL setup

The current agent uses:

- **Environment:** `SuperMarioBros-1-1-v0`
- **Action space:** `["right"]` and `["right", "A"]`
- **Frame skip:** `4`
- **Frame stack:** `4`
- **Replay buffer:** `LazyMemmapStorage(20000)` on CPU
- **Batch size:** `32`
- **Gamma:** `0.9`
- **Burn-in:** `10000` steps
- **Learn frequency:** every `3` steps
- **Target sync:** every `10000` steps

## Notes for GitHub upload

This repo generates large and changing files during training, especially:

- checkpoints
- plots
- logs
- Python cache files
- local virtual environment files

Those should not be committed. The `.gitignore` has been set up for that.