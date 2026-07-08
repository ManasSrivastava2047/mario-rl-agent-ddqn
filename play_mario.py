"""
Controls:
    Right Arrow  -> walk/run right
    Left Arrow   -> walk/run left
    Down Arrow   -> crouch / enter pipe
    Space        -> A button (jump)
    Left Shift   -> B button (run / fireball)
    R            -> reset the level
    Esc / close  -> quit
    
"""

import sys

import numpy as np
import pygame
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# ---------------------------------------------------------------------------
# Action set: every combo a human might actually press at once.
# Each entry is a list of NES button names, matching nes_py's convention.
# ---------------------------------------------------------------------------
ACTIONS = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
    ["up"],
]

# Precompute a lookup from a frozenset of buttons -> action index, so we can
# turn "which keys are currently held" into a single discrete action.
_ACTION_LOOKUP = {}
for idx, combo in enumerate(ACTIONS):
    key = frozenset(b for b in combo if b != "NOOP")
    _ACTION_LOOKUP[key] = idx
NOOP_INDEX = _ACTION_LOOKUP[frozenset()]


def keys_to_action(keys_pressed) -> int:
    """Map the set of currently-held pygame keys to an action index."""
    buttons = set()
    if keys_pressed[pygame.K_RIGHT]:
        buttons.add("right")
    if keys_pressed[pygame.K_LEFT]:
        buttons.add("left")
    if keys_pressed[pygame.K_DOWN]:
        buttons.add("down")
    if keys_pressed[pygame.K_UP]:
        buttons.add("up")
    if keys_pressed[pygame.K_SPACE]:
        buttons.add("A")
    if keys_pressed[pygame.K_LSHIFT] or keys_pressed[pygame.K_RSHIFT]:
        buttons.add("B")

    frozen = frozenset(buttons)
    if frozen in _ACTION_LOOKUP:
        return _ACTION_LOOKUP[frozen]

    # Fall back: drop "down"/"up" (rarely combined with anything else) and
    # retry, so e.g. holding Down + Right still does something reasonable.
    for drop in ("up", "down"):
        if drop in frozen:
            retry = frozen - {drop}
            if retry in _ACTION_LOOKUP:
                return _ACTION_LOOKUP[retry]
    return NOOP_INDEX


# ---------------------------------------------------------------------------
# gym/gymnasium API compatibility helpers
# (requirements.txt pins gym==0.25.2, so we take the "old" step/reset shape,
#  but these helpers keep the script working if that ever changes)
# ---------------------------------------------------------------------------
def make_env():
    if gym.__version__ < "0.26":
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True
        )
    return JoypadSpace(env, ACTIONS)


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
    """Grab the current RGB frame as an (H, W, 3) uint8 array."""
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    return np.asarray(frame)


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------
SCALE = 3          # upscale factor so the NES resolution isn't tiny on screen
FPS = 60


def main():
    env = make_env()
    obs = reset_env(env)

    frame = get_frame(env)
    h, w = frame.shape[0], frame.shape[1]

    pygame.init()
    pygame.display.set_caption("Super Mario Bros - Play It Yourself")
    screen = pygame.display.set_mode((w * SCALE, h * SCALE + 40))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    info = {"coins": 0, "life": 2, "score": 0, "time": 400, "world": 1, "stage": 1}
    done = False

    print("Controls: Right/Left/Down/Up arrows, Space=Jump, Shift=Run, R=Reset, Esc=Quit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs = reset_env(env)
                    done = False

        keys = pygame.key.get_pressed()

        if not done:
            action = keys_to_action(keys)
            obs, reward, done, info = step_env(env, action)

        frame = get_frame(env)
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, (w * SCALE, h * SCALE))
        screen.blit(surface, (0, 0))

        # HUD bar
        pygame.draw.rect(screen, (0, 0, 0), (0, h * SCALE, w * SCALE, 40))
        status = (
            f"World {info.get('world', 1)}-{info.get('stage', 1)}   "
            f"Coins: {info.get('coins', 0)}   "
            f"Score: {info.get('score', 0)}   "
            f"Time: {info.get('time', 0)}   "
            f"Lives: {info.get('life', 0)}"
        )
        if done:
            status += "   -- GAME OVER / LEVEL END, press R to reset --"
        text_surface = font.render(status, True, (255, 255, 255))
        screen.blit(text_surface, (8, h * SCALE + 10))

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()