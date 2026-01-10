import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
import numpy as np

import ale_py

def save_frame(frame, name):
    plt.figure(figsize=(4, 4))  # adjust size for publication
    plt.axis('off')  # remove axes
    plt.imshow(frame)  # for MinAtar, may need: state[:,:,0] for a single channel
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)

def atari_frame(actions):
    env = gym.make("ALE/Breakout-v5")
    obs = env.reset()
    for action in actions:
        obs, _, _, _, _ = env.step(action)

    save_frame(obs, "atari_breakout")

# Atari Breakout action meanings (minimal action set)
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3

def human_playable_breakout():
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="rgb_array",
        full_action_space=False
    )

    obs, _ = env.reset()

    pygame.init()
    size = (600, 800)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Human Playable Breakout")
    clock = pygame.time.Clock()

    running = True
    img_idx = 0
    observations = []

    while running:
        observations.append(obs)
        # Handle input
        action = NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False
        elif keys[pygame.K_a]:
            action = LEFT
        elif keys[pygame.K_d]:
            action = RIGHT
        elif keys[pygame.K_SPACE]:
            action = FIRE
        elif keys[pygame.K_s]:
            img_idx +=1
            save_frame(obs, f"game_frames/atari_breakout_{img_idx}")

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)

        # Render frame
        frame = np.transpose(obs, (1, 0, 2))  # rotate for pygame
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(surf, size), (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(10)  # control speed

    env.close()
    pygame.quit()


if __name__ == "__main__":
    human_playable_breakout()
