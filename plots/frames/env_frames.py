import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py

def save_frame(frame, name):
    plt.figure(figsize=(4, 4))  # adjust size for publication
    plt.axis('off')  # remove axes
    plt.imshow(frame)  # for MinAtar, may need: state[:,:,0] for a single channel
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=300)
    plt.show()

def atari_frame(actions):
    env = gym.make("ALE/Breakout-v5")
    obs = env.reset()
    for action in actions:
        obs, _, _, _, _ = env.step(action)

    save_frame(obs, "atari_breakout")


if __name__ == "__main__":
    actions = [1, 3, 3, 0, 0, 0, 0]
    atari_frame(actions)