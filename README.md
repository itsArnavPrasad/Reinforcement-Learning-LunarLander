# LunarLander-v2 Reinforcement Learning Agents

This repository contains implementations of **Random Agent**, **DQN from scratch**, and **PPO (Stable-Baselines3)** for the **LunarLander-v2** environment in Gymnasium. The project demonstrates reinforcement learning concepts including policy gradient methods, deep Q-learning, and generalized advantage estimation.

---

## Project Overview

* Implemented a **baseline Random Agent** to measure initial environment performance.
* Built **Deep Q-Network (DQN) from scratch** using PyTorch with experience replay and target networks.
* Leveraged **PPO (Proximal Policy Optimization)** from Stable-Baselines3 with **MlpPolicy** and **GAE** for faster convergence and higher sample efficiency.
* Tracked **quantitative metrics** such as reward, landing success rate, and stability (standard deviation) to evaluate agent performance.
* Rendered gameplay videos for qualitative evaluation.

---

## Results

### **Random Agent (Baseline)**

| Metric                 | Value           |
| ---------------------- | --------------- |
| Mean Reward            | -189.77         |
| Std Reward             | 106.59          |
| Min Reward             | -509.46         |
| Max Reward             | 12.60           |
| Average Episode Length | 94.23 timesteps |
| Std Episode Length     | 19.44           |

---

### **DQN from Scratch (PyTorch)**

| Episode | Avg Score | Landing Rate | Stability (Std) |
| ------- | --------- | ------------ | --------------- |
| 100     | -162.49   | 0.0%         | 100.61          |
| 200     | -87.34    | 0.0%         | 82.28           |
| 300     | -33.99    | 1.0%         | 91.02           |
| 400     | -8.32     | 2.0%         | 53.28           |
| 500     | 18.83     | 7.0%         | 87.12           |
| 600     | 131.87    | 34.0%        | 104.58          |
| 674     | 200.05    | 65.0%        | 83.96           |

**Environment solved in 574 episodes** with final average score **200.05** and landing success rate **65%**.

---

### **PPO (Stable-Baselines3)**

| Episode | Avg Score | Landing Rate | Stability (Std) |
| ------- | --------- | ------------ | --------------- |
| 100     | 68.34     | 32.0%        | 154.49          |
| 200     | 191.44    | 70.0%        | 96.11           |

**Environment solved in 167 episodes (≈3.4× faster than DQN)** with final average score **200.15** and landing success rate **74%**.

---

## Implementation Details

### Random Agent

* Uses **random action selection**.
* Serves as a **baseline** to compare RL agent performance.

```python
num_episodes = 100
total_rewards, episode_lengths = [], []
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward, step_count = 0, 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        episode_reward += reward
        step_count += 1
    total_rewards.append(episode_reward)
    episode_lengths.append(step_count)
```

---

### DQN (Deep Q-Network)

* **Implemented from scratch** using PyTorch.
* Includes **experience replay**, **target networks**, and **epsilon-greedy exploration**.
* Tracks **average reward**, **landing success**, and **stability** over episodes.

Key features:

* Fully connected network (`64-64`) layers for Q-function approximation.
* Soft update of target network for stable learning.
* Success threshold: **score ≥ 200**.

---

### PPO (Stable-Baselines3)

* **Policy Gradient method** using **Proximal Policy Optimization**.
* **MlpPolicy** (fully connected neural network) maps states to action probabilities.
* **GAE (Generalized Advantage Estimation)** used for stable advantage computation.
* Training every episode with reward logging for reproducibility.

```python
from stable_baselines3 import PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)
```

* Sample-efficient: Solved environment in **167 episodes** compared to **574 episodes** for DQN.

---

## Video Rendering

* Gameplay videos are saved as `ppo_lunar.gif` for qualitative evaluation:

```python
import imageio
frames = []
state, _ = env.reset()
done = False
while not done:
    frame = env.render()
    frames.append(frame)
    action, _ = model.predict(state, deterministic=True)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
imageio.mimsave('ppo_lunar.gif', frames, fps=30)
```

---

## Key Takeaways

* **Random agent** establishes a low-performance baseline.
* **DQN from scratch** demonstrates deep Q-learning with replay buffer and target networks.
* **PPO** shows **faster convergence**, higher sample efficiency, and improved landing success rate.
* Using **MlpPolicy + GAE** significantly enhances stability and reproducibility.

---

## References

* [OpenAI Gym: LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)
* [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
* [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)