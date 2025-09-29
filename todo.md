
# **Phase 0: Project Setup & Baseline (Notebook Version)**

**Goal:** Set up the environment and establish a baseline agent.

**Subtasks:**

1. Install/import libraries in a notebook cell:

   ```python
   import gymnasium as gym
   import torch, torch.nn as nn
   import numpy as np
   import matplotlib.pyplot as plt
   from stable_baselines3 import DQN, PPO
   import wandb  # optional for logging
   ```
2. Explore **LunarLander-v2** environment:

   * Print `env.observation_space` and `env.action_space`.
   * Run `env.reset()` and take a few random actions to see reward structure.
3. Implement and run **random agent** for 10–20 episodes.
4. Plot **random agent reward distribution** inline.

**Deliverables:**

* Baseline reward plot.
* Notebook section titled “Baseline Random Agent.”

---

# **Phase 1: Deep Q-Network (DQN) Implementation (Notebook Version)**

**Goal:** Build and train a classical RL agent entirely in the notebook.

**Subtasks:**

1. Define DQN network as a class in a code cell.
2. Implement replay buffer and epsilon-greedy action selection inline.
3. Train DQN for ~1M timesteps (or fewer for demo).
4. Log rewards per episode using a cell: `plt.plot(...)` or `wandb.log()`.
5. Visualize:

   * Reward curve (inline plot).
   * Histogram of final rewards.

**Deliverables:**

* Inline plots.
* Trained DQN agent stored as notebook variable or saved with `torch.save()`.

---

# **Phase 2: Proximal Policy Optimization (PPO) Implementation (Notebook Version)**

**Goal:** Train PPO agent in the same notebook for comparison.

**Subtasks:**

1. Use `stable-baselines3` or define PPO network inline.
2. Train PPO for ~500k timesteps.
3. Log training metrics (reward, loss) inline.
4. Visualize learning curves in notebook cells.
5. Compare speed of convergence with DQN visually.

**Deliverables:**

* PPO reward curves plotted inline.
* Trained PPO agent stored in notebook variable or saved.

---

# **Phase 3: Hyperparameter Tuning & Experiments (Notebook Version)**

**Goal:** Include metric-driven improvements all in notebook.

**Subtasks:**

1. Define hyperparameter variations in cells (learning rate, batch size, clipping epsilon, epsilon decay).
2. Run 2–3 seeds for reproducibility.
3. Record metrics in notebook variables (arrays/dicts).
4. Plot comparison graphs (bar charts, reward curves) inline.

**Deliverables:**

* Inline hyperparameter comparison table/plot.
* Demonstrates experimentation rigor.

---

# **Phase 4: Benchmarking & Analysis (Notebook Version)**

**Goal:** Directly compare DQN vs PPO.

**Subtasks:**

1. Compare **sample efficiency** (timesteps to reach 200 reward).
2. Compare **stability** (variance across seeds).
3. Plot side-by-side learning curves inline.
4. Plot reward distributions for DQN vs PPO.
5. Write **markdown cells** for analysis (key takeaways).

**Deliverables:**

* Notebook contains **plots + text analysis**.
* Shows recruiter-friendly metric comparison.

---

# **Phase 5: Visualization & Demonstration (Notebook Version)**

**Goal:** Showcase agent performance in the notebook itself.

**Subtasks:**

1. Use `env.render(mode='rgb_array')` to generate frames.
2. Convert frames to GIF or inline animation using `matplotlib.animation` or `imageio`.
3. Plot agent trajectories (x, y, velocity) inline.
4. Annotate successes vs failures using notebook cells.

**Deliverables:**

* Inline GIF or animation.
* Annotated trajectory plots.

---

# **Phase 6: Notebook Documentation & Resume Readiness**

**Goal:** Make notebook polished, readable, and professional.

**Subtasks:**

1. Add **markdown sections**:

   * Introduction, objectives, methods, results, key takeaways.
2. Include inline **plots, tables, GIFs** for visual appeal.
3. Save notebook as `.ipynb` and optionally as `.html` for sharing.
4. Optional: Push to GitHub with folder for results (plots, GIFs).

**Deliverables:**

* Single notebook that is **self-contained, polished, and recruiter-ready**.
* Demonstrates both **RL expertise and professionalism**.
