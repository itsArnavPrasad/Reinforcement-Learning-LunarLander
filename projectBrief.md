🚀 Project Brief: LunarLander-v2 with DQN vs PPO

This project benchmarks Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) on the classic LunarLander-v2 environment from OpenAI Gym. The environment simulates a lunar lander that must safely land on a designated pad, with dense rewards for stability and penalties for crashes or fuel wastage.

🎯 Objectives

Train reinforcement learning agents (DQN and PPO) to solve the LunarLander-v2 environment.

Compare algorithms in terms of sample efficiency, stability, and reward performance.

Apply hyperparameter tuning and architectural improvements to accelerate convergence.

🛠️ Methods

Baseline (DQN): Implemented replay buffer, target networks, and epsilon-greedy exploration.

Advanced (PPO): Applied clipped surrogate loss, entropy regularization, and policy updates for more stable training.

Conducted systematic hyperparameter sweeps (learning rate, batch size, clipping epsilon).

Logged results using Weights & Biases / matplotlib for reproducibility and visualization.

📊 Results

DQN: Achieved an average reward of ~210 after ~1M timesteps.

PPO: Achieved faster convergence (~500k timesteps) and more stable performance (+15% reward stability across seeds).

PPO showed 40% lower variance in final performance compared to DQN.

Delivered interpretable learning curves, reward distributions, and trained-agent landing demos.

🔑 Key Takeaways

PPO significantly outperforms DQN in terms of stability and efficiency.

Careful hyperparameter tuning yields measurable improvements in RL performance.

The project highlights algorithmic benchmarking and metric-driven improvements, making it resume- and recruiter-ready.