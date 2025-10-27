Function Approximation in Reinforcement Learning - Assignment 2
==============================================================

Overview
--------
This project implements and compares three RL algorithms (DQN, REINFORCE, A2C) using neural networks as function approximators on the CartPole-v1 environment. All algorithms share the same architecture and are assessed over 50 runs each to ensure robust statistical analysis of their mean and variance.

Project Structure
-----------------
1. Main Notebook (RL-HW2.ipynb)
   - Environment setup: CartPole-v1 from Gymnasium.
   - Algorithms: 
        - DQN (with replay buffer and target network)
        - REINFORCE (policy gradient)
        - A2C (actor-critic)
   - Common network: 4 input → 128 → 128 → 2 output (or 1 for critic)
   - Experiment loop: Each algorithm runs 50 seeds, results are saved.
   - Analysis and visualization: Plots mean ± stdev, saves all relevant data.

2. Results Directory (results/)
   - learning_curves_comparison.png      <- Main learning curve plot
   - best_hyperparameters.json           <- Hyperparameters used
   - statistical_summary.json            <- Table of mean ± stdev, convergence info
   - dqn_run_0.pkl ... a2c_run_49.pkl    <- Rewards for each run per algorithm

Installation & Usage
--------------------
1. Open RL-HW2.ipynb in Google Colab.
2. Run all cells. Dependencies will be installed automatically.
3. All outputs/plots will appear at the end and files will save to results/.

Key Hyperparameters
-------------------
- DQN: lr=0.01, batch=16, buffer=2000, gamma=0.99
- REINFORCE: lr=0.003, gamma=0.99
- A2C: lr=0.003, gamma=0.99

Experiment Workflow
-------------------
- Each algorithm is evaluated over 50 runs for statistical stability.
- Convergence condition: average reward ≥475 over last 100 episodes, or max episodes reached.
- Output figures compare learning speed and stability.

Outputs
-------
- Comparative learning curve (.png)
- Per-run reward logs (.pkl/x3/50)
- JSON logs for hyperparameters and final statistics

Notes
-----
- All code and results reproducible: just run the notebook from top to bottom.
- Random seeds are controlled per run.
- This setup is tested to finish within Google Colab’s free runtime on CPU.

