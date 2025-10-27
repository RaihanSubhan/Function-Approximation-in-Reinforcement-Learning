DQN, REINFORCE, A2C on CartPole-v1
==================================

Overview
--------
This project implements and compares three RL algorithms—DQN (value-based), REINFORCE (policy-based),
and A2C (actor–critic)—using the SAME neural network architecture. Each algorithm is trained until
convergence and evaluated over 100 independent runs. The final deliverables are a mean learning-curve
plot with ±1 standard deviation and a short report.

Environment
-----------
• Task: CartPole-v1 (Gymnasium)
• Discount factor: gamma = 0.99
• Optimizer: Adam
• Network architecture: identical across all algorithms (same hidden layers/units/activations)

Stop Condition (per run)
------------------------
Training for a run stops when:
• the average reward >= 475 over the most recent 100 consecutive episodes, OR
• the run reaches 10,000 episodes.

Folder Structure (expected)
---------------------------
.
├─ RL-HW2.ipynb            # notebook you ran
├─ results/
│  ├─ dqn/                 # per-run reward arrays (e.g., seed_001.npz, ...)
│  ├─ reinforce/
│  └─ a2c/
├─ figures/
│  └─ comparison.png       # mean ± std plot across 100 runs (all three algorithms)

Dependencies
------------
Python 3.9+ recommended. Install:
  pip install torch gymnasium numpy matplotlib tqdm

(If your notebook already installs packages, this step can be skipped.)

How to Reproduce
----------------
1) Open RL-HW2.ipynb in Jupyter/Colab and run all cells for each algorithm (DQN, REINFORCE, A2C).
   - Ensure the SAME neural network architecture is used for all three (same hidden layers/units).
   - Use different random seeds across the 100 runs.
   - Save per-episode total rewards for each run into:
       results/dqn/seed_XXX.npz
       results/reinforce/seed_XXX.npz
       results/a2c/seed_XXX.npz
     (Each .npz should contain an array named "rewards".)

2) Aggregate and Plot
   - Compute the mean and standard deviation of rewards across the 100 runs for each algorithm.
   - Plot the mean curve and shade ±1 std. Save plot to:
       figures/comparison.png

3) Report
   - Use the provided LaTeX (Overleaf) template.
   - Insert figures/comparison.png into the report.
   - Briefly document: algorithms, shared network architecture, hyperparameter tuning process,
     learning speed comparison, variance/stability, and final convergence behavior.

Notes on Hyperparameters
------------------------
• Learning rates may be tuned per algorithm (document the grid and selection rule).
• DQN typically needs: replay buffer, batch size, target-network update interval, epsilon schedule.
• REINFORCE uses Monte Carlo returns; variance can be reduced by normalizing returns.
• A2C uses a value baseline and a small entropy bonus to stabilize exploration.

Reproducibility
---------------
• Use a different seed for each independent run (100 runs).
• Keep the training/stop criteria identical for all three algorithms.
• Confirm that only the algorithm logic (not the architecture) differs between methods.

Outputs to Check Before Submitting
----------------------------------
1) results/<algo>/seed_*.npz  (100 files per algorithm)
2) figures/comparison.png      (one figure with mean ± std for DQN, REINFORCE, A2C)
3) Final PDF report (compiled from the Overleaf LaTeX below)
