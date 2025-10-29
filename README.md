# DQN, REINFORCE, A2C on CartPole-v1

## Overview
This repository contains the implementation of three reinforcement learning (RL) algorithms—DQN (Deep Q-Learning), REINFORCE (Policy Gradient), and A2C (Advantage Actor-Critic)—applied to the CartPole-v1 environment. All models share the same neural network architecture and were evaluated across 100 independent runs to analyze both mean performance and variance.

## Environment
- **Environment**: CartPole-v1 (Gymnasium)
- **State Dimension**: 4 (position, velocity, angle, angular velocity)
- **Action Space**: Discrete(2) → {Left, Right}
- **Maximum Episode Length**: Until convergence or 10,000 episodes

## Running the Code
1. **Clone this repository and navigate to the project folder**:
    ```bash
    git clone <repo_link>
    cd <project_folder>
    ```

2. **Install dependencies**:
    Ensure that Python 3.9+ is installed, then use pip to install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Experiment**:
    The code trains three models (DQN, REINFORCE, A2C) on CartPole-v1 and saves the results (rewards per episode) in `results/` for each algorithm.

    To run the experiments:
    ```bash
    python run_experiments.py --algo all --runs 100
    ```

4. **Aggregate the Results**:
    After the experiments are completed, aggregate the results and plot the learning curves (mean ± standard deviation) for each algorithm:
    ```bash
    python analyze_and_plot.py
    ```

5. **Generate the Report**:
    Finally, generate a LaTeX-based PDF report that includes the learning curve plot and the analysis:
    ```bash
    python make_report.py
    ```

## Key Hyperparameters
- **DQN**: 
    - Learning rate: 1e-3
    - Batch size: 64
    - Replay buffer size: 50,000
    - Target update frequency: Every 500 steps
    - Epsilon decay: 1.0 → 0.05 over 20,000 steps

- **REINFORCE**: 
    - Learning rate: 1e-3
    - Return normalization: Applied

- **A2C**: 
    - Learning rate: 1e-3
    - Value coefficient: 0.5
    - Entropy coefficient: 0.01

## Results
- **Stop Condition**: The algorithm stops when the moving average of the last 100 episodes exceeds a reward of 475 or after 10,000 episodes.
- **Data Output**: Results are saved as `results/<algo>/seed_XXX.npz` (100 independent runs for each algorithm).
- **Learning Curve Plot**: Saved as `figures/comparison.png` showing the mean ± std of the reward for each algorithm.
- **Final Report**: Generated as `report/RL_HW2_report.pdf`.

## Notes
- All models use the same architecture:
    - **Input**: 4-dimensional state
    - **Hidden Layers**: 2 layers of 128 units with ReLU activation
    - **Output**: DQN (Q-values), REINFORCE (action probabilities), A2C (action probabilities and state values)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
