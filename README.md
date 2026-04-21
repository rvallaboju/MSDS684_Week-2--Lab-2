# Week 2 Lab: Dynamic Programming

## Overview

This comprehensive Jupyter notebook implements and demonstrates classical **Dynamic Programming (DP)** algorithms for solving Markov Decision Processes (MDPs). The lab covers both the theory and practical implementation of core DP algorithms including policy evaluation, policy improvement, policy iteration, and value iteration.

---

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone or download this repository:
```bash
git clone https://github.com/rvallaboju/MSDS684_Week-2--Lab-2.git
cd MSDS684_Week-2--Lab-2
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

---

## Key Learning Objectives

- ✅ Implement classical DP algorithms for known MDPs
- ✅ Understand the **Policy Improvement Theorem** and **Generalized Policy Iteration (GPI)** framework
- ✅ Analyze convergence properties and computational efficiency
- ✅ Recognize when DP is applicable versus other reinforcement learning methods
- ✅ Navigate Gymnasium's environment structure for model-based planning
- ✅ Work with discrete state/action spaces using tabular representations

---

## Topics Covered

### 1. **Policy Evaluation (Prediction Problem)**
   - Synchronous vs. in-place evaluation methods
   - Convergence criteria and delta tracking
   - Bootstrap-based value updates

### 2. **Policy Improvement Theorem**
   - Greedy policy extraction from value functions
   - Guaranteed policy improvement guarantee
   - Deterministic policy extraction

### 3. **Policy Iteration**
   - Combined policy evaluation and improvement
   - Convergence to optimal policy
   - Tracking evaluation sweeps per outer iteration

### 4. **Value Iteration**
   - Direct value function optimization
   - Bellman optimality equation updates
   - Typically faster convergence than policy iteration

### 5. **Generalized Policy Iteration Framework**
   - Interleaving evaluation and improvement
   - GPI convergence properties
   - Applications to various DP algorithms

### 6. **Environment Integration**
   - Custom GridWorld implementation with Gymnasium API
   - Deterministic and stochastic transitions
   - FrozenLake-v1 environment from Gymnasium
   - Accessing environment dynamics via `env.unwrapped.P`

---

## Code Structure

### Section 0: Setup and Imports
- Import required libraries: NumPy, Matplotlib, Gymnasium
- Configure reproducibility with random seed

### Section 1: Custom GridWorld Environment
**`GridWorld` Class** - A configurable 2D grid environment supporting:
- Configurable grid size, start position, goal position, and obstacles
- Deterministic or stochastic (slippery) transitions
- Custom reward structure (step rewards and goal rewards)
- Full Gymnasium API compliance
- Efficient transition model storage as `P[s][a]`

**Visualization Functions:**
- `plot_value_function()` - Heatmap visualization of value estimates
- `plot_policy()` - Arrow-based visualization of deterministic policies

### Section 2: Dynamic Programming Algorithms

#### 2.1 Policy Evaluation
```python
policy_evaluation_sync()      # Synchronous (two-array method)
policy_evaluation_inplace()   # In-place (single-array method)
```