# Adaptive Urban Traffic Control with Reinforcement Learning

A bachelor's thesis project that explores the use of deep reinforcement learning to manage traffic lights in simulated urban environments. The project demonstrates how a centralized learning agent can optimize traffic flow, outperforming traditional fixed-timing systems.

---

## Overview

Traditional traffic light systems are rigid and fail to adapt to real-time traffic fluctuations. This project introduces a reinforcement learning approach using the **Proximal Policy Optimization (PPO)** algorithm to dynamically manage multiple intersections based on live traffic conditions, simulated with **SUMO (Simulation of Urban MObility)**.

---

## Objectives

- Simulate realistic urban traffic networks.
- Develop a custom environment compatible with Gymnasium.
- Train a centralized PPO-based agent to control all traffic lights.
- Minimize average travel time and waiting time.
- Compare the learned policy to traditional fixed-timing systems.

---

## Technologies

- **Python**, **OpenAI Gym / Gymnasium**
- **SUMO** (Simulation of Urban MObility)
- **TraCI** (Traffic Control Interface)
- **Stable-Baselines3 / sb3-plus**
- **TensorBoard** for training monitoring
- **OpenStreetMap**, **NetEdit** for map creation

---

## Methodology

1. **Urban Network Design**: Built multiple maps (small/medium cities) based on real road layouts.
2. **Traffic Generation**: Simulated cars, pedestrians, and trucks using `randomTrips.py` and custom scripts.
3. **Environment Design**: Defined action, observation, and reward spaces tailored to traffic signal control.
4. **Training**:
   - Phase 1: Overfit on a fixed traffic pattern for system familiarization.
   - Phase 2: Generalized with dynamic traffic episodes.
5. **Evaluation**: Compared the RL agent to fixed-timing control using key performance metrics.

---

## Results

- **~2.8% reduction** in average travel time.
- **Up to 41% improvement** in high-congestion scenarios.
- Robust behavior in dynamic traffic, with limited generalization in extreme outliers.

---

## Limitations

- No simulation of driver unpredictability or weather conditions.
- Topology changes may degrade performance.
- Uses teleportation in persistent gridlocks (unrealistic in real-world systems).

---

## Future Work

- Incorporate additional reward metrics (e.g. emissions, fuel usage).
- Explore multi-agent or hybrid control strategies.
- Extend adaptability for real-time infrastructure and event-based updates.

---
