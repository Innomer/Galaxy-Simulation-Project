# 🌌 Galaxy Collision Simulation (CUDA)

This project simulates the gravitational interaction between multiple galaxies in 3D space using **CUDA** for high-performance parallel computation. Each galaxy contains a supermassive black hole and thousands of orbiting stars, forming realistic spiral structures that interact over time.

## Features

- **N-body simulation** of gravitational forces using Newtonian physics
- **CUDA acceleration** for massive parallel computation
- Spiral galaxy generation with stable orbital mechanics
- Support for **multiple galaxies** and customizable parameters
- Saves simulation states every few steps as `.csv` for visualization
- Generates Python file to visualize using Matplotlib
- Simulates galaxy mergers and dynamic black hole interactions

## Demo

Available as .mp4 in the repository

![Galaxy Simulation](https://github.com/Innomer/Galaxy-Simulation-Project/blob/main/galaxy_collision.mp4)  

---

## Physics Overview

The simulation uses:
- Newton's Law of Gravitation with softening to prevent singularities.
- Leapfrog integration for stable and energy-conserving time evolution.
- Velocity initialization based on circular orbits around black holes.

---

## Simulation Parameters

| Parameter         | Value/Description                       |
|-------------------|------------------------------------------|
| `N`               | Total number of bodies (e.g., 16384)     |
| `STEPS`           | Number of simulation time steps          |
| `DT`              | Time step size (e.g., 0.5)               |
| `SOFTENING`       | Softening term to avoid infinite forces  |
| `NUM_GALAXIES`    | Number of galaxies (e.g., 3)             |
| `BLACK_HOLE_MASS` | Mass of the central black hole           |
| `GALAXY_SCALE`    | Controls galaxy size and spread          |
| `SAVE_INTERVAL`   | Interval to write body positions to file |

---

## Installation

Make sure you have:

- A CUDA-capable NVIDIA GPU
- CUDA Toolkit installed
- A C/C++ compiler (e.g., `nvcc`)

```bash
git clone https://github.com/Innomer/Galaxy-Simulation-Project.git
cd Galaxy-Simulation-Project
nvcc -o galaxy_sim sim_with_cuda.cu
./galaxy_sim
```
