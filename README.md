# MarsExplorer

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MarsExplorer)

## Description

MarsExplorer is an environment for evaluating agents on grid-based terrain exploration and coverage. Based on the [MarsExplorer](https://github.com/dimikout3/MarsExplorer) environment by Koutras et al., agents control a rover navigating a procedurally-generated 2D grid with obstacles, using a simulated LIDAR sensor to reveal unknown terrain. The goal is to explore as much of the map as possible while avoiding obstacles and staying within bounds.

## Capabilities

- Spatial reasoning and navigation planning on ASCII grid maps
- Obstacle avoidance from partial observations
- Exploration strategy under step budgets
- Adapting to procedurally-generated terrain of varying difficulty

## Compute Requirements

MarsExplorer does not require a sandbox. All game logic runs in-process with minimal compute.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are 1,000 tasks in the `train` split, across three difficulty tiers:

- **Small** (334 tasks): 11x11 grid, 5 obstacles, 150 max steps, LIDAR range 4
- **Medium** (333 tasks): 21x21 grid, 12 obstacles, 400 max steps, LIDAR range 6
- **Large** (333 tasks): 41x41 grid, 30 obstacles, 1000 max steps, LIDAR range 8

Each task uses a fixed seed for reproducible map generation. The agent starts at position (0, 0) and must explore the grid by issuing directional move commands.

## Reward Structure

This is a dense, verifiable reward environment matching the original MarsExplorer reward structure. No LLM graders are used.

**Per-step reward** (every move):
- `new_explored_cells - movement_cost` (movement_cost = 0.2)
- Exploring new terrain yields positive reward; revisiting explored areas costs 0.2

**Terminal rewards:**
- **95%+ explored (success)**: the final move's reward is the +400 bonus (in place of the step reward), episode ends
- **Collision with obstacle**: -400 (in place of the step reward), episode ends
- **Out of bounds**: -400 (in place of the step reward), episode ends
- **Max steps reached**: episode ends with final step reward

## Data

No external data is used. Maps are generated procedurally from each task's seed.

## Tools

- `move(direction)`: Move the rover one cell in a cardinal direction (`"up"`, `"down"`, `"left"`, `"right"`). Returns an ASCII map observation showing the current state of exploration, along with step count, position, and exploration percentage.

## Map Representation

The agent receives a text-based ASCII grid after each move:

```
Step: 15/400 | Position: (5, 3) | Explored: 34.2% (151/441)

    01234567890123456789
 0  ..............??????
 1  ..............??????
 2  ......##......??????
 3  .....@.#......??????
 4  ..............??????
 ...

Legend: @ = rover, # = obstacle, . = explored, ? = unexplored
```

## Time Horizon

MarsExplorer is a multi-turn environment with one `move` call per step, up to 150, 400 or 1,000 moves depending on the tier.

## Other Environment Requirements

There are no further environment requirements. MarsExplorer needs no secrets.

## Safety

MarsExplorer is a grid navigation task with no safety concerns. The agent interacts only with an abstract grid world and cannot affect any real systems.

## Citations

```bibtex
@article{koutras2021marsexplorer,
  title={MarsExplorer: Exploration of Unknown Terrains via Deep Reinforcement Learning and Procedurally Generated Environments},
  author={Koutras, Dimitrios I. and Kapoutsis, Athanasios Ch. and Amanatiadis, Angelos A. and Kosmatopoulos, Elias B.},
  journal={Electronics},
  volume={10},
  number={22},
  pages={2751},
  year={2021},
  publisher={MDPI},
  doi={10.3390/electronics10222751}
}
```
