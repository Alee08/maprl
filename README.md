# MAPRL — Multi‑Agent Planning for Reinforcement Learning with Reward Machines


**Repository:** https://github.com/Alee08/maprl  
**Paper (ECAI 2025):** https://doi.org/10.3233/FAIA251253  
**Companion library (required):** https://github.com/Alee08/multiagent-rl-rm

> **Note**  
> MAPRL **explicitly depends on** `multiagent-rl-rm`.  
> This dependency is declared in `setup.py`, so `pip install -e .` will automatically pull it from PyPI.

---
## Overview

**MAPRL** integrates **partial-order planning (POP)** with **Reward Machines (RMs)** to address **strongly-coupled cooperative MARL** problems.

Workflow:
1. A **multi-agent planning (MAP)** model is solved to a **POP**.
2. The POP is transformed into **one RM per agent** (linear chains that encode the relevant preconditions/effects).
3. Each agent learns **locally** (e.g., Q-learning) guided by its RM state; **public/joint actions** are executed automatically when their preconditions hold.

This repository provides the **MAPRL library** that performs POP→RM synthesis and exposes utilities to coordinate distributed learners via RMs.


## Installation (developer mode)

> MAPRL is **not yet on PyPI**. Install it in **developer mode**.

```bash
# Clone MAPRL
git clone https://github.com/Alee08/maprl
cd maprl

pip install -e .
```

The command above will automatically install the dependency multiagent-rl-rm from PyPI (as specified in setup.py).

### Running the Experiment

You can run the experiment using the command-line interface:

```bash
python ma_maze_office.py --num_episodes 20000 --wandb_enabled
```

## Example Environment Configuration

The environment is defined using a grid-like structure with various objects such as walls, plants, and goals. The `map_1` string in `ma_maze_office.py` represents the grid layout, where:

- 🟩 - Empty space
- ⛔ - Wall
- 🚪 - Door
- 🥤 - Coffee station
- 🪴 - Plant
- ✉️ - Letter
- `A`, `B`, `C`, `O` etc. - Indicate specific goal locations

The positions and connections of the grid are parsed using custom parsing functions, such as `parse_office_world`.

## Example Map

Below is an example of a `12x12` grid environment used in the project. 
```python
map_1 = """
 B  🟩 🟩 ⛔ 🟩 🥤 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ O  🪴 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ ✉️ 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🥤 🟩 🟩 ⛔ 🟩 🟩 🟩
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 A  🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 C 
 """
```

## Actions

The agents can perform various actions, including:

- `move_up`, `move_down`, `move_left`, `move_right` - Move within the grid.
- `cross_up`, `cross_down`, `cross_left`, `cross_right` - Cross to adjacent locations using bridges.
- `row_up`, `row_down`, `row_left`, `row_right` - Row across to different locations using boats.
- `low_up`, `low_down`, `low_left`, `low_right` - Move within sub-cells of a grid cell.
- `wait` - Stay in the current position.

