from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagentplanning_rl.multi_agent.reward_machine import RewardMachine
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from collections import namedtuple
from unified_planning.io.ma_pddl_writer import MAPDDLWriter
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagentplanning_rl.utils.ma_sequential_simulator import (
    UPSequentialSimulatorMA as SequentialSimulatorMA,
)
from multiagentplanning_rl.environments.utils_envs.evaluation_metrics import *
import cProfile
import json
from building_RM import RM_dict, RM_dict_true, RM_dict_true_seq
from multiagentplanning_rl.utils.message import Message
from ma_maze_office import MAP_RL_Env
from multiagentplanning_rl.render.render import EnvironmentRenderer
from multiagentplanning_rl.environments.integration_planing_and_learning.state_encoder_maze_office import (
    StateEncoderMAPRL,
)
from multiagentplanning_rl.environments.integration_planing_and_learning.detect_event import (
    PositionEventDetector,
)
from multiagentplanning_rl.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)
import wandb
import logging
import argparse

logging.basicConfig(level=logging.INFO)


NUM_EPISODES = 20000  # Numero di partite da giocare per l'apprendimento
# wandb.init(project="maze_RL_new", entity="alee8", mode="disabled")
grid_height = 4
grid_width = 4

map_maze = """
 🟩 🟩 🟩 🟩 
 🟩 🟩 🟩 🟩 
 🟩 🟩 🟩 🟩
 1  🟩 🟩 🟩
 """
map_3 = """
 D  🟩 🟩 ⛔ 🟩 🥤 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 ⛔ 🟩 🪴 🪴
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ O  🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🪴
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ ✉️ 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 
 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 ⛔ B  🟩 🪴 ⛔ 🟩 🪴 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🥤 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 
 🟩 🟩 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ C  🟩 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ 🟩 🥤 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🪴
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ O  🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 ⛔ 🟩 🪴 🪴
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ ✉️ 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🟩 🟩 🚪 🟩 🪴 🪴 
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 
 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🥤 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 
 🟩 A  🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 E  ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 F  🟩 
 """
# Stai attento al fatto che le azioni sono impostate per consiederare una griglia 3x3
map_2 = """
 B  🟩 🟩 ⛔ 🟩 🥤 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ O  🪴 🟩 ⛔ 🟩 🪴 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪
 🟩 🟩 🟩 ⛔ ✉️ 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 ⛔ 🟩 🪴 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🥤 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ 
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 A  🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 C  ⛔ 🟩 🟩 🟩
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ 
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 🟩 A  🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 C  ⛔ 🟩 🟩 🟩
 """
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
# walls, goals = parse_map_emoji(map_maze)
# coordinates, goals, office_walls = parse_office_world(map_1)

# Parse the map
coordinates_obj, goals, walls, rooms, _connections = parse_office_world_(map_1)


print("Coordinates:", coordinates_obj)
print("Goals:", goals)
print("Walls:", walls)
print("Rooms:", rooms)
print("connections:", _connections)

object_positions = {
    "plant": coordinates_obj["plant"],
    "coffee": coordinates_obj["coffee"],
    "letter": coordinates_obj["letter"],
    "office_walls": walls,
}

env = MAP_RL_Env(
    width=grid_width,
    height=grid_height,
    walls=walls,
    plant=coordinates_obj["plant"],
    cell_size=3,
)

renderer = EnvironmentRenderer(
    grid_width=env.grid_width,
    grid_height=env.grid_height,
    agents=env.agents,
    object_positions=object_positions,
    goals=goals,
    cell_size=100,  # Dimensione in pixel di una stanza
    in_cell_size=env.cell_size,  # Numero di sottocelle per dimensione all'interno della stanza
)
renderer.init_pygame()

a1 = AgentRL("a1", env)
a2 = AgentRL("a2", env)
a3 = AgentRL("a3", env)
a4 = AgentRL("a4", env)
a5 = AgentRL("a5", env)

AGENT_ORDER = [
    ("a1", a1),
    ("a2", a2),
    ("a3", a3),
    ("a4", a4),
    ("a5", a5),
]
AGENTS_BY_LABEL = dict(AGENT_ORDER)

Location = UserType("Location")
max_x_value = env.grid_width
max_y_value = env.grid_height

# prova:
l11 = Object("l11", Location)
l12 = Object("l12", Location)
l13 = Object("l13", Location)
l14 = Object("l14", Location)
l15 = Object("l15", Location)
l21 = Object("l21", Location)
l22 = Object("l22", Location)
l23 = Object("l23", Location)
l24 = Object("l24", Location)
l25 = Object("l25", Location)
l31 = Object("l31", Location)
l32 = Object("l32", Location)
l33 = Object("l33", Location)
l34 = Object("l34", Location)
l35 = Object("l35", Location)
l41 = Object("l41", Location)
l42 = Object("l42", Location)
l43 = Object("l43", Location)
l44 = Object("l44", Location)


def generate_grid_locations_and_coordinates(grid_size):
    """
    Generates locations and corresponding coordinates for a grid of given size.

    :param grid_size: Size of the grid (number of rows/columns)
    :return: List of location objects and a list of coordinate-location pairs
    """
    Location = UserType("Location")

    locations = []  # List to track created locations
    coordinates = []  # List to track coordinates and corresponding locations

    # Loop through grid rows and columns to generate locations and coordinates
    for row in range(1, grid_size + 1):
        for col in range(1, grid_size + 1):
            # Create location name based on row and column
            location_name = f"l{row}{col}"

            # Create a Location object with the generated name
            location = Object(location_name, Location)

            # Add the Location object to the list of locations
            locations.append(location)

            # Add the coordinate-location pair to the list of coordinates
            coordinates.append(((row - 1, col - 1), location))

    return locations, coordinates


def connect_locations(locations, grid_size):
    """
    Connects locations in a grid to their neighboring locations in both directions.

    :param locations: List of location objects
    :param grid_size: Size of one dimension of the grid (assuming a square grid)
    :return: List of tuples representing connected locations
    """
    connections = []

    # Dictionary to map location names to location objects
    location_dict = {str(location): location for location in locations}

    for row in range(1, grid_size + 1):
        for col in range(1, grid_size + 1):
            current_location_name = f"l{row}{col}"
            current_location = location_dict.get(current_location_name)

            # Connect to the right (bidirectional)
            if col < grid_size:
                right_location_name = f"l{row}{col + 1}"
                right_location = location_dict.get(right_location_name)
                if current_location and right_location:
                    connections.append((current_location, right_location))
                    connections.append(
                        (right_location, current_location)
                    )  # Reverse connection

            # Connect downwards (bidirectional)
            if row < grid_size:
                down_location_name = f"l{row + 1}{col}"
                down_location = location_dict.get(down_location_name)
                if current_location and down_location:
                    connections.append((current_location, down_location))
                    connections.append(
                        (down_location, current_location)
                    )  # Reverse connection

    return connections


def create_cell_connections(connections, env):
    """
    Creates connections between rooms using provided connection data and adds fluents to indicate if rooms are connected.

    :param connections: Dictionary of rooms and their connected rooms
    :param env: The environment instance where connections are set
    :return: Fluent representing the connected state between locations
    """
    Location = UserType("Location")
    location_objects = {}

    # Crea gli oggetti per ogni stanza
    for room_name in connections:
        location_objects[room_name] = Object(room_name, Location)

    # Crea il fluente is_connected
    is_connected = Fluent("is_connected", BoolType(), l1=Location, l2=Location)

    # Imposta le connessioni
    for room_name, connected_rooms in connections.items():
        for connected_room in connected_rooms:
            # Imposta il valore iniziale della connessione
            env.set_initial_value(
                is_connected(
                    location_objects[room_name], location_objects[connected_room]
                ),
                True,
            )
            # Facciamo la connessione bidirezionale
            env.set_initial_value(
                is_connected(
                    location_objects[connected_room], location_objects[room_name]
                ),
                True,
            )

    return is_connected


def create_wall_connections(walls, env):
    """
    Creates connections representing walls between coordinates, adding fluents to indicate the presence of walls.

    :param walls: List of wall pairs representing connections that have walls between them
    :param env: The environment instance where walls are set
    :return: Fluent representing the wall state between locations
    """
    Location = UserType("Location")
    location_objects = {}

    # Crea gli oggetti per ogni coordinata
    for wall_pair in walls:
        for coord in wall_pair:
            coord_name = f"({coord[0]},{coord[1]})"
            if coord_name not in location_objects:
                location_objects[coord_name] = Object(coord_name, Location)

    # Crea il fluente is_wall
    is_wall = Fluent("is_wall", BoolType(), l1=Location, l2=Location)

    # Imposta i muri
    for wall_pair in walls:
        coord1_name = f"({wall_pair[0][0]},{wall_pair[0][1]})"
        coord2_name = f"({wall_pair[1][0]},{wall_pair[1][1]})"
        # Imposta il valore iniziale per indicare la presenza di un muro
        env.set_initial_value(
            is_wall(location_objects[coord1_name], location_objects[coord2_name]),
            True,
        )
        # Facciamo il muro bidirezionale
        env.set_initial_value(
            is_wall(location_objects[coord2_name], location_objects[coord1_name]),
            True,
        )

    return is_wall


def test_policy(rm_env, episode, play=False):
    """
    Tests the current policy, logging results. Uses a different number of episodes depending on whether in play mode.

    :param rm_env: The Reward Machine environment instance.
    :param episode: Current episode number.
    :param play: Indicates the number of episodes to test during play mode, defaults to config settings if False.
    :return: The success rate and average timesteps during testing.
    """
    log_data = {}
    test_episode = (episode % 100 == 0) and (episode != 0)
    # Determine the number of test episodes based on whether play is specified
    if play:
        num_test_episodes = play
    else:
        num_test_episodes = 1
    (
        success_rate_per_agente,
        _,
        average_timesteps,
        avg_reward_per_agente,
        avg_arps_per_agente,
    ) = test_policy_optima_MAPRL(
        rm_env, episodi_test=num_test_episodes, optimal_steps=29, gamma=0.9
    )

    for _, rewag in success_rate_per_agente.items():
        if rewag > 0:
            logging.info(
                f"[{episode}] Test success rate: {success_rate_per_agente} - avg timesteps {average_timesteps} - avg reward {avg_reward_per_agente} - avg arps {avg_arps_per_agente}"
            )

    # Se non siamo in modalità play, logghiamo i dati su wandb
    if not play:
        for ag_name, arps in avg_arps_per_agente.items():
            log_data[f"avg_arps_{ag_name}"] = arps

        for ag_name, success_rate in success_rate_per_agente.items():
            log_data[f"success_rate_optima_{ag_name}"] = success_rate

        for ag_name, avg_reward in avg_reward_per_agente.items():
            log_data[f"avg_reward_optima_{ag_name}"] = avg_reward

        log_data["average_timesteps"] = average_timesteps

        wandb.log(log_data, step=episode)

    return success_rate_per_agente, average_timesteps


def initialize_experiment_metrics(agents):
    """
    Initializes metrics to track the experiment's progress, including successes, rewards, and actions.

    :param agents: List of agents in the environment
    :return: Initialized dictionaries for tracking metrics
    """
    successi_per_agente = {agent.name: 0 for agent in agents}
    ricompense_per_episodio = {agent.name: [] for agent in agents}
    actions_log = {agent.name: [] for agent in agents}
    finestra_media_mobile = 1000
    return (
        successi_per_agente,
        ricompense_per_episodio,
        actions_log,
        finestra_media_mobile,
    )


def log_wandb_data(
    rm_env,
    episode,
    rewards_agents,
    successi_per_agente,
    ricompense_per_episodio,
    finestra_media_mobile,
    total_step,
    training_steps,
):
    """
    Logs data to Weights & Biases during the experiment, including successes, rewards, and total steps.

    :param rm_env: The Reward Machine environment instance
    :param episode: The current episode number
    :param rewards_agents: Rewards obtained by each agent
    :param successi_per_agente: Success count per agent
    :param ricompense_per_episodio: Rewards per episode
    :param finestra_media_mobile: Moving average window size
    :param total_step: Total steps taken in the current run
    """
    log_data = prepare_log_data(
        rm_env.env,
        episode,
        rewards_agents,
        successi_per_agente,
        ricompense_per_episodio,
        finestra_media_mobile,
    )
    log_data.update(
        {
            f"Steps_total": total_step,
            "Training_steps": training_steps,
        }
    )

    wandb.log(log_data, step=episode)


# Example usage with a 4x4 grid (rooms)
grid_size = 4
locations, coordinates = generate_grid_locations_and_coordinates(grid_size)
env.add_objects(locations)
connections_ = connect_locations(locations, grid_size)


pos = Fluent("pos", BoolType(), pos=Location)
pos_x = Fluent(
    "pos_x",
    RealType(
        0,
    ),
)
pos_y = Fluent(
    "pos_y",
    RealType(
        0,
    ),
)
pos_i = Fluent(
    "pos_i",
    RealType(
        0,
    ),
)
pos_j = Fluent(
    "pos_j",
    RealType(
        0,
    ),
)
a1.add_to_state("timestep", 0)
a2.add_to_state("timestep", 0)
a3.add_to_state("timestep", 0)
a4.add_to_state("timestep", 0)
a5.add_to_state("timestep", 0)

a1.add_public_fluent(pos_i)
a1.add_public_fluent(pos_j)
a1.add_public_fluent(pos_x)
a1.add_public_fluent(pos_y)
a1.add_public_fluent(pos, default_initial_value=False)
a1.add_state_encoder(StateEncoderMAPRL(a1))
env.add_agent(a1)
env.set_initial_value(Dot(a1, pos_x), 3)  # 83
env.set_initial_value(Dot(a1, pos_y), 0)
env.set_initial_value(Dot(a1, pos_i), 0)  # 83
env.set_initial_value(Dot(a1, pos_j), 0)
# a1.set_initial_position_i_j(0, 0)
# env.set_initial_value(Dot(a1, pos(l33)), False)
# env.set_initial_value(Dot(a1, pos(l34)), True)

a2.add_public_fluent(pos_x)
a2.add_public_fluent(pos_y)
a2.add_public_fluent(pos_i)
a2.add_public_fluent(pos_j)
a2.add_public_fluent(pos, default_initial_value=False)
a2.add_state_encoder(StateEncoderMAPRL(a2))
env.add_agent(a2)
env.set_initial_value(Dot(a2, pos_x), 2)  # 77
env.set_initial_value(Dot(a2, pos_y), 2)
env.set_initial_value(Dot(a2, pos_i), 0)  # 83
env.set_initial_value(Dot(a2, pos_j), 0)
# a2.set_initial_position_i_j(0, 0)

a3.add_public_fluent(pos_x)
a3.add_public_fluent(pos_y)
a3.add_public_fluent(pos_i)
a3.add_public_fluent(pos_j)
a3.add_public_fluent(pos, default_initial_value=False)
a3.add_state_encoder(StateEncoderMAPRL(a3))
env.add_agent(a3)
env.set_initial_value(Dot(a3, pos_x), 1)  # 10
env.set_initial_value(Dot(a3, pos_y), 0)
env.set_initial_value(Dot(a3, pos_i), 0)  # 83
env.set_initial_value(Dot(a3, pos_j), 0)
# a3.set_initial_position_i_j(0, 0)
# env.set_initial_value(Dot(a3, pos(l34)), True)
# env.set_initial_value(Dot(a3, pos(l33)), False)

a4.add_public_fluent(pos_x)
a4.add_public_fluent(pos_y)
a4.add_public_fluent(pos, default_initial_value=False)
a4.add_state_encoder(StateEncoderMAPRL(a4))
a4.add_public_fluent(pos_i)
a4.add_public_fluent(pos_j)
env.add_agent(a4)
env.set_initial_value(Dot(a4, pos_x), 0)  # 00
env.set_initial_value(Dot(a4, pos_y), 0)
env.set_initial_value(Dot(a4, pos_i), 0)
env.set_initial_value(Dot(a4, pos_j), 0)
# a4.set_initial_position_i_j(0, 0)

a5.add_public_fluent(pos_x)
a5.add_public_fluent(pos_y)
a5.add_public_fluent(pos, default_initial_value=False)
a5.add_state_encoder(StateEncoderMAPRL(a5))
a5.add_public_fluent(pos_i)
a5.add_public_fluent(pos_j)
env.add_agent(a5)
env.set_initial_value(Dot(a5, pos_x), 1)  # 98
env.set_initial_value(Dot(a5, pos_y), 3)
env.set_initial_value(Dot(a5, pos_i), 0)  # 83
env.set_initial_value(Dot(a5, pos_j), 0)
# a5.set_initial_position_i_j(0, 0)

env.initialize_location_mapping(coordinates)


connections = []


# Utilizzo della funzione create_wall_connections


is_connected = create_cell_connections(_connections, env)
is_wall = create_wall_connections(walls, env)


door = UserType("door")
dr1 = Object("dr1", door)
dr2 = Object("dr2", door)
dr3 = Object("dr3", door)
env.add_object(dr1)
env.add_object(dr2)
env.add_object(dr3)
has_door = Fluent("has_door", BoolType(), connect_from=Location, connect_to=Location)
has_door_manager = Fluent(
    "has_door_manager", BoolType(), connect_from=Location, connect_to=Location
)
env.ma_environment.add_fluent(has_door, default_initial_value=False)
env.ma_environment.add_fluent(has_door_manager, default_initial_value=False)

# Setto i ponti
env.set_initial_value(has_door(l13, l14), True)  # TODO attenzione qui
env.set_initial_value(has_door(l14, l13), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l13, l14), False)
env.set_initial_value(is_connected(l14, l13), False)

env.set_initial_value(has_door_manager(l14, l24), True)  # TODO attenzione qui
env.set_initial_value(has_door_manager(l24, l14), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l14, l24), False)
env.set_initial_value(is_connected(l24, l14), False)

# 10x10 griglia:
env.set_initial_value(is_connected(l14, l15), False)
env.set_initial_value(is_connected(l15, l14), False)

env.ma_environment.add_fluent(is_connected, default_initial_value=False)
# Azione move_down
move_up = InstantaneousAction("up", l_from=Location, l_to=Location)
l_from = move_up.parameter("l_from")
l_to = move_up.parameter("l_to")
move_up.add_precondition(LT(0, pos_y))  # Precondizione: pos_y > 0
move_up.add_precondition(is_connected(l_from, l_to))
move_up.add_precondition(Equals(pos_j, 0))
# move_up.add_precondition(pos(l_from))
move_up.add_decrease_effect(pos_y, 1)
move_up.add_effect(pos(l_to), True)
move_up.add_effect(pos(l_from), False)
move_up.add_effect(pos_j, env.cell_size - 1)

# move_down.add_effect(pos_y, Minus(pos_y, 1))  # Effetto: decrementa pos_y di 1
a1.add_rl_action(move_up)
a2.add_rl_action(move_up)
a3.add_rl_action(move_up)
a4.add_rl_action(move_up)
a5.add_rl_action(move_up)

# Azione move_up
move_down = InstantaneousAction("down", l_from=Location, l_to=Location)
move_down.add_precondition(
    LT(pos_y, max_y_value - 1)
)  # Precondizione: pos_y < max_y_value
move_down.add_precondition(is_connected(l_from, l_to))
move_down.add_precondition(Equals(pos_j, env.cell_size - 1))
# move_down.add_precondition(pos(l_from))
move_down.add_increase_effect(pos_y, 1)
move_down.add_effect(pos(l_to), True)
move_down.add_effect(pos(l_from), False)
move_down.add_effect(pos_j, 0)

# move_up.add_effect(pos_y, Plus(pos_y, 1))  # Effetto: incrementa pos_y di 1
a1.add_rl_action(move_down)
a2.add_rl_action(move_down)
a3.add_rl_action(move_down)
a4.add_rl_action(move_down)
a5.add_rl_action(move_down)

# Azione move_left
move_left = InstantaneousAction("left", l_from=Location, l_to=Location)
move_left.add_precondition(LT(0, pos_x))  # Precondizione: pos_x > 0
move_left.add_precondition(is_connected(l_from, l_to))
move_left.add_precondition(Equals(pos_i, 0))
# move_left.add_precondition(pos(l_from))
move_left.add_effect(pos(l_to), True)
move_left.add_effect(pos(l_from), False)
move_left.add_effect(pos_i, env.cell_size - 1)
move_left.add_decrease_effect(pos_x, 1)
# move_left.add_effect(pos_x, Minus(pos_x, 1))  # Effetto: decrementa pos_x di 1
a1.add_rl_action(move_left)
a2.add_rl_action(move_left)
a3.add_rl_action(move_left)
a4.add_rl_action(move_left)
a5.add_rl_action(move_left)

# Azione move_right
move_right = InstantaneousAction("right", l_from=Location, l_to=Location)
move_right.add_precondition(
    LT(pos_x, max_x_value - 1)
)  # Precondizione: pos_x < max_x_value
move_right.add_precondition(is_connected(l_from, l_to))
move_right.add_precondition(Equals(pos_i, env.cell_size - 1))
# move_right.add_precondition(pos(l_from))
move_right.add_effect(pos(l_to), True)
move_right.add_effect(pos(l_from), False)
move_right.add_effect(pos_i, 0)
move_right.add_increase_effect(pos_x, 1)

# move_right.add_effect(pos_x, Plus(pos_x, 1))  # Effetto: incrementa pos_x di 1
a1.add_rl_action(move_right)
a2.add_rl_action(move_right)
a3.add_rl_action(move_right)
a4.add_rl_action(move_right)
a5.add_rl_action(move_right)


low_up = InstantaneousAction("low_up", l_from=Location, l_to=Location)
# low_up.add_precondition(Not(is_wall(l_from, l_to)))  # precondizione: non deve esserci un muro
low_up.add_precondition(LT(0, pos_j))  # right > left
low_up.add_decrease_effect(pos_j, 1)

low_down = InstantaneousAction("low_down", l_from=Location, l_to=Location)
# low_down.add_precondition(Not(is_wall(l_from, l_to)))  # precondizione: non deve esserci un muro
low_down.add_precondition(LT(pos_j, env.cell_size - 1))
low_down.add_increase_effect(pos_j, 1)

low_left = InstantaneousAction("low_left", l_from=Location, l_to=Location)
# low_left.add_precondition(Not(is_wall(l_from, l_to)))  # precondizione: non deve esserci un muro
low_left.add_precondition(LT(0, pos_i))
low_left.add_decrease_effect(pos_i, 1)

low_right = InstantaneousAction("low_right", l_from=Location, l_to=Location)
# low_right.add_precondition(Not(is_wall(l_from, l_to)))  # precondizione: non deve esserci un muro
low_right.add_precondition(LT(pos_i, env.cell_size - 1))
low_right.add_increase_effect(pos_i, 1)

cross_up = InstantaneousAction("cross_up", l_from=Location, l_to=Location)
cross_up.add_precondition(LT(0, pos_y))
cross_up.add_precondition(has_door(l_from, l_to))
cross_up.add_precondition(Equals(pos_j, 0))
cross_up.add_decrease_effect(pos_y, 1)
cross_up.add_effect(pos_j, env.cell_size - 1)
cross_up.add_effect(pos(l_to), True)
cross_up.add_effect(pos(l_from), False)
# cross_up.add_effect(has_door(l_from, l_to), False)
# cross_up.add_effect(pos_x, env.get_coordinates_by_location(a1, l_to)[0], True)
# cross_up.add_effect(pos_y, env.get_coordinates_by_location(a1, l_to)[1], True)

cross_down = InstantaneousAction("cross_down", l_from=Location, l_to=Location)
cross_down.add_precondition(LT(pos_y, max_y_value - 1))
cross_down.add_precondition(has_door(l_from, l_to))
cross_down.add_precondition(Equals(pos_j, env.cell_size - 1))
cross_down.add_increase_effect(pos_y, 1)
cross_down.add_effect(pos_j, 0)
cross_down.add_effect(pos(l_to), True)
cross_down.add_effect(pos(l_from), False)
# cross_down.add_effect(has_door(l_from, l_to), False)

cross_right = InstantaneousAction("cross_right", l_from=Location, l_to=Location)
cross_right.add_precondition(LT(pos_x, max_x_value - 1))
cross_right.add_precondition(has_door(l_from, l_to))
cross_right.add_precondition(Equals(pos_i, env.cell_size - 1))
cross_right.add_increase_effect(pos_x, 1)
cross_right.add_effect(pos_i, 0)
cross_right.add_effect(pos(l_to), True)
cross_right.add_effect(pos(l_from), False)
# cross_right.add_effect(has_door(l_from, l_to), False)

cross_left = InstantaneousAction("cross_left", l_from=Location, l_to=Location)
cross_left.add_precondition(LT(0, pos_x))
cross_left.add_precondition(has_door(l_from, l_to))
cross_left.add_precondition(Equals(pos_i, 0))
cross_left.add_decrease_effect(pos_x, 1)
cross_left.add_effect(pos_i, env.cell_size - 1)
cross_left.add_effect(pos(l_to), True)
cross_left.add_effect(pos(l_from), False)
# cross_left.add_effect(has_door(l_from, l_to), False)

wait = InstantaneousAction("wait", l_from=Location, l_to=Location)
wait.add_decrease_effect(pos_x, 0)

row_up = InstantaneousAction("row_up", l_from=Location, l_to=Location)
row_up.add_precondition(LT(0, pos_y))
row_up.add_precondition(has_door_manager(l_from, l_to))
row_up.add_effect(pos_i, env.cell_size - 1)
row_up.add_decrease_effect(pos_y, 1)
row_up.add_effect(pos_j, env.cell_size - 1)
# row_up.add_effect(has_door_manager(l_from, l_to), False)
row_up.add_effect(pos(l_to), True)
row_up.add_effect(pos(l_from), False)


row_down = InstantaneousAction("row_down", l_from=Location, l_to=Location)
row_down.add_precondition(LT(pos_y, max_y_value - 1))
row_down.add_precondition(has_door_manager(l_from, l_to))
row_down.add_precondition(Equals(pos_j, env.cell_size - 1))
row_down.add_increase_effect(pos_y, 1)
row_down.add_effect(pos_j, 0)
# row_down.add_effect(has_door_manager(l_from, l_to), False)
row_down.add_effect(pos(l_to), True)
row_down.add_effect(pos(l_from), False)

row_right = InstantaneousAction("row_right", l_from=Location, l_to=Location)
row_right.add_precondition(LT(pos_x, max_x_value - 1))
row_right.add_precondition(has_door_manager(l_from, l_to))
row_right.add_precondition(Equals(pos_i, env.cell_size - 1))
row_right.add_increase_effect(pos_x, 1)
row_right.add_effect(pos_i, 0)
# row_right.add_effect(has_door_manager(l_from, l_to), False)
row_right.add_effect(pos(l_to), True)
row_right.add_effect(pos(l_from), False)

row_left = InstantaneousAction("row_left", l_from=Location, l_to=Location)
row_left.add_precondition(LT(0, pos_x))
row_left.add_precondition(has_door_manager(l_from, l_to))
row_left.add_precondition(Equals(pos_i, 0))
row_left.add_decrease_effect(pos_x, 1)
row_left.add_effect(pos_i, env.cell_size - 1)
# row_right.add_effect(has_door_manager(l_from, l_to), False)
row_left.add_effect(pos(l_to), True)
row_left.add_effect(pos(l_from), False)

a1.add_rl_action(low_up)
a1.add_rl_action(low_down)
a1.add_rl_action(low_left)
a1.add_rl_action(low_right)
a1.add_rl_action(cross_up)
a1.add_rl_action(cross_down)
a1.add_rl_action(cross_right)
a1.add_rl_action(cross_left)
a1.add_rl_action(wait)
"""a1.add_rl_action(row_up)
a1.add_rl_action(row_down)
a1.add_rl_action(row_right)
a1.add_rl_action(row_left)"""

a2.add_rl_action(low_up)
a2.add_rl_action(low_down)
a2.add_rl_action(low_left)
a2.add_rl_action(low_right)
"""a2.add_rl_action(cross_up)
a2.add_rl_action(cross_down)
a2.add_rl_action(cross_right)
a2.add_rl_action(cross_left)"""
a2.add_rl_action(wait)
a2.add_rl_action(row_up)
a2.add_rl_action(row_down)
a2.add_rl_action(row_right)
a2.add_rl_action(row_left)

a3.add_rl_action(low_up)
a3.add_rl_action(low_down)
a3.add_rl_action(low_left)
a3.add_rl_action(low_right)
a3.add_rl_action(cross_up)
a3.add_rl_action(cross_down)
a3.add_rl_action(cross_right)
a3.add_rl_action(cross_left)
a3.add_rl_action(wait)
"""a3.add_rl_action(row_up)
a3.add_rl_action(row_down)
a3.add_rl_action(row_right)
a3.add_rl_action(row_left)"""

a4.add_rl_action(low_up)
a4.add_rl_action(low_down)
a4.add_rl_action(low_left)
a4.add_rl_action(low_right)
a4.add_rl_action(cross_up)
a4.add_rl_action(cross_down)
a4.add_rl_action(cross_right)
a4.add_rl_action(cross_left)
a4.add_rl_action(wait)
"""a4.add_rl_action(row_up)
a4.add_rl_action(row_down)
a4.add_rl_action(row_right)
a4.add_rl_action(row_left)"""

a5.add_rl_action(low_up)
a5.add_rl_action(low_down)
a5.add_rl_action(low_left)
a5.add_rl_action(low_right)
"""a5.add_rl_action(cross_up)
a5.add_rl_action(cross_down)
a5.add_rl_action(cross_right)
a5.add_rl_action(cross_left)"""
a5.add_rl_action(wait)
a5.add_rl_action(row_up)
a5.add_rl_action(row_down)
a5.add_rl_action(row_right)
a5.add_rl_action(row_left)


def setup_agent_rm(agent, transitions):
    """
    Sets up the Reward Machine for a specific agent and adds an event detector for the extracted events.

    :param agent: The agent to set up with the Reward Machine
    :param transitions: Transitions defining the Reward Machine
    :return: Reward Machine and event detector instances
    """
    RM = RewardMachine(transitions, None)
    event_detector = PositionEventDetector(RM.extract_events(), agent)
    RM.event_detector = event_detector
    agent.set_reward_machine(RM)
    return RM, event_detector


new_transitions_ag_1 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((goals["O"], True),)): ("state4", 0),
    ("state4", ((goals["C"], True),)): ("state5", 0),
}

new_transitions_ag_2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((goals["B"], True),)): ("state4", 0),
    ("state4", ((goals["O"], True),)): ("state5", 0),
}

new_transitions_ag_3 = {
    ("state1", ((goals["C"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((coordinates_obj["coffee"][0], True),)): ("state4", 0),
    ("state3", ((coordinates_obj["coffee"][1], True),)): ("state4", 0),
    ("state3", ((goals["O"], True),)): ("state5", 0),
}

new_transitions_ag_4 = {
    ("state1", ((goals["O"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["coffee"][0], True),)): ("state3", 0),
    ("state2", ((coordinates_obj["coffee"][1], True),)): ("state3", 0),
    ("state3", ((coordinates_obj["letter"][0], True),)): ("state4", 0),
    ("state4", ((goals["B"], True),)): ("state5", 0),
}

new_transitions_ag_5 = {
    ("state1", ((goals["C"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((coordinates_obj["coffee"][0], True),)): ("state4", 0),
    ("state3", ((coordinates_obj["coffee"][1], True),)): ("state4", 0),
    ("state4", ((goals["O"], True),)): ("state5", 0),
}

new_transitions_ag_5_and_ag2_exp = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
}

a2_new_transitions_ag_5_and_ag2_exp2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((goals["B"], True),)): ("state3", 0),
}

a5_new_transitions_ag_5_and_ag2_exp2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((goals["C"], True),)): ("state3", 0),
}

# TODO IQL exp2
transitions_ag_2_exp2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((goals["B"], True),)): ("state3", 0),
    ("state3", ((("pos(l14)"), True),)): ("state4", 100),
}
transitions_ag_5_exp2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((goals["C"], True),)): ("state3", 0),
    ("state3", ((("pos(l14)"), True),)): ("state4", 100),
}

# TODO IQL exp1
transitions_ag_2_exp1 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 10),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 10),
    ("state2", ((("pos(l14)"), True),)): ("state4", 100),
}
transitions_ag_5_exp1 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 10),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 10),
    ("state2", ((("pos(l14)"), True),)): ("state4", 100),
}

# TODO IQL exp 0 5agents (only MAP)
transitions_ag5_ag2_exp0 = {
    ("state2", ((("pos(l14)"), True),)): ("state4", 100),
}
transitions_ag1_ag3_ag4_exp0 = {
    ("state2", ((("pos(l14)"), True),)): ("state4", 100),
}
# Funzione principale per eseguire l'esperimento
def initialize_reward_machines(experiment):
    rm_event_pairs = {}
    for agent_label, agent in AGENT_ORDER:
        transitions = RM_dict_true_seq[agent_label]
        rm, event_detector = setup_agent_rm(agent, transitions)
        rm_event_pairs[agent_label] = (rm, event_detector)

    if experiment == "exp2":
        rm_event_pairs["a2"][0].add_transitions_with_merge(
            a2_new_transitions_ag_5_and_ag2_exp2, position="before", prefix="new"
        )
        rm_event_pairs["a5"][0].add_transitions_with_merge(
            a5_new_transitions_ag_5_and_ag2_exp2, position="before", prefix="new"
        )
    elif experiment == "exp3":
        exp3_transitions = {
            "a1": new_transitions_ag_1,
            "a2": new_transitions_ag_2,
            "a3": new_transitions_ag_3,
            "a4": new_transitions_ag_4,
            "a5": new_transitions_ag_5,
        }
        for agent_label, transitions in exp3_transitions.items():
            rm_event_pairs[agent_label][0].add_transitions_with_merge(
                transitions, position="before", prefix="new"
            )

    for agent_label, (rm, event_detector) in rm_event_pairs.items():
        event_detector.add_events(rm.extract_events())
        rm.event_detector = event_detector
        AGENTS_BY_LABEL[agent_label].set_reward_machine(rm)

    return {agent_label: rm for agent_label, (rm, _) in rm_event_pairs.items()}


# Funzione principale per eseguire l'esperimento
def run_experiment(num_episodes, wandb_enabled, experiment):
    reward_machines = initialize_reward_machines(experiment)
    if wandb_enabled:
        wandb.init(project="maze_RL_new", entity="alee8", mode="online")
    else:
        wandb.init(project="maze_RL_new", entity="alee8", mode="disabled")
    global NUM_EPISODES
    NUM_EPISODES = num_episodes
    # TODO deccomentare
    rm_env = RMEnvironmentWrapper(
        env, [a1, a2, a3, a4, a5]
    )  # [a2, a5]) #[a1, a2, a3, a4, a5])
    q_learning1 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * reward_machines["a1"].numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=42,
        qtable_init=2,
        use_qrm=True,
    )

    q_learning2 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * reward_machines["a2"].numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=43,
        qtable_init=2,
        use_qrm=True,
    )

    q_learning3 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * reward_machines["a3"].numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=44,
        qtable_init=2,
        use_qrm=True,
    )

    q_learning4 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * reward_machines["a4"].numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=45,
        qtable_init=2,
        use_qrm=True,
    )

    q_learning5 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * reward_machines["a5"].numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=46,
        qtable_init=2,
        use_qrm=True,
    )

    a1.set_learning_algorithm(q_learning1)
    a2.set_learning_algorithm(q_learning2)
    a3.set_learning_algorithm(q_learning3)
    a4.set_learning_algorithm(q_learning4)
    a5.set_learning_algorithm(q_learning5)

    successi_per_agente = {agent.name: 0 for agent in env.agents}
    ricompense_per_episodio = {agent.name: [] for agent in env.agents}
    actions_log = {}
    q_tables = {}
    total_step = 0
    # env.reset()
    rm_env.env.initialize_state()

    seed = 111
    total_training_steps = 0

    for episode in range(NUM_EPISODES):
        states, infos = rm_env.reset(seed)
        states = copy.deepcopy(states)
        done = {a.name: False for a in rm_env.agents}
        rewards_agents = {a.name: 0 for a in rm_env.agents}
        total_steps_per_agent = {a.name: 0 for a in rm_env.agents}
        episode_total_steps = 0

        # Determina se questo è un episodio di test
        test_episode = (episode % 100 == 0) and (episode != 0)

        # Imposta il flag per l'esplorazione
        if test_episode:
            exploration = False  # Usa la policy ottima
        else:
            exploration = True  # Usa la policy con esplorazione

        record_episode = episode % 500 == 0 and episode != 0
        # record_episode = False
        if record_episode:
            renderer.render(episode, states)  # Cattura frame durante l'episodio
            actions_log = {agent.name: [] for agent in env.agents}
            """renderer.save_episode(
                episode
            )"""

        while any(rm_env.env.active_agents.values()):
            total_training_steps += 1
            episode_total_steps += 1
            actions = {}
            rewards = {
                a.name: 0 for a in rm_env.agents
            }  # Inizializza le ricompense episodiche
            infos = {a.name: {} for a in rm_env.agents}
            for ag in rm_env.agents:
                if not rm_env.env.active_agents.get(ag.name, True):
                    continue  # Salta gli agenti che erano già inattivi
                current_state = rm_env.env.get_state(ag)
                action = ag.select_action(current_state, best=not exploration)
                actions[ag.name] = action
                # Log delle azioni nell'ultimo episodio
                if record_episode:
                    actions_log[ag.name].append(actions[ag.name].name)
            new_states, rewards, done, truncations, infos = rm_env.step(actions)

            for agent in rm_env.agents:
                agent_just_terminated = (
                    done[agent.name] and rm_env.env.active_agents[agent.name]
                )
                if (
                    not rm_env.env.active_agents[agent.name]
                    and not agent_just_terminated
                ):
                    continue  # Salta gli agenti inattivi che non hanno appena terminato

                # Aggiorna il conteggio dei passi per agente
                total_steps_per_agent[agent.name] += 1

                agent.update_policy(
                    state=states[agent.name],
                    action=actions[agent.name],
                    reward=rewards[agent.name],
                    next_state=new_states[agent.name],
                    terminated=done[agent.name],
                    infos=infos[agent.name],
                )
                rewards_agents[agent.name] += rewards[agent.name]

            # Aggiorna lo stato degli agenti dopo aver processato le ricompense
            for agent in rm_env.agents:
                if done.get(agent.name, False):
                    rm_env.env.active_agents[agent.name] = False

            states = copy.deepcopy(new_states)
            if record_episode:
                renderer.render(episode, states)  # Cattura frame durante l'episodio

            if all(truncations.values() or done.values()):
                break

        if record_episode:
            renderer.save_episode(
                episode
            )  # Salva il video solo alla fine dell'episodio
        # Dopo l'episodio, logga i dati se è un episodio di test
        if test_episode and wandb_enabled:
            log_data = {
                "total_steps_episode": episode_total_steps,  # Step totali per completare l'episodio
                "training_steps": total_training_steps,
            }
            log_test_episode_data(
                rm_env, total_steps_per_agent, rewards_agents, episode
            )

            wandb.log(log_data, step=episode)

        epsilon_str = get_epsilon_summary(rm_env.agents)

        logging.info(
            f"Episodio {episode + 1}: Ricompensa = {rewards_agents}, Total Steps: {total_step + 1}, Episode Step: {rm_env.env.timestep}, Agents Step = {rm_env.env.agent_steps}, Epsilon agents= [{epsilon_str}]"
        )
    wandb.finish()

    # Salva il log delle azioni e le Q-table in un file JSON
    with open("final_episode_log.json", "w") as f:
        json.dump({"actions_log": actions_log, "q_tables": q_tables}, f, indent=4)


# Imposta argparse per gestire la linea di comando
def parse_args():
    parser = argparse.ArgumentParser(
        description="Lancia esperimenti multi-agente su maze RL"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20000,
        help="Numero di episodi per cui eseguire l'apprendimento",
    )
    parser.add_argument(
        "--wandb_enabled", action="store_true", help="Abilita l'invio dei log a WandB"
    )
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "exp3"],
        default="exp1",
        help="Seleziona il task da eseguire",
    )
    return parser.parse_args()


# Lancia l'esperimento con i parametri dalla linea di comando
if __name__ == "__main__":
    args = parse_args()

    # Esegui l'esperimento con i parametri definiti da argparse
    run_experiment(
        num_episodes=args.num_episodes,
        wandb_enabled=args.wandb_enabled,
        experiment=args.experiment,
    )
