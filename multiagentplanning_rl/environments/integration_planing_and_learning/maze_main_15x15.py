import functools
import random
import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import pygame
from time import *
from pettingzoo import ParallelEnv
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagentplanning_rl.multi_agent.reward_machine import RewardMachine
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from collections import namedtuple
from unified_planning.io.ma_pddl_writer import MAPDDLWriter
from multiagentplanning_rl.multi_agent.agent_rl import AgentRL
import cv2
from multiagentplanning_rl.utils.ma_sequential_simulator import (
    UPSequentialSimulatorMA as SequentialSimulatorMA,
)
from multiagentplanning_rl.environments.utils_envs.evaluation_metrics import *
import cProfile
import json
import pickle
from building_RM import RM_dict, RM_dict_true, RM_dict_true_seq
from multiagentplanning_rl.utils.message import Message
from ma_maze_office import MAP_RL_Env
from multiagentplanning_rl.render.render import EnvironmentRenderer
from multiagentplanning_rl.environments.integration_planing_and_learning.state_encoder_maze_office import (
    StateEncoderMAPRL,
)
from multiagentplanning_rl.environments.integration_planing_and_learning.detect_event_2 import (
    PositionEventDetector,
)
from multiagentplanning_rl.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)
import wandb
from pettingzoo.test import parallel_api_test
import random
from multiagentplanning_rl.utils.utils import (
    encode_state,
    parse_map_string,
    parse_map_emoji,
    parse_office_world,
)

# from heatmap import generate_heatmaps
#from multiagentplanning_rl.render.heatmap import generate_heatmaps
import string
from multiagentplanning_rl.multi_agent.action_rl import ActionRL
import logging
import argparse

logging.basicConfig(level=logging.INFO)


NUM_EPISODES = 20000  # Numero di partite da giocare per l'apprendimento
# wandb.init(project="maze_RL_new", entity="alee8", mode="disabled")

map_maze = """
 🟩 🟩 🟩 🟩 
 🟩 🟩 🟩 🟩 
 🟩 🟩 🟩 🟩
 1  🟩 🟩 🟩
 """
map_3 = """
 D  🟩 🟩 ⛔ 🟩 🥤 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🪴 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🥤
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 ⛔ 🟩 🪴 🪴
 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🪴
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🟩 🟩 🚪 🟩 🪴 🪴 
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 ⛔ 🪴 🟩 🪴 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 🚪 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 ⛔ B  🟩 🪴 ⛔ 🟩 🪴 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ #TODO aggiungere bridge a stanza B
 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 
 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 
 ✉️ 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩
 🪴 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🪴
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ O  🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 ⛔ 🟩 🪴 🪴
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🟩 🟩 🚪 🟩 🪴 🪴 
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 🚪 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ 
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🪴 🟩 ✉️ 
 🪴 🟩 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🪴 
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🪴 🟩 🟩 
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ 
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🪴 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🪴 
 🟩 A  🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🪴 
 🪴 🟩 🟩 ⛔ 🟩 🟩 🥤 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 C  ⛔ 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🪴 ✉️ 🪴 
 """
# Stai attento al fatto che le azioni sono impostate per consiederare una griglia 3x3
map_2 = """
 🪴 🪴 B  ⛔ 🟩 🥤 🟩 ⛔ 🪴 🟩 🪴 ⛔ 🟩 🪴 🪴 ⛔ D  🪴 ✉️
 🪴 🟩 🟩 🚪 🟩 🪴 🪴 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩
 🪴 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ E  🪴 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪
 🪴 🟩 🟩 ⛔ ✉️ 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🪴 🪴 🟩
 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🪴 ⛔ 🪴 🪴 🟩
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔
 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🪴 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🟩 🟩 🟩
 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🪴 🪴
 🟩 🟩 🟩 ⛔ 🟩 🪴 🟩 ⛔ 🥤 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩
 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ 
 🟩 🟩 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴 ⛔ 🟩 🪴 🪴
 🪴 🪴 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 ⛔ 🟩 🟩 🪴
 🟩 🟩 A  ⛔ 🟩 🟩 🟩 ⛔ 🪴 🟩 🟩 ⛔ 🟩 🟩 C  ⛔ 🟩 🟩 🪴
 🚪 ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🚪 ⛔ ⛔ 🚪 ⛔ ⛔ 
 🟩 🪴 🟩 🚪 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🟩 ⛔ 🟩 🟩 🪴
 🟩 🟩 🟩 ⛔ 🪴 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🪴 🟩 🚪 🟩 🟩 🟩
 🪴 🪴 🪴 ⛔ 🥤 🟩 🟩 ⛔ 🪴 🟩 🪴 ⛔ 🪴 🟩 B  ⛔ 🟩 🪴 🪴
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
MAP = map_2
if MAP == map_3:
    grid_height = 8
    grid_width = 8
    grid_size = 8
elif MAP == map_1:
    grid_height = 4
    grid_width = 4
    grid_size = 4
elif MAP == map_2:
    grid_height = 5
    grid_width = 5
    grid_size = 5
# Parse the map
coordinates_obj, goals, walls, rooms, _connections = parse_office_world_(MAP)


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

a6 = AgentRL("a6", env)
a7 = AgentRL("a7", env)
a8 = AgentRL("a8", env)
a9 = AgentRL("a9", env)
a10 = AgentRL("a10", env)

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
a6.add_to_state("timestep", 0)
a7.add_to_state("timestep", 0)
a8.add_to_state("timestep", 0)
a9.add_to_state("timestep", 0)
a10.add_to_state("timestep", 0)

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
env.set_initial_value(Dot(a2, pos_x), 4)  # 77
env.set_initial_value(Dot(a2, pos_y), 1)
env.set_initial_value(Dot(a2, pos_i), 2)  # 83
env.set_initial_value(Dot(a2, pos_j), 2)
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
env.set_initial_value(Dot(a4, pos_i), 1)
env.set_initial_value(Dot(a4, pos_j), 1)
# a4.set_initial_position_i_j(0, 0)

a5.add_public_fluent(pos_x)
a5.add_public_fluent(pos_y)
a5.add_public_fluent(pos, default_initial_value=False)
a5.add_state_encoder(StateEncoderMAPRL(a5))
a5.add_public_fluent(pos_i)
a5.add_public_fluent(pos_j)
env.add_agent(a5)
env.set_initial_value(Dot(a5, pos_x), 4)  # 98
env.set_initial_value(Dot(a5, pos_y), 3)
env.set_initial_value(Dot(a5, pos_i), 0)  # 83
env.set_initial_value(Dot(a5, pos_j), 0)
# a5.set_initial_position_i_j(0, 0)

a6.add_public_fluent(pos_x)
a6.add_public_fluent(pos_y)
a6.add_public_fluent(pos, default_initial_value=False)
a6.add_state_encoder(StateEncoderMAPRL(a6))
a6.add_public_fluent(pos_i)
a6.add_public_fluent(pos_j)
env.add_agent(a6)
env.set_initial_value(Dot(a6, pos_x), 4)  # 98
env.set_initial_value(Dot(a6, pos_y), 4)
env.set_initial_value(Dot(a6, pos_i), 0)  # 83
env.set_initial_value(Dot(a6, pos_j), 0)
# a5.set_initial_position_i_j(0, 0)

a7.add_public_fluent(pos_x)
a7.add_public_fluent(pos_y)
a7.add_public_fluent(pos, default_initial_value=False)
a7.add_state_encoder(StateEncoderMAPRL(a7))
a7.add_public_fluent(pos_i)
a7.add_public_fluent(pos_j)
env.add_agent(a7)
env.set_initial_value(Dot(a7, pos_x), 3)  # 98
env.set_initial_value(Dot(a7, pos_y), 3)
env.set_initial_value(Dot(a7, pos_i), 0)  # 83
env.set_initial_value(Dot(a7, pos_j), 0)
# a5.set_initial_position_i_j(0, 0)

a8.add_public_fluent(pos_x)
a8.add_public_fluent(pos_y)
a8.add_public_fluent(pos, default_initial_value=False)
a8.add_state_encoder(StateEncoderMAPRL(a8))
a8.add_public_fluent(pos_i)
a8.add_public_fluent(pos_j)
env.add_agent(a8)
env.set_initial_value(Dot(a8, pos_x), 2)  # 98
env.set_initial_value(Dot(a8, pos_y), 2)
env.set_initial_value(Dot(a8, pos_i), 2)  # 83
env.set_initial_value(Dot(a8, pos_j), 2)
# a5.set_initial_position_i_j(0, 0)

"""a9.add_public_fluent(pos_x)
a9.add_public_fluent(pos_y)
a9.add_public_fluent(pos, default_initial_value=False)
a9.add_state_encoder(StateEncoderMAPRL(a9))
a9.add_public_fluent(pos_i)
a9.add_public_fluent(pos_j)
env.add_agent(a9)
env.set_initial_value(Dot(a9, pos_x), 7)  # 98
env.set_initial_value(Dot(a9, pos_y), 3)
env.set_initial_value(Dot(a9, pos_i), 0)  # 83
env.set_initial_value(Dot(a9, pos_j), 0)
# a5.set_initial_position_i_j(0, 0)

a10.add_public_fluent(pos_x)
a10.add_public_fluent(pos_y)
a10.add_public_fluent(pos, default_initial_value=False)
a10.add_state_encoder(StateEncoderMAPRL(a10))
a10.add_public_fluent(pos_i)
a10.add_public_fluent(pos_j)
env.add_agent(a10)
env.set_initial_value(Dot(a10, pos_x), 4)  # 98
env.set_initial_value(Dot(a10, pos_y), 7)
env.set_initial_value(Dot(a10, pos_i), 0)  # 83
env.set_initial_value(Dot(a10, pos_j), 0)
# a5.set_initial_position_i_j(0, 0)"""

env.initialize_location_mapping(coordinates)


connections = []


# Utilizzo della funzione create_wall_connections


is_connected = create_cell_connections(_connections, env)
is_wall = create_wall_connections(walls, env)


bridge = UserType("bridge")
br1 = Object("br1", bridge)
br2 = Object("br2", bridge)
br3 = Object("br3", bridge)
env.add_object(br1)
env.add_object(br2)
env.add_object(br3)
has_bridge = Fluent(
    "has_bridge", BoolType(), connect_from=Location, connect_to=Location
)
has_boat = Fluent("has_boat", BoolType(), connect_from=Location, connect_to=Location)
env.ma_environment.add_fluent(has_bridge, default_initial_value=False)
env.ma_environment.add_fluent(has_boat, default_initial_value=False)


# Setto i ponti
# Lista dei nomi desiderati
# Crea un dizionario di tutte le location:
loc_map = {loc.name: loc for loc in locations}

# Ora accedi in modo esplicito ai singoli oggetti:
l45 = loc_map["l45"]
# l78 = loc_map["l78"]
# l88 = loc_map["l88"]
# l86 = loc_map["l86"]
# l87 = loc_map["l87"]
# l66 = loc_map["l66"]
# l67 = loc_map["l67"]
# l72 = loc_map["l72"]
# l73 = loc_map["l73"]
# l17 = loc_map["l17"]
# l18 = loc_map["l18"]
# l28 = loc_map["l28"]


# Setto i ponti
env.set_initial_value(has_bridge(l13, l14), True)  # TODO attenzione qui
env.set_initial_value(has_bridge(l14, l13), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l13, l14), False)
env.set_initial_value(is_connected(l14, l13), False)

env.set_initial_value(has_boat(l14, l24), True)  # TODO attenzione qui
env.set_initial_value(has_boat(l24, l14), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l14, l24), False)
env.set_initial_value(is_connected(l24, l14), False)

# TODO MAZE
env.set_initial_value(has_bridge(l31, l41), True)  # TODO attenzione qui
env.set_initial_value(has_bridge(l41, l31), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l31, l41), False)
env.set_initial_value(is_connected(l41, l31), False)

env.set_initial_value(has_boat(l41, l42), True)  # TODO attenzione qui
env.set_initial_value(has_boat(l42, l41), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l41, l42), False)
env.set_initial_value(is_connected(l42, l41), False)

env.set_initial_value(has_bridge(l25, l35), True)  # TODO attenzione qui
env.set_initial_value(has_bridge(l35, l25), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l25, l35), False)
env.set_initial_value(is_connected(l35, l25), False)

env.set_initial_value(has_bridge(l35, l45), True)  # TODO attenzione qui
env.set_initial_value(has_bridge(l45, l35), True)  # TODO attenzione qui
env.set_initial_value(is_connected(l35, l45), False)
env.set_initial_value(is_connected(l45, l35), False)


# l17, l18, l28
"""env.set_initial_value(has_bridge(l78, l88), True) #TODO attenzione qui
env.set_initial_value(has_bridge(l88, l78), True) #TODO attenzione qui
env.set_initial_value(is_connected(l78, l88), False)
env.set_initial_value(is_connected(l88, l78), False)

env.set_initial_value(has_boat(l86, l87), True) #TODO attenzione qui
env.set_initial_value(has_boat(l87, l86), True) #TODO attenzione qui
env.set_initial_value(is_connected(l86, l87), False)
env.set_initial_value(is_connected(l87, l86), False)

env.set_initial_value(has_bridge(l66, l67), True) #TODO attenzione qui
env.set_initial_value(has_bridge(l67, l66), True) #TODO attenzione qui
env.set_initial_value(is_connected(l66, l67), False)
env.set_initial_value(is_connected(l67, l66), False)

env.set_initial_value(has_bridge(l72, l73), True) #TODO attenzione qui 6 agenti
env.set_initial_value(has_bridge(l73, l72), True) #TODO attenzione qui 6 agenti
env.set_initial_value(is_connected(l72, l73), False)
env.set_initial_value(is_connected(l73, l72), False)

env.set_initial_value(has_boat(l17, l18), True) #TODO attenzione qui 2 agenti
env.set_initial_value(has_boat(l18, l17), True) #TODO attenzione qui 2 agenti
env.set_initial_value(is_connected(l17, l18), False)
env.set_initial_value(is_connected(l18, l17), False)

env.set_initial_value(has_boat(l28, l18), True) #TODO attenzione qui 2 agenti
env.set_initial_value(has_boat(l18, l28), True) #TODO attenzione qui 2 agenti
env.set_initial_value(is_connected(l18, l28), False)
env.set_initial_value(is_connected(l28, l18), False)"""


# 10x10 griglia:
# env.set_initial_value(is_connected(l14, l15), False)
# env.set_initial_value(is_connected(l15, l14), False)

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

a6.add_rl_action(move_up)
a7.add_rl_action(move_up)
a8.add_rl_action(move_up)
# a9.add_rl_action(move_up)
# a10.add_rl_action(move_up)

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

a6.add_rl_action(move_down)
a7.add_rl_action(move_down)
a8.add_rl_action(move_down)
# a9.add_rl_action(move_down)
# a10.add_rl_action(move_down)

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

a6.add_rl_action(move_left)
a7.add_rl_action(move_left)
a8.add_rl_action(move_left)
# a9.add_rl_action(move_left)
# a10.add_rl_action(move_left)

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

a6.add_rl_action(move_right)
a7.add_rl_action(move_right)
a8.add_rl_action(move_right)
# a9.add_rl_action(move_right)
# a10.add_rl_action(move_right)


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
cross_up.add_precondition(has_bridge(l_from, l_to))
cross_up.add_precondition(Equals(pos_j, 0))
cross_up.add_decrease_effect(pos_y, 1)
cross_up.add_effect(pos_j, env.cell_size - 1)
cross_up.add_effect(pos(l_to), True)
cross_up.add_effect(pos(l_from), False)
# cross_up.add_effect(has_bridge(l_from, l_to), False)
# cross_up.add_effect(pos_x, env.get_coordinates_by_location(a1, l_to)[0], True)
# cross_up.add_effect(pos_y, env.get_coordinates_by_location(a1, l_to)[1], True)

cross_down = InstantaneousAction("cross_down", l_from=Location, l_to=Location)
cross_down.add_precondition(LT(pos_y, max_y_value - 1))
cross_down.add_precondition(has_bridge(l_from, l_to))
cross_down.add_precondition(Equals(pos_j, env.cell_size - 1))
cross_down.add_increase_effect(pos_y, 1)
cross_down.add_effect(pos_j, 0)
cross_down.add_effect(pos(l_to), True)
cross_down.add_effect(pos(l_from), False)
# cross_down.add_effect(has_bridge(l_from, l_to), False)

cross_right = InstantaneousAction("cross_right", l_from=Location, l_to=Location)
cross_right.add_precondition(LT(pos_x, max_x_value - 1))
cross_right.add_precondition(has_bridge(l_from, l_to))
cross_right.add_precondition(Equals(pos_i, env.cell_size - 1))
cross_right.add_increase_effect(pos_x, 1)
cross_right.add_effect(pos_i, 0)
cross_right.add_effect(pos(l_to), True)
cross_right.add_effect(pos(l_from), False)
# cross_right.add_effect(has_bridge(l_from, l_to), False)

cross_left = InstantaneousAction("cross_left", l_from=Location, l_to=Location)
cross_left.add_precondition(LT(0, pos_x))
cross_left.add_precondition(has_bridge(l_from, l_to))
cross_left.add_precondition(Equals(pos_i, 0))
cross_left.add_decrease_effect(pos_x, 1)
cross_left.add_effect(pos_i, env.cell_size - 1)
cross_left.add_effect(pos(l_to), True)
cross_left.add_effect(pos(l_from), False)
# cross_left.add_effect(has_bridge(l_from, l_to), False)

wait = InstantaneousAction("wait", l_from=Location, l_to=Location)
wait.add_decrease_effect(pos_x, 0)

row_up = InstantaneousAction("row_up", l_from=Location, l_to=Location)
row_up.add_precondition(LT(0, pos_y))
row_up.add_precondition(has_boat(l_from, l_to))
row_up.add_effect(pos_i, env.cell_size - 1)
row_up.add_decrease_effect(pos_y, 1)
row_up.add_effect(pos_j, env.cell_size - 1)
# row_up.add_effect(has_boat(l_from, l_to), False)
row_up.add_effect(pos(l_to), True)
row_up.add_effect(pos(l_from), False)


row_down = InstantaneousAction("row_down", l_from=Location, l_to=Location)
row_down.add_precondition(LT(pos_y, max_y_value - 1))
row_down.add_precondition(has_boat(l_from, l_to))
row_down.add_precondition(Equals(pos_j, env.cell_size - 1))
row_down.add_increase_effect(pos_y, 1)
row_down.add_effect(pos_j, 0)
# row_down.add_effect(has_boat(l_from, l_to), False)
row_down.add_effect(pos(l_to), True)
row_down.add_effect(pos(l_from), False)

row_right = InstantaneousAction("row_right", l_from=Location, l_to=Location)
row_right.add_precondition(LT(pos_x, max_x_value - 1))
row_right.add_precondition(has_boat(l_from, l_to))
row_right.add_precondition(Equals(pos_i, env.cell_size - 1))
row_right.add_increase_effect(pos_x, 1)
row_right.add_effect(pos_i, 0)
# row_right.add_effect(has_boat(l_from, l_to), False)
row_right.add_effect(pos(l_to), True)
row_right.add_effect(pos(l_from), False)

row_left = InstantaneousAction("row_left", l_from=Location, l_to=Location)
row_left.add_precondition(LT(0, pos_x))
row_left.add_precondition(has_boat(l_from, l_to))
row_left.add_precondition(Equals(pos_i, 0))
row_left.add_decrease_effect(pos_x, 1)
row_left.add_effect(pos_i, env.cell_size - 1)
# row_right.add_effect(has_boat(l_from, l_to), False)
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

a6.add_rl_action(low_up)
a6.add_rl_action(low_down)
a6.add_rl_action(low_left)
a6.add_rl_action(low_right)
a6.add_rl_action(cross_up)
a6.add_rl_action(cross_down)
a6.add_rl_action(cross_right)
a6.add_rl_action(cross_left)
a6.add_rl_action(wait)
"""a6.add_rl_action(row_up)
a6.add_rl_action(row_down)
a6.add_rl_action(row_right)
a6.add_rl_action(row_left)"""

a7.add_rl_action(low_up)
a7.add_rl_action(low_down)
a7.add_rl_action(low_left)
a7.add_rl_action(low_right)
a7.add_rl_action(cross_up)
a7.add_rl_action(cross_down)
a7.add_rl_action(cross_right)
a7.add_rl_action(cross_left)
a7.add_rl_action(wait)
"""a7.add_rl_action(row_up)
a7.add_rl_action(row_down)
a7.add_rl_action(row_right)
a7.add_rl_action(row_left)"""

a8.add_rl_action(low_up)
a8.add_rl_action(low_down)
a8.add_rl_action(low_left)
a8.add_rl_action(low_right)
a8.add_rl_action(cross_up)
a8.add_rl_action(cross_down)
a8.add_rl_action(cross_right)
a8.add_rl_action(cross_left)
a8.add_rl_action(wait)
"""a8.add_rl_action(row_up)
a8.add_rl_action(row_down)
a8.add_rl_action(row_right)
a8.add_rl_action(row_left)"""

"""a9.add_rl_action(low_up)
a9.add_rl_action(low_down)
a9.add_rl_action(low_left)
a9.add_rl_action(low_right)
a9.add_rl_action(cross_up)
a9.add_rl_action(cross_down)
a9.add_rl_action(cross_right)
a9.add_rl_action(cross_left)
a9.add_rl_action(wait)
a9.add_rl_action(row_up)
a9.add_rl_action(row_down)
a9.add_rl_action(row_right)
a9.add_rl_action(row_left)

a10.add_rl_action(low_up)
a10.add_rl_action(low_down)
a10.add_rl_action(low_left)
a10.add_rl_action(low_right)
#a10.add_rl_action(cross_up)
#a10.add_rl_action(cross_down)
#a10.add_rl_action(cross_right)
#a10.add_rl_action(cross_left)
a10.add_rl_action(wait)
a10.add_rl_action(row_up)
a10.add_rl_action(row_down)
a10.add_rl_action(row_right)
a10.add_rl_action(row_left)"""

# Sequenza di azioni concorrenti
transitions_ag_1 = RM_dict_true_seq["a1"]
transitions_ag_2 = RM_dict_true_seq["a2"]
transitions_ag_3 = RM_dict_true_seq["a3"]
transitions_ag_4 = RM_dict_true_seq["a4"]
transitions_ag_5 = RM_dict_true_seq["a5"]
# breakpoint


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
    ("state3", ((goals["D"], True),)): ("state4", 0),
}

new_transitions_ag_2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((goals["B"], True),)): ("state4", 0),
}

new_transitions_ag_3 = {
    ("state1", ((goals["C"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((coordinates_obj["coffee"][0], True),)): ("state4", 0),
    ("state3", ((coordinates_obj["coffee"][1], True),)): ("state4", 0),
}

new_transitions_ag_4 = {
    ("state1", ((goals["D"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["coffee"][0], True),)): ("state3", 0),
    ("state2", ((coordinates_obj["coffee"][1], True),)): ("state3", 0),
    ("state3", ((coordinates_obj["letter"][0], True),)): ("state4", 0),
}

new_transitions_ag_5 = {
    ("state1", ((goals["C"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["letter"][0], True),)): ("state3", 0),
    ("state3", ((coordinates_obj["coffee"][0], True),)): ("state4", 0),
    ("state3", ((coordinates_obj["coffee"][1], True),)): ("state4", 0),
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

# TODO ag8 - maze exp1
transitions_ag_1 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a3", "pos(l13)"), True), ("pos(l13)", True), (("a4", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
transitions_ag_2 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        (
            (("a5", "pos(l31)"), True),
            ("pos(l31)", True),
        ),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}
transitions_ag_3 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a1", "pos(l13)"), True), ("pos(l13)", True), (("a4", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
transitions_ag_4 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a1", "pos(l13)"), True), ("pos(l13)", True), (("a3", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
transitions_ag_5 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        (
            (("a2", "pos(l31)"), True),
            ("pos(l31)", True),
        ),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}

transitions_ag_6 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a7", "pos(l31)"), True), ("pos(l31)", True), (("a8", "pos(l31)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}
transitions_ag_7 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a6", "pos(l31)"), True), ("pos(l31)", True), (("a8", "pos(l31)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}
transitions_ag_8 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a7", "pos(l31)"), True), ("pos(l31)", True), (("a6", "pos(l31)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}
# transitions_ag_9 = {('state1', (("pos(l13)", True),)): ('state2X', 10), ('state2X', ((('a7', "pos(l13)"), True), (('a8', "pos(l13)"), True), ("pos(l13)", True))): ('state3', 20), ('state3', (("pos(l14)", True),)): ('state4', 30)}
# transitions_ag_10 = {('state1', (("pos(l24)", True),)): ('state1X', 10), ('state1X', ((('a6', "pos(l24)"), True), ("pos(l24)", True))): ('state2', 20), ('state2', (("pos(l14)", True),)): ('state3', 30)}


# TODO exp2
new_transitions_ag_exp2_PRE = {
    ("state1", ((goals["D"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["coffee"][0], True),)): ("state3", 0),
    ("state2", ((coordinates_obj["coffee"][1], True),)): ("state3", 0),
    ("state2", ((coordinates_obj["coffee"][2], True),)): ("state3", 0),
}
"""new_transitions_ag_exp2_POST = {
    ("state10", ((coordinates_obj["letter"][0], True),)): ("state11", 0),
}"""

# TODO exp3 ag:7/8/9 cross/up_stone in B insieme agli ag:1/3/4
new_transitions_ag_1_exp3 = {
    ("state12", (("pos(l72)", True),)): ("state13X", 20),
    (
        "state13X",
        (
            (("a8", "pos(l72)"), True),
            ("pos(l72)", True),
            (("a9", "pos(l72)"), True),
            (("a7", "pos(l72)"), True),
            (("a3", "pos(l72)"), True),
            (("a4", "pos(l72)"), True),
        ),
    ): ("state14", 30),
    ("state14", (("pos(l73)", True),)): ("state15", 40),
}
new_transitions_ag_2_exp3 = {
    ("state14", (("pos(l17)", True),)): ("state15X", 40),
    (
        "state15X",
        (
            ("pos(l17)", True),
            (("a5", "pos(l17)"), True),
        ),
    ): ("state16", 50),
    ("state16", (("pos(l18)", True),)): ("state17", 60),
}
new_transitions_ag_3_exp3 = {
    ("state12", (("pos(l72)", True),)): ("state13X", 20),
    (
        "state13X",
        (
            (("a8", "pos(l72)"), True),
            ("pos(l72)", True),
            (("a9", "pos(l72)"), True),
            (("a7", "pos(l72)"), True),
            (("a1", "pos(l72)"), True),
            (("a4", "pos(l72)"), True),
        ),
    ): ("state14", 30),
    ("state14", (("pos(l73)", True),)): ("state15", 40),
}
new_transitions_ag_4_exp3 = {
    ("state12", (("pos(l72)", True),)): ("state13X", 20),
    (
        "state13X",
        (
            (("a8", "pos(l72)"), True),
            ("pos(l72)", True),
            (("a9", "pos(l72)"), True),
            (("a7", "pos(l72)"), True),
            (("a3", "pos(l72)"), True),
            (("a1", "pos(l72)"), True),
        ),
    ): ("state14", 30),
    ("state14", (("pos(l73)", True),)): ("state15", 40),
}
new_transitions_ag_5_exp3 = {
    ("state11", (("pos(l17)", True),)): ("state11X", 10),
    (
        "state11X",
        (
            ("pos(l17)", True),
            (("a2", "pos(l17)"), True),
        ),
    ): ("state12", 20),
    ("state12", (("pos(l18)", True),)): ("state13", 30),
}

new_transitions_ag_6_exp3 = {
    ("state14", (("pos(l28)", True),)): ("state15X", 40),
    (
        "state15X",
        (
            ("pos(l28)", True),
            (("a10", "pos(l28)"), True),
        ),
    ): ("state16", 50),
    ("state16", (("pos(l18)", True),)): ("state17", 60),
}
new_transitions_ag_7_exp3 = {
    ("state12", (("pos(l72)", True),)): ("state13X", 20),
    (
        "state13X",
        (
            (("a8", "pos(l72)"), True),
            ("pos(l72)", True),
            (("a9", "pos(l72)"), True),
            (("a1", "pos(l72)"), True),
            (("a3", "pos(l72)"), True),
            (("a4", "pos(l72)"), True),
        ),
    ): ("state14", 30),
    ("state14", (("pos(l73)", True),)): ("state15", 40),
}
new_transitions_ag_8_exp3 = {
    ("state12", (("pos(l72)", True),)): ("state13X", 20),
    (
        "state13X",
        (
            (("a7", "pos(l72)"), True),
            ("pos(l72)", True),
            (("a9", "pos(l72)"), True),
            (("a1", "pos(l72)"), True),
            (("a3", "pos(l72)"), True),
            (("a4", "pos(l72)"), True),
        ),
    ): ("state14", 30),
    ("state14", (("pos(l73)", True),)): ("state15", 40),
}
new_transitions_ag_9_exp3 = {
    ("state12", (("pos(l72)", True),)): ("state13X", 20),
    (
        "state13X",
        (
            (("a7", "pos(l72)"), True),
            ("pos(l72)", True),
            (("a8", "pos(l72)"), True),
            (("a1", "pos(l72)"), True),
            (("a3", "pos(l72)"), True),
            (("a4", "pos(l72)"), True),
        ),
    ): ("state14", 30),
    ("state14", (("pos(l73)", True),)): ("state15", 40),
}
new_transitions_ag_10_exp3 = {
    ("state11", (("pos(l28)", True),)): ("state11X", 10),
    (
        "state11X",
        (
            ("pos(l28)", True),
            (("a6", "pos(l28)"), True),
        ),
    ): ("state12", 20),
    ("state12", (("pos(l18)", True),)): ("state13", 30),
}
# TODO###################################################################################################################


"""transitions_ag_6 ={("state2", ((("pos(l14)"), True),)): ("state4", 100)}
transitions_ag_7 ={("state2", ((("pos(l14)"), True),)): ("state4", 100)}
transitions_ag_8 ={("state2", ((("pos(l14)"), True),)): ("state4", 100)}
transitions_ag_9 ={("state2", ((("pos(l14)"), True),)): ("state4", 100)}
transitions_ag_10 ={("state2", ((("pos(l14)"), True),)): ("state4", 100)}
"""

RM_1, event_detector1 = setup_agent_rm(a1, transitions_ag_1)
RM_2, event_detector2 = setup_agent_rm(a2, transitions_ag_2)
RM_3, event_detector3 = setup_agent_rm(a3, transitions_ag_3)
RM_4, event_detector4 = setup_agent_rm(a4, transitions_ag_4)
RM_5, event_detector5 = setup_agent_rm(a5, transitions_ag_5)

RM_6, event_detector6 = setup_agent_rm(a6, transitions_ag_6)
RM_7, event_detector7 = setup_agent_rm(a7, transitions_ag_7)
RM_8, event_detector8 = setup_agent_rm(a8, transitions_ag_8)
# RM_9, event_detector9 = setup_agent_rm(a9, transitions_ag_9)
# RM_10, event_detector10 = setup_agent_rm(a10, transitions_ag_10)


# TODO EXP2 MAZE
new_transitions_ag_torcia = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
}

new_transitions_ag_remi = {
    ("state1", ((coordinates_obj["letter"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["letter"][0], True),)): ("state2", 0),
}

transitions_ag_1 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a3", "pos(l13)"), True), ("pos(l13)", True), (("a4", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
transitions_ag_2 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        (
            (("a5", "pos(l31)"), True),
            ("pos(l31)", True),
        ),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}

# TODO EXP3 MAZE
new_transitions_ag_torcia = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((goals["B"], True),)): ("state3", 0),
    ("state3", ((goals["C"], True),)): ("state4", 0),
    ("state4", ((goals["D"], True),)): ("state5", 0),
}

new_transitions_ag_remi = {
    ("state1", ((coordinates_obj["letter"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["letter"][0], True),)): ("state2", 0),
    ("state2", ((goals["B"], True),)): ("state3", 0),
    ("state3", ((goals["C"], True),)): ("state4", 0),
    ("state4", ((goals["D"], True),)): ("state5", 0),
}
transitions_ag_1_exp3 = {
    ("state11", (("pos(l25)", True),)): ("state11X", 20),
    (
        "state11X",
        (
            (("a3", "pos(l25)"), True),
            (("a4", "pos(l25)"), True),
            (("a6", "pos(l25)"), True),
            (("a7", "pos(l25)"), True),
            ("pos(l25)", True),
            (("a8", "pos(l25)"), True),
        ),
    ): ("state12", 30),
    ("state12", (("pos(l41)", True),)): ("F", 40),
}
transitions_ag_3_exp3 = {
    ("state11", (("pos(l25)", True),)): ("state11X", 20),
    (
        "state11X",
        (
            (("a1", "pos(l25)"), True),
            (("a4", "pos(l25)"), True),
            (("a6", "pos(l25)"), True),
            (("a7", "pos(l25)"), True),
            ("pos(l25)", True),
            (("a8", "pos(l25)"), True),
        ),
    ): ("state12", 30),
    ("state12", (("pos(l41)", True),)): ("F", 40),
}
transitions_ag_4_exp3 = {
    ("state11", (("pos(l25)", True),)): ("state11X", 20),
    (
        "state11X",
        (
            (("a3", "pos(l25)"), True),
            (("a1", "pos(l25)"), True),
            (("a6", "pos(l25)"), True),
            (("a7", "pos(l25)"), True),
            ("pos(l25)", True),
            (("a8", "pos(l25)"), True),
        ),
    ): ("state12", 30),
    ("state12", (("pos(l41)", True),)): ("F", 40),
}
transitions_ag_6_exp3 = {
    ("state11", (("pos(l25)", True),)): ("state11X", 20),
    (
        "state11X",
        (
            (("a3", "pos(l25)"), True),
            (("a4", "pos(l25)"), True),
            (("a1", "pos(l25)"), True),
            (("a7", "pos(l25)"), True),
            ("pos(l25)", True),
            (("a8", "pos(l25)"), True),
        ),
    ): ("state12", 30),
    ("state12", (("pos(l41)", True),)): ("F", 40),
}
transitions_ag_7_exp3 = {
    ("state11", (("pos(l25)", True),)): ("state11X", 20),
    (
        "state11X",
        (
            (("a3", "pos(l25)"), True),
            (("a4", "pos(l25)"), True),
            (("a6", "pos(l25)"), True),
            (("a1", "pos(l25)"), True),
            ("pos(l25)", True),
            (("a8", "pos(l25)"), True),
        ),
    ): ("state12", 30),
    ("state12", (("pos(l41)", True),)): ("F", 40),
}
transitions_ag_8_exp3 = {
    ("state11", (("pos(l25)", True),)): ("state11X", 20),
    (
        "state11X",
        (
            (("a3", "pos(l25)"), True),
            (("a4", "pos(l25)"), True),
            (("a6", "pos(l25)"), True),
            (("a7", "pos(l25)"), True),
            ("pos(l25)", True),
            (("a1", "pos(l25)"), True),
        ),
    ): ("state12", 30),
    ("state12", (("pos(l41)", True),)): ("F", 40),
}

"""
# Aggiungi le nuove transizioni con il collegamento
RM_1.add_transitions_with_merge(
    new_transitions_ag_torcia, position="before", prefix="new"
)
RM_2.add_transitions_with_merge(
    new_transitions_ag_remi, position="before", prefix="new"
)
RM_3.add_transitions_with_merge(
    new_transitions_ag_torcia, position="before", prefix="new"
)
RM_4.add_transitions_with_merge(
    new_transitions_ag_torcia, position="before", prefix="new"
)
RM_5.add_transitions_with_merge(
    new_transitions_ag_remi, position="before", prefix="new"
)
RM_6.add_transitions_with_merge(
    new_transitions_ag_torcia, position="before", prefix="new"
)
RM_7.add_transitions_with_merge(
    new_transitions_ag_torcia, position="before", prefix="new"
)
RM_8.add_transitions_with_merge(
    new_transitions_ag_torcia, position="before", prefix="new"
)

RM_1.add_transitions_with_merge(transitions_ag_1_exp3, position="before", prefix="new")
RM_3.add_transitions_with_merge(transitions_ag_3_exp3, position="before", prefix="new")
RM_4.add_transitions_with_merge(transitions_ag_4_exp3, position="before", prefix="new")
RM_6.add_transitions_with_merge(transitions_ag_6_exp3, position="before", prefix="new")
RM_7.add_transitions_with_merge(transitions_ag_7_exp3, position="before", prefix="new")
RM_8.add_transitions_with_merge(transitions_ag_8_exp3, position="before", prefix="new")"""

# RM_9.add_transitions_with_merge(new_transitions_ag_exp2_PRE, position="before", prefix="new")
# RM_10.add_transitions_with_merge(new_transitions_ag_exp2_PRE, position="before", prefix="new")

"""RM_1.add_transitions_with_merge(new_transitions_ag_1_exp3, position="before", prefix="new")
RM_2.add_transitions_with_merge(new_transitions_ag_2_exp3, position="before", prefix="new")
RM_3.add_transitions_with_merge(new_transitions_ag_3_exp3, position="before", prefix="new")
RM_4.add_transitions_with_merge(new_transitions_ag_4_exp3, position="before", prefix="new")
RM_5.add_transitions_with_merge(new_transitions_ag_5_exp3, position="before", prefix="new")
RM_6.add_transitions_with_merge(new_transitions_ag_6_exp3, position="before", prefix="new")
RM_7.add_transitions_with_merge(new_transitions_ag_7_exp3, position="before", prefix="new")
RM_8.add_transitions_with_merge(new_transitions_ag_8_exp3, position="before", prefix="new")
RM_9.add_transitions_with_merge(new_transitions_ag_9_exp3, position="before", prefix="new")
RM_10.add_transitions_with_merge(new_transitions_ag_10_exp3, position="before", prefix="new")"""

# TODO deccomentare


# Estrai gli eventi usando il metodo della classe
a1_all_events = RM_1.extract_events()
a2_all_events = RM_2.extract_events()
a3_all_events = RM_3.extract_events()
a4_all_events = RM_4.extract_events()
a5_all_events = RM_5.extract_events()

a6_all_events = RM_6.extract_events()
a7_all_events = RM_7.extract_events()
a8_all_events = RM_8.extract_events()
# a9_all_events = RM_9.extract_events()
# a10_all_events = RM_10.extract_events()

# Aggiungi gli eventi estratti all'EventDetector
event_detector1.add_events(a1_all_events)
event_detector2.add_events(a2_all_events)
event_detector3.add_events(a3_all_events)
event_detector4.add_events(a4_all_events)
event_detector5.add_events(a5_all_events)
event_detector6.add_events(a6_all_events)
event_detector7.add_events(a7_all_events)
event_detector8.add_events(a8_all_events)
# event_detector9.add_events(a9_all_events)
# event_detector10.add_events(a10_all_events)


a1.set_reward_machine(RM_1)
a2.set_reward_machine(RM_2)
a3.set_reward_machine(RM_3)
a4.set_reward_machine(RM_4)
a5.set_reward_machine(RM_5)

a6.set_reward_machine(RM_6)
a7.set_reward_machine(RM_7)
a8.set_reward_machine(RM_8)
# a9.set_reward_machine(RM_9)
# a10.set_reward_machine(RM_10)


# Funzione principale per eseguire l'esperimento
def run_experiment(num_episodes, wandb_enabled):
    if wandb_enabled:
        wandb.init(project="maze_RL_new", entity="alee8", mode="online")
    else:
        wandb.init(project="maze_RL_new", entity="alee8", mode="disabled")
    global NUM_EPISODES
    NUM_EPISODES = num_episodes
    # TODO deccomentare
    rm_env = RMEnvironmentWrapper(
        env, [a1, a2, a3, a4, a5, a6, a7, a8]
    )  # , a9, a10])#[a2, a5]) #[a1, a2, a3, a4, a5])
    q_learning1 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * RM_1.numbers_state(),  # env.num_rm_states,
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
        * RM_2.numbers_state(),  # env.num_rm_states,
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
        * RM_3.numbers_state(),  # env.num_rm_states,
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
        * RM_4.numbers_state(),  # env.num_rm_states,
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
        * RM_5.numbers_state(),  # env.num_rm_states,
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

    q_learning6 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * RM_6.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=47,
        qtable_init=2,
        use_qrm=True,
    )
    q_learning7 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * RM_7.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=48,
        qtable_init=2,
        use_qrm=True,
    )
    q_learning8 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * RM_8.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=49,
        qtable_init=2,
        use_qrm=True,
    )
    """q_learning9 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * RM_9.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=50,
        qtable_init=2,
        use_qrm=True,
    )
    q_learning10 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * RM_10.numbers_state(),  # env.num_rm_states,
        action_space_size=13,
        learning_rate=0.5,
        gamma=0.9,
        action_selection="greedy",
        epsilon_start=0.1,
        epsilon_end=0.1,
        epsilon_decay=0.9995,
        seed=51,
        qtable_init=2,
        use_qrm=True,
    )"""

    a1.set_learning_algorithm(q_learning1)
    a2.set_learning_algorithm(q_learning2)
    a3.set_learning_algorithm(q_learning3)
    a4.set_learning_algorithm(q_learning4)
    a5.set_learning_algorithm(q_learning5)

    a6.set_learning_algorithm(q_learning6)
    a7.set_learning_algorithm(q_learning7)
    a8.set_learning_algorithm(q_learning8)
    # a9.set_learning_algorithm(q_learning9)
    # a10.set_learning_algorithm(q_learning10)

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
        test_episode = episode % 100 == 0

        # Imposta il flag per l'esplorazione
        if test_episode:
            exploration = False  # Usa la policy ottima
        else:
            exploration = True  # Usa la policy con esplorazione

        record_episode = episode % 1000 == 0 and episode != 0
        # record_episode = False
        if record_episode:
            renderer.render(episode, states)  # Cattura frame durante l'episodio
            actions_log = {agent.name: [] for agent in env.agents}

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
    return parser.parse_args()


# Lancia l'esperimento con i parametri dalla linea di comando
if __name__ == "__main__":
    args = parse_args()

    # Esegui l'esperimento con i parametri definiti da argparse
    run_experiment(num_episodes=args.num_episodes, wandb_enabled=args.wandb_enabled)
