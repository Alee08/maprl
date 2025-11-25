import functools
import random
import copy
import pygame
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagentplanning_rl.multi_agent.reward_machine import RewardMachine
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from multiagentplanning_rl.multi_agent.agent_rl import AgentRL
from multiagentplanning_rl.utils.ma_sequential_simulator import (
    UPSequentialSimulatorMA as SequentialSimulatorMA,
)
from multiagentplanning_rl.environments.utils_envs.evaluation_metrics import *
import json
from multiagentplanning_rl.utils.message import Message
from ma_maze_office import MAP_RL_Env
from multiagentplanning_rl.render.render import EnvironmentRenderer
from multiagentplanning_rl.environments.integration_planing_and_learning.state_encoder import (
    StateEncoderMAPRL,
)
from multiagentplanning_rl.environments.integration_planing_and_learning.detect_event_2 import (
    PositionEventDetector,
)
from multiagentplanning_rl.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)
import wandb
import random
from multiagentplanning_rl.utils.utils import (
    encode_state,
    parse_map_string,
    parse_map_emoji,
    parse_office_world,
)

import logging
import argparse

logging.basicConfig(level=logging.INFO)


NUM_EPISODES = 20000  # Number of episodes
# wandb.init(project="maze_RL_new", entity="alee8", mode="disabled")


# Be aware that the actions are set to consider a 3x3 rooms
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

MAPS = {"medium": map_2}
MAP_SELECTION = "medium"
MAP = MAPS[MAP_SELECTION]

GRID_DIMENSIONS = {"large": (8, 8, 8), "medium": (5, 5, 5), "small": (4, 4, 4)}
grid_height, grid_width, grid_size = GRID_DIMENSIONS[MAP_SELECTION]
# Parse the map
coordinates_obj, goals, walls, rooms, _connections = parse_office_world_(MAP)


def build_object_positions(coordinates, walls, extra=None):
    """Create default object locations and merge optional extra positions."""
    base_positions = {
        "plant": coordinates["plant"],
        "coffee": coordinates["coffee"],
        "letter": coordinates["letter"],
        "office_walls": walls,
    }

    if extra:
        base_positions.update(extra)

    return base_positions


object_positions = build_object_positions(
    coordinates_obj,
    walls,
    extra={"bridges": [], "boats": []},
)

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
    cell_size=100,  # Dimension in pixel of a room
    in_cell_size=env.cell_size,  # Number of subcells per dimension within the room
    resource_overrides={
        "plant": lambda renderer: (
            "img/buco_lab.png",
            (renderer.inner_cell_size, renderer.inner_cell_size),
        ),
        "coffee": lambda renderer: (
            "img/torcia.png",
            (renderer.inner_cell_size - 6, renderer.inner_cell_size - 6),
        ),
        "letter": lambda renderer: (
            "img/remi.png",
            (renderer.inner_cell_size, renderer.inner_cell_size),
        ),
    },
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

AGENT_ORDER = [
    ("a1", a1),
    ("a2", a2),
    ("a3", a3),
    ("a4", a4),
    ("a5", a5),
    ("a6", a6),
    ("a7", a7),
    ("a8", a8),
]
AGENTS_BY_LABEL = dict(AGENT_ORDER)

Location = UserType("Location")
max_x_value = env.grid_width
max_y_value = env.grid_height

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

    # Create objects for each room
    for room_name in connections:
        location_objects[room_name] = Object(room_name, Location)

    # Create the fluent is_connected
    is_connected = Fluent("is_connected", BoolType(), l1=Location, l2=Location)

    # Set up connections
    for room_name, connected_rooms in connections.items():
        for connected_room in connected_rooms:
            # Set the initial value of the connection
            env.set_initial_value(
                is_connected(
                    location_objects[room_name], location_objects[connected_room]
                ),
                True,
            )
            # Let's make the two-way connection
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

    # Create objects for each coordinate
    for wall_pair in walls:
        for coord in wall_pair:
            coord_name = f"({coord[0]},{coord[1]})"
            if coord_name not in location_objects:
                location_objects[coord_name] = Object(coord_name, Location)

    # Create the fluent is_wall
    is_wall = Fluent("is_wall", BoolType(), l1=Location, l2=Location)

    # Set the walls
    for wall_pair in walls:
        coord1_name = f"({wall_pair[0][0]},{wall_pair[0][1]})"
        coord2_name = f"({wall_pair[1][0]},{wall_pair[1][1]})"
        # Set the initial value to indicate the presence of a wall
        env.set_initial_value(
            is_wall(location_objects[coord1_name], location_objects[coord2_name]),
            True,
        )
        # Let's make the two-way wall
        env.set_initial_value(
            is_wall(location_objects[coord2_name], location_objects[coord1_name]),
            True,
        )

    return is_wall


def find_connector_between_rooms(room_a, room_b, walls_set):
    """Return the first pair of adjacent cells between two rooms without a wall."""

    room_b_cells = set(room_b)
    for x, y in room_a:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in room_b_cells and ((x, y), neighbor) not in walls_set:
                return (x, y), neighbor
    return None


def build_connectors(pairs, rooms, walls):
    """Build drawable connectors for the provided room pairs."""

    walls_set = set()
    for cell_a, cell_b in walls:
        walls_set.add((cell_a, cell_b))
        walls_set.add((cell_b, cell_a))

    connectors = []
    for room_a, room_b in pairs:
        if room_a not in rooms or room_b not in rooms:
            continue

        connector_cells = find_connector_between_rooms(
            rooms[room_a], rooms[room_b], walls_set
        )
        if connector_cells:
            connectors.append(connector_cells)

    return connectors


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

    # If we are not in play mode, we log data to wandb
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
    success_per_agent = {agent.name: 0 for agent in agents}
    rewards_per_episode = {agent.name: [] for agent in agents}
    actions_log = {agent.name: [] for agent in agents}
    moving_average_window = 1000
    return (
        success_per_agent,
        rewards_per_episode,
        actions_log,
        moving_average_window,
    )


def log_wandb_data(
    rm_env,
    episode,
    rewards_agents,
    success_per_agent,
    rewards_per_episode,
    moving_average_window,
    total_step,
    training_steps,
):
    """
    Logs data to Weights & Biases during the experiment, including successes, rewards, and total steps.

    :param rm_env: The Reward Machine environment instance
    :param episode: The current episode number
    :param rewards_agents: Rewards obtained by each agent
    :param success_per_agent: Success count per agent
    :param rewards_per_episode: Rewards per episode
    :param moving_average_window: Moving average window size
    :param total_step: Total steps taken in the current run
    """
    log_data = prepare_log_data(
        rm_env.env,
        episode,
        rewards_agents,
        success_per_agent,
        rewards_per_episode,
        moving_average_window,
    )
    log_data.update(
        {
            f"Steps_total": total_step,
            "Training_steps": training_steps,
        }
    )

    wandb.log(log_data, step=episode)


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

env.initialize_location_mapping(coordinates)


connections = []
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


# Create a dictionary of all locations:
loc_map = {loc.name: loc for loc in locations}


l45 = loc_map["l45"]


# Configure bridge availability
env.set_initial_value(has_bridge(l13, l14), True)
env.set_initial_value(has_bridge(l14, l13), True)
env.set_initial_value(is_connected(l13, l14), False)
env.set_initial_value(is_connected(l14, l13), False)

env.set_initial_value(has_boat(l14, l24), True)
env.set_initial_value(has_boat(l24, l14), True)
env.set_initial_value(is_connected(l14, l24), False)
env.set_initial_value(is_connected(l24, l14), False)

# TODO: refine maze bridge and boat layout
env.set_initial_value(has_bridge(l31, l41), True)
env.set_initial_value(has_bridge(l41, l31), True)
env.set_initial_value(is_connected(l31, l41), False)
env.set_initial_value(is_connected(l41, l31), False)

env.set_initial_value(has_boat(l41, l42), True)
env.set_initial_value(has_boat(l42, l41), True)
env.set_initial_value(is_connected(l41, l42), False)
env.set_initial_value(is_connected(l42, l41), False)

env.set_initial_value(has_bridge(l25, l35), True)
env.set_initial_value(has_bridge(l35, l25), True)
env.set_initial_value(is_connected(l25, l35), False)
env.set_initial_value(is_connected(l35, l25), False)

env.set_initial_value(has_bridge(l35, l45), True)
env.set_initial_value(has_bridge(l45, l35), True)
env.set_initial_value(is_connected(l35, l45), False)
env.set_initial_value(is_connected(l45, l35), False)


bridge_pairs = {("l13", "l14"), ("l31", "l41"), ("l25", "l35"), ("l35", "l45")}
boat_pairs = {("l14", "l24"), ("l41", "l42")}

renderer.object_positions["bridges"] = build_connectors(bridge_pairs, rooms, walls)
renderer.object_positions["boats"] = build_connectors(boat_pairs, rooms, walls)


env.ma_environment.add_fluent(is_connected, default_initial_value=False)
# Action: move up between rooms
move_up = InstantaneousAction("up", l_from=Location, l_to=Location)
l_from = move_up.parameter("l_from")
l_to = move_up.parameter("l_to")
move_up.add_precondition(LT(0, pos_y))  # Precondition: pos_y > 0
move_up.add_precondition(is_connected(l_from, l_to))
move_up.add_precondition(Equals(pos_j, 0))
move_up.add_decrease_effect(pos_y, 1)
move_up.add_effect(pos(l_to), True)
move_up.add_effect(pos(l_from), False)
move_up.add_effect(pos_j, env.cell_size - 1)

a1.add_rl_action(move_up)
a2.add_rl_action(move_up)
a3.add_rl_action(move_up)
a4.add_rl_action(move_up)
a5.add_rl_action(move_up)

a6.add_rl_action(move_up)
a7.add_rl_action(move_up)
a8.add_rl_action(move_up)

# Action: move down between rooms
move_down = InstantaneousAction("down", l_from=Location, l_to=Location)
move_down.add_precondition(
    LT(pos_y, max_y_value - 1)
)  # Precondition: pos_y < max_y_value
move_down.add_precondition(is_connected(l_from, l_to))
move_down.add_precondition(Equals(pos_j, env.cell_size - 1))
move_down.add_increase_effect(pos_y, 1)
move_down.add_effect(pos(l_to), True)
move_down.add_effect(pos(l_from), False)
move_down.add_effect(pos_j, 0)

a1.add_rl_action(move_down)
a2.add_rl_action(move_down)
a3.add_rl_action(move_down)
a4.add_rl_action(move_down)
a5.add_rl_action(move_down)

a6.add_rl_action(move_down)
a7.add_rl_action(move_down)
a8.add_rl_action(move_down)

# Action: move left between rooms
move_left = InstantaneousAction("left", l_from=Location, l_to=Location)
move_left.add_precondition(LT(0, pos_x))  # Precondition: pos_x > 0
move_left.add_precondition(is_connected(l_from, l_to))
move_left.add_precondition(Equals(pos_i, 0))
move_left.add_effect(pos(l_to), True)
move_left.add_effect(pos(l_from), False)
move_left.add_effect(pos_i, env.cell_size - 1)
move_left.add_decrease_effect(pos_x, 1)
a1.add_rl_action(move_left)
a2.add_rl_action(move_left)
a3.add_rl_action(move_left)
a4.add_rl_action(move_left)
a5.add_rl_action(move_left)

a6.add_rl_action(move_left)
a7.add_rl_action(move_left)
a8.add_rl_action(move_left)

# Action: move right between rooms
move_right = InstantaneousAction("right", l_from=Location, l_to=Location)
move_right.add_precondition(
    LT(pos_x, max_x_value - 1)
)  # Precondition: pos_x < max_x_value
move_right.add_precondition(is_connected(l_from, l_to))
move_right.add_precondition(Equals(pos_i, env.cell_size - 1))
move_right.add_effect(pos(l_to), True)
move_right.add_effect(pos(l_from), False)
move_right.add_effect(pos_i, 0)
move_right.add_increase_effect(pos_x, 1)

a1.add_rl_action(move_right)
a2.add_rl_action(move_right)
a3.add_rl_action(move_right)
a4.add_rl_action(move_right)
a5.add_rl_action(move_right)

a6.add_rl_action(move_right)
a7.add_rl_action(move_right)
a8.add_rl_action(move_right)


low_up = InstantaneousAction("low_up", l_from=Location, l_to=Location)
low_up.add_precondition(LT(0, pos_j))  # right > left
low_up.add_decrease_effect(pos_j, 1)

low_down = InstantaneousAction("low_down", l_from=Location, l_to=Location)
low_down.add_precondition(LT(pos_j, env.cell_size - 1))
low_down.add_increase_effect(pos_j, 1)

low_left = InstantaneousAction("low_left", l_from=Location, l_to=Location)
low_left.add_precondition(LT(0, pos_i))
low_left.add_decrease_effect(pos_i, 1)

low_right = InstantaneousAction("low_right", l_from=Location, l_to=Location)
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

cross_down = InstantaneousAction("cross_down", l_from=Location, l_to=Location)
cross_down.add_precondition(LT(pos_y, max_y_value - 1))
cross_down.add_precondition(has_bridge(l_from, l_to))
cross_down.add_precondition(Equals(pos_j, env.cell_size - 1))
cross_down.add_increase_effect(pos_y, 1)
cross_down.add_effect(pos_j, 0)
cross_down.add_effect(pos(l_to), True)
cross_down.add_effect(pos(l_from), False)

cross_right = InstantaneousAction("cross_right", l_from=Location, l_to=Location)
cross_right.add_precondition(LT(pos_x, max_x_value - 1))
cross_right.add_precondition(has_bridge(l_from, l_to))
cross_right.add_precondition(Equals(pos_i, env.cell_size - 1))
cross_right.add_increase_effect(pos_x, 1)
cross_right.add_effect(pos_i, 0)
cross_right.add_effect(pos(l_to), True)
cross_right.add_effect(pos(l_from), False)

cross_left = InstantaneousAction("cross_left", l_from=Location, l_to=Location)
cross_left.add_precondition(LT(0, pos_x))
cross_left.add_precondition(has_bridge(l_from, l_to))
cross_left.add_precondition(Equals(pos_i, 0))
cross_left.add_decrease_effect(pos_x, 1)
cross_left.add_effect(pos_i, env.cell_size - 1)
cross_left.add_effect(pos(l_to), True)
cross_left.add_effect(pos(l_from), False)

wait = InstantaneousAction("wait", l_from=Location, l_to=Location)
wait.add_decrease_effect(pos_x, 0)

row_up = InstantaneousAction("row_up", l_from=Location, l_to=Location)
row_up.add_precondition(LT(0, pos_y))
row_up.add_precondition(has_boat(l_from, l_to))
row_up.add_effect(pos_i, env.cell_size - 1)
row_up.add_decrease_effect(pos_y, 1)
row_up.add_effect(pos_j, env.cell_size - 1)
row_up.add_effect(pos(l_to), True)
row_up.add_effect(pos(l_from), False)


row_down = InstantaneousAction("row_down", l_from=Location, l_to=Location)
row_down.add_precondition(LT(pos_y, max_y_value - 1))
row_down.add_precondition(has_boat(l_from, l_to))
row_down.add_precondition(Equals(pos_j, env.cell_size - 1))
row_down.add_increase_effect(pos_y, 1)
row_down.add_effect(pos_j, 0)
row_down.add_effect(pos(l_to), True)
row_down.add_effect(pos(l_from), False)

row_right = InstantaneousAction("row_right", l_from=Location, l_to=Location)
row_right.add_precondition(LT(pos_x, max_x_value - 1))
row_right.add_precondition(has_boat(l_from, l_to))
row_right.add_precondition(Equals(pos_i, env.cell_size - 1))
row_right.add_increase_effect(pos_x, 1)
row_right.add_effect(pos_i, 0)
row_right.add_effect(pos(l_to), True)
row_right.add_effect(pos(l_from), False)

row_left = InstantaneousAction("row_left", l_from=Location, l_to=Location)
row_left.add_precondition(LT(0, pos_x))
row_left.add_precondition(has_boat(l_from, l_to))
row_left.add_precondition(Equals(pos_i, 0))
row_left.add_decrease_effect(pos_x, 1)
row_left.add_effect(pos_i, env.cell_size - 1)
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

a2.add_rl_action(low_up)
a2.add_rl_action(low_down)
a2.add_rl_action(low_left)
a2.add_rl_action(low_right)
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

a4.add_rl_action(low_up)
a4.add_rl_action(low_down)
a4.add_rl_action(low_left)
a4.add_rl_action(low_right)
a4.add_rl_action(cross_up)
a4.add_rl_action(cross_down)
a4.add_rl_action(cross_right)
a4.add_rl_action(cross_left)
a4.add_rl_action(wait)

a5.add_rl_action(low_up)
a5.add_rl_action(low_down)
a5.add_rl_action(low_left)
a5.add_rl_action(low_right)
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

# States for experiment: IQL exp2
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

# States for experiment: IQL exp1
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

# States for experiment: IQL exp 0, 5agents (only MAP)
transitions_ag5_ag2_exp0 = {
    ("state2", ((("pos(l14)"), True),)): ("state4", 100),
}
transitions_ag1_ag3_ag4_exp0 = {
    ("state2", ((("pos(l14)"), True),)): ("state4", 100),
}

# States for experiment: ag8 - maze exp1
exp1_transitions_ag_1 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a3", "pos(l13)"), True), ("pos(l13)", True), (("a4", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
exp1_transitions_ag_2 = {
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
exp1_transitions_ag_3 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a1", "pos(l13)"), True), ("pos(l13)", True), (("a4", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
exp1_transitions_ag_4 = {
    ("state2", (("pos(l13)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a1", "pos(l13)"), True), ("pos(l13)", True), (("a3", "pos(l13)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l14)", True),)): ("state5", 40),
}
exp1_transitions_ag_5 = {
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

exp1_transitions_ag_6 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a7", "pos(l31)"), True), ("pos(l31)", True), (("a8", "pos(l31)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}
exp1_transitions_ag_7 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a6", "pos(l31)"), True), ("pos(l31)", True), (("a8", "pos(l31)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}
exp1_transitions_ag_8 = {
    ("state2", (("pos(l31)", True),)): ("state3X", 20),
    (
        "state3X",
        ((("a7", "pos(l31)"), True), ("pos(l31)", True), (("a6", "pos(l31)"), True)),
    ): ("state4", 30),
    ("state4", (("pos(l41)", True),)): ("state5", 40),
}


EXP1_TRANSITIONS = {
    "a1": exp1_transitions_ag_1,
    "a2": exp1_transitions_ag_2,
    "a3": exp1_transitions_ag_3,
    "a4": exp1_transitions_ag_4,
    "a5": exp1_transitions_ag_5,
    "a6": exp1_transitions_ag_6,
    "a7": exp1_transitions_ag_7,
    "a8": exp1_transitions_ag_8,
}


# States for experiment: exp2
new_transitions_ag_torcia_exp2 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
}

new_transitions_ag_remi_exp2 = {
    ("state1", ((coordinates_obj["letter"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["letter"][0], True),)): ("state2", 0),
}

new_transitions_ag_exp2_PRE = {
    ("state1", ((goals["D"], True),)): ("state2", 0),
    ("state2", ((coordinates_obj["coffee"][0], True),)): ("state3", 0),
    ("state2", ((coordinates_obj["coffee"][1], True),)): ("state3", 0),
    ("state2", ((coordinates_obj["coffee"][2], True),)): ("state3", 0),
}
"""new_transitions_ag_exp2_POST = {
    ("state10", ((coordinates_obj["letter"][0], True),)): ("state11", 0),
}"""

# States for experiment: exp3 ag:7/8/9 cross/up_stone in B together with ag:1/3/4
new_transitions_ag_torcia_exp3 = {
    ("state1", ((coordinates_obj["coffee"][0], True),)): ("state2", 0),
    ("state1", ((coordinates_obj["coffee"][1], True),)): ("state2", 0),
    ("state2", ((goals["B"], True),)): ("state3", 0),
    ("state3", ((goals["C"], True),)): ("state4", 0),
    ("state4", ((goals["D"], True),)): ("state5", 0),
}

new_transitions_ag_remi_exp3 = {
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


def initialize_reward_machines(experiment):
    """Build Reward Machines for each agent and enrich them per experiment."""
    rm_event_pairs = {}
    for agent_label, agent in AGENT_ORDER:
        base_transitions = copy.deepcopy(EXP1_TRANSITIONS[agent_label])
        rm, event_detector = setup_agent_rm(agent, base_transitions)
        rm_event_pairs[agent_label] = (rm, event_detector)

    if experiment in {"exp2", "exp3"}:
        for agent_label in ["a1", "a3", "a4", "a6", "a7", "a8"]:
            rm_event_pairs[agent_label][0].add_transitions_with_merge(
                new_transitions_ag_torcia_exp2, position="before", prefix="exp2"
            )
        for agent_label in ["a2", "a5"]:
            rm_event_pairs[agent_label][0].add_transitions_with_merge(
                new_transitions_ag_remi_exp2, position="before", prefix="exp2"
            )
        rm_event_pairs["a2"][0].add_transitions_with_merge(
            a2_new_transitions_ag_5_and_ag2_exp2, position="before", prefix="exp2"
        )
        rm_event_pairs["a5"][0].add_transitions_with_merge(
            a5_new_transitions_ag_5_and_ag2_exp2, position="before", prefix="exp2"
        )
        rm_event_pairs["a2"][0].add_transitions_with_merge(
            transitions_ag_2_exp2, position="before", prefix="exp2"
        )
        rm_event_pairs["a5"][0].add_transitions_with_merge(
            transitions_ag_5_exp2, position="before", prefix="exp2"
        )

    if experiment == "exp3":
        for agent_label in ["a1", "a3", "a4", "a6", "a7", "a8"]:
            rm_event_pairs[agent_label][0].add_transitions_with_merge(
                new_transitions_ag_torcia_exp3, position="before", prefix="exp3"
            )
        for agent_label in ["a2", "a5"]:
            rm_event_pairs[agent_label][0].add_transitions_with_merge(
                new_transitions_ag_remi_exp3, position="before", prefix="exp3"
            )

        exp3_exit_transitions = {
            "a1": transitions_ag_1_exp3,
            "a3": transitions_ag_3_exp3,
            "a4": transitions_ag_4_exp3,
            "a6": transitions_ag_6_exp3,
            "a7": transitions_ag_7_exp3,
            "a8": transitions_ag_8_exp3,
        }
        for agent_label, transitions in exp3_exit_transitions.items():
            rm_event_pairs[agent_label][0].add_transitions_with_merge(
                transitions, position="before", prefix="exp3_exit"
            )

    for agent_label, (rm, event_detector) in rm_event_pairs.items():
        event_detector.add_events(rm.extract_events())
        rm.event_detector = event_detector
        AGENTS_BY_LABEL[agent_label].set_reward_machine(rm)

    return {agent_label: rm for agent_label, (rm, _) in rm_event_pairs.items()}


# Funzione principale per eseguire l'esperimento
def run_experiment(num_episodes, wandb_enabled, experiment):
    """Train agents for the selected experiment while logging to WandB if enabled."""
    if wandb_enabled:
        wandb.init(project="maze_RL_new", entity="alee8", mode="online")
    else:
        wandb.init(project="maze_RL_new", entity="alee8", mode="disabled")
    global NUM_EPISODES
    NUM_EPISODES = num_episodes
    reward_machines = initialize_reward_machines(experiment)
    # TODO deccomentare
    rm_env = RMEnvironmentWrapper(
        env, [a1, a2, a3, a4, a5, a6, a7, a8]
    )  # , a9, a10])#[a2, a5]) #[a1, a2, a3, a4, a5])
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

    q_learning6 = QLearning(
        state_space_size=env.grid_width
        * env.grid_height
        * env.cell_size
        * env.cell_size
        * reward_machines["a6"].numbers_state(),  # env.num_rm_states,
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
        * reward_machines["a7"].numbers_state(),  # env.num_rm_states,
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
        * reward_machines["a8"].numbers_state(),  # env.num_rm_states,
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

    success_per_agent = {agent.name: 0 for agent in env.agents}
    rewards_per_episode = {agent.name: [] for agent in env.agents}
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

        # Determine if this is a test episode
        test_episode = episode % 100 == 0

        # Set the flag for exploration
        if test_episode:
            exploration = False  # Use optimal policy
        else:
            exploration = True  # Use policy with exploration

        record_episode = episode % 1000 == 0 and episode != 0
        if record_episode:
            renderer.render(episode, states)  # Capture frames during the episode
        actions_log = {agent.name: [] for agent in env.agents}

        while any(rm_env.env.active_agents.values()):
            total_training_steps += 1
            episode_total_steps += 1
            actions = {}
            rewards = {a.name: 0 for a in rm_env.agents}
            infos = {a.name: {} for a in rm_env.agents}
            for ag in rm_env.agents:
                if not rm_env.env.active_agents.get(ag.name, True):
                    continue  # Skip agents that were already inactive
                current_state = rm_env.env.get_state(ag)
                action = ag.select_action(current_state, best=not exploration)
                actions[ag.name] = action

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
                    continue  # Skip inactive agents that have not just terminated

                # Update the step count for each agent
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

            for agent in rm_env.agents:
                if done.get(agent.name, False):
                    rm_env.env.active_agents[agent.name] = False

            states = copy.deepcopy(new_states)
            if record_episode:
                renderer.render(episode, states)  # Capture frames during the episode

            if all(truncations.values() or done.values()):
                break

        if record_episode:
            renderer.save_episode(episode)  # Total steps to complete the episode

        if test_episode and wandb_enabled:
            log_data = {
                "total_steps_episode": episode_total_steps,  # Total steps to complete the episode
                "training_steps": total_training_steps,
            }
            log_test_episode_data(
                rm_env, total_steps_per_agent, rewards_agents, episode
            )

            wandb.log(log_data, step=episode)

        epsilon_str = get_epsilon_summary(rm_env.agents)

        logging.info(
            f"Episode {episode + 1}: Reward = {rewards_agents}, Total Steps: {total_step + 1}, Episode Step: {rm_env.env.timestep}, Agents Step = {rm_env.env.agent_steps}, Epsilon agents= [{epsilon_str}]"
        )
    wandb.finish()

    # Save the action log and Q-tables to a JSON file
    with open("final_episode_log.json", "w") as f:
        json.dump({"actions_log": actions_log, "q_tables": q_tables}, f, indent=4)


# Set argparse to handle the command line
def parse_args():
    """Define and parse command-line options for running maze experiments."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent experiments on the maze RL environment"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20000,
        help="Number of episodes to run the learning loop",
    )
    parser.add_argument(
        "--wandb_enabled", action="store_true", help="Enable sending logs to WandB"
    )
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "exp3"],
        default="exp1",
        help="Select which task to execute",
    )
    return parser.parse_args()


# Run the experiment with parameters from the command line
if __name__ == "__main__":
    args = parse_args()

    # Run the experiment with the parameters defined by argparse
    run_experiment(
        num_episodes=args.num_episodes,
        wandb_enabled=args.wandb_enabled,
        experiment=args.experiment,
    )
