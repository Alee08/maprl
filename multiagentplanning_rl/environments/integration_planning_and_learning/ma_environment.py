from collections import defaultdict
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagentplanning_rl.multi_agent.reward_machine import RewardMachine
from multiagentplanning_rl.utils.ma_sequential_simulator import (
    UPSequentialSimulatorMA as SequentialSimulatorMA,
)
from multiagentplanning_rl.utils.message import Message
from multiagent_rlrm.multi_agent.base_environment import BaseEnvironment
from unified_planning.shortcuts import *


class MAP_RL_Env(BaseEnvironment):
    """Multi-agent Planning + RL environment with Reward Machines."""

    # The metadata holds environment constants (Gym-like).
    metadata = {
        "name": "multi_agent_planning_with_RL",
    }

    # Map domain-specific labels -> canonical labels used by the environment logic.
    # Office World uses the left-hand side; Maze / generic code uses the right-hand side.
    ACTION_ALIASES = {
        # Office world domain-specific labels
        "cross_door_up": "cross_up",
        "cross_door_down": "cross_down",
        "cross_door_left": "cross_left",
        "cross_door_right": "cross_right",
        "manager_door_up": "row_up",
        "manager_door_down": "row_down",
        "manager_door_left": "row_left",
        "manager_door_right": "row_right",
        "cell_up": "low_up",
        "cell_down": "low_down",
        "cell_left": "low_left",
        "cell_right": "low_right",
    }

    # Canonical movement deltas in (i, j) within a room (relative coordinates).
    CANONICAL_ACTION_DELTAS = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
        "cross_up": (0, -1),
        "cross_down": (0, 1),
        "cross_left": (-1, 0),
        "cross_right": (1, 0),
        "row_up": (0, -1),
        "row_down": (0, 1),
        "row_left": (-1, 0),
        "row_right": (1, 0),
        "low_up": (0, -1),
        "low_down": (0, 1),
        "low_right": (1, 0),
        "low_left": (-1, 0),
        "wait": (0, 0),
    }

    def __init__(self, width, height, walls, plant, cell_size):
        super().__init__(width, height)
        self.walls = walls
        self.grid_width = width
        self.grid_height = height
        self.cell_size = cell_size

        self.Location = UserType("Location")

        self.rewards = 0
        self.current_state = None
        self.new_state = None
        self.current_state_env = None
        self.initial_states = None
        self.penalty_amount = 0
        self.frozen_lake = False
        self.max_time = 1000
        self.plant = plant
        self.centralized = False
        self.no_qrm = False

        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.bridge_crossings = []
        self.boat_crossings = []

    def normalize_action_name(self, action_name: str) -> str:
        """Return a canonical action label understood by the environment logic."""
        return self.ACTION_ALIASES.get(action_name, action_name)

    @property
    def kind(self) -> "up.model.problem_kind.ProblemKind":
        """
        Calculates and returns the `problem kind` of this planning problem.
        If the Problem is modified, this method must be called again.
        """
        self._kind = up.model.problem_kind.ProblemKind(
            version=LATEST_PROBLEM_KIND_VERSION
        )
        self._kind.set_problem_class("ACTION_BASED_MULTI_AGENT")

        for fluent in self.ma_environment.fluents:
            self._update_problem_kind_fluent(fluent)
        for ag in self.agents:
            self._update_agent_goal_kind(ag)
            for action in ag.actions:
                self._update_problem_kind_action(action)
        for goal in self._goals:
            self._update_problem_kind_condition(goal)
        return self._kind

    def initialize_state(self):
        """Creates a sequential simulator and retrieves the initial state."""
        self.seq_ag = SequentialSimulatorMA(self)
        self.initial_states = self.seq_ag.get_initial_state()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a starting point.

        Returns:
            observations, infos for each agent.
        """
        self.rewards = {agent.name: 0 for agent in self.agents}
        self.timestep = 0
        self.current_state = {}
        self.current_state_env = self.initial_states
        self.agent_states = {agent.name: {} for agent in self.agents}
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}

        for agent in self.agents:
            agent.reset_messages()
            agent.message_sent = False
            agent.get_reward_machine().reset_to_initial_state()

            ag_state = self.current_state_env.get_dot_values(
                agent, only_true_values=True
            )
            self.agent_states[agent.name] = ag_state

            l_algo = agent.get_learning_algorithm()
            if isinstance(l_algo, QLearning):
                l_algo.learn_done_episode()

            self.agent_states[agent.name][(agent.name, "timestep")] = 0

        observations = self.agent_states
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """
        Executes the given actions for each agent.

        Args:
            actions (dict): agent_name -> action

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.rewards = {a.name: 0 for a in self.agents}
        infos = {a.name: {} for a in self.agents}

        for agent in self.agents:
            if not self.centralized and not self.active_agents[agent.name]:
                # Populate infos with default values when an agent is inactive
                infos[agent.name]["prev_s"] = self.get_state(agent)
                infos[agent.name]["s"] = self.get_state(agent)
                infos[agent.name]["Renv"] = 0
                continue  # Skip terminated agents

            prev_state = self.get_state(agent)
            action = actions[agent.name]

            self.execute_agent_action(agent, action, prev_state)
            new_state = self.get_state(agent)

            reward_env = 0  # or self.plants_in_the_office(new_state, agent.name)
            self.rewards[agent.name] += reward_env
            self.agent_steps[agent.name] += 1

            infos[agent.name]["prev_s"] = prev_state
            infos[agent.name]["s"] = new_state
            infos[agent.name]["Renv"] = reward_env

        self.timestep += 1

        terminations, truncations = self.check_terminations()
        observations = self.agent_states
        return observations, self.rewards, terminations, truncations, infos

    def _find_dot_node_in_state(self, agent, fluent_label):
        """Return the existing Dot fluent for the given agent and label if present."""
        if fluent_label is None:
            return None

        label = str(fluent_label).strip()
        current_instance = self.current_state_env
        while current_instance is not None:
            for key in current_instance._values.keys():
                if not hasattr(key, "is_dot"):
                    continue
                if key.is_dot() and key.agent() == agent.name:
                    key_label = str(key.args[0])
                    if key_label == label or str(key) == label:
                        return key
            current_instance = current_instance._father
        return None

    def _wrap_public_effect_value(self, fluent_node, value):
        """Convert raw Python literals into UP expressions compatible with the fluent."""
        if isinstance(value, up.model.FNode):
            return value

        fluent_type = fluent_node.type
        if fluent_type.is_bool_type():
            return Bool(bool(value))
        if fluent_type.is_int_type():
            return Int(int(value))
        if fluent_type.is_real_type():
            return Real(float(value))
        return value

    def _resolve_public_effect_target(self, agent, fluent_identifier):
        """Map the identifier coming from the RM metadata to an existing Dot fluent."""
        if isinstance(fluent_identifier, up.model.FNode):
            if fluent_identifier.is_dot():
                label = str(fluent_identifier.args[0])
            else:
                label = str(fluent_identifier)
        elif isinstance(fluent_identifier, tuple):
            label = str(fluent_identifier[0])
        else:
            label = fluent_identifier

        return self._find_dot_node_in_state(agent, label)

    def apply_public_effects(self, triggering_agent, event_conditions):
        """Apply public effects fired by RM events to other agents' states."""
        if not event_conditions:
            return

        updates_per_agent = defaultdict(list)

        for condition in event_conditions:
            if not isinstance(condition, (tuple, list)) or len(condition) < 2:
                continue
            key, value = condition[0], condition[1]
            if isinstance(key, tuple) and key and not isinstance(key[0], int):
                agent_name = key[0]
                fluent_identifier = key[1]
            else:
                agent_name = triggering_agent.name
                fluent_identifier = key
            updates_per_agent[agent_name].append((fluent_identifier, value))

        if not updates_per_agent:
            return

        state = self.current_state_env
        for agent_name, fluents in updates_per_agent.items():
            agent_obj = next((a for a in self.agents if a.name == agent_name), None)
            if agent_obj is None:
                continue
            for fluent_identifier, value in fluents:
                dot_node = self._resolve_public_effect_target(
                    agent_obj, fluent_identifier
                )
                if dot_node is None:
                    continue
                assigned_value = self._wrap_public_effect_value(dot_node, value)
                state = state.make_child({dot_node: assigned_value})

        self.current_state_env = state

        for agent_name in updates_per_agent.keys():
            agent_obj = next((a for a in self.agents if a.name == agent_name), None)
            if agent_obj is None:
                continue
            ag_state = self.current_state_env.get_dot_values(
                agent_obj, only_true_values=False
            )
            self.agent_states[agent_name].update(ag_state)
            false_keys = [
                fluent
                for fluent, val in self.agent_states[agent_name].items()
                if val is False
            ]
            for fluent in false_keys:
                del self.agent_states[agent_name][fluent]

    def update_bridges_and_boats(self):
        """
        Reset "has_bridge" and "has_boat" to False after crossings are completed.
        """
        for crossing in self.boat_crossings:
            current_location, new_location = crossing
            boat_flu = self.ma_environment.fluent("has_boat")
            node_bt1 = boat_flu(current_location, new_location)
            node_bt2 = boat_flu(new_location, current_location)
            new_state = self.current_state_env.make_child({node_bt1: Bool(False)})
            new_state = new_state.make_child({node_bt2: Bool(False)})
            self.current_state_env = new_state

        self.boat_crossings.clear()

    def walls_reward(self, agent, new_state):
        """
        Checks if the agent has moved to a penalty cell and applies the penalty.
        """
        agent_pos = (new_state[(agent.name, "pos_x")], new_state[(agent.name, "pos_y")])
        if agent_pos in self.penalty_cells:
            self.rewards[agent.name] += self.penalty_amount

    def execute_agent_action(self, agent, action, current_state):
        """
        Executes the specified action for the given agent and updates its state.
        """
        key_x = (agent.name, "pos_x")
        key_y = (agent.name, "pos_y")
        key_i = (agent.name, "pos_i")
        key_j = (agent.name, "pos_j")

        x = current_state[key_x]
        y = current_state[key_y]
        i = current_state[key_i]
        j = current_state[key_j]

        normalized_action = self.normalize_action_name(action.name)

        current_location = self.get_location_by_coordinates(agent, x, y)
        new_location = self.update_coordinates(
            agent, normalized_action, current_location
        )

        blocked = self.is_move_blocked(
            normalized_action, x, y, i, j, self.walls, self.cell_size
        )
        plant = self.is_move_to_plant(normalized_action, x, y, i, j, self.cell_size)

        if plant or blocked:
            return

        if new_location is not None:
            state = self.seq_ag._apply(
                self.ma_environment,
                agent,
                self.current_state_env,
                action,
                (current_location, new_location),
            )
        else:
            state = None

        if state is not None:
            self.current_state_env = state

            ag_state = self.current_state_env.get_dot_values(
                agent, only_true_values=False
            )
            self.agent_states[agent.name].update(ag_state)

            # Remove false fluents from the agent state
            fluents_to_remove = [
                fluent
                for fluent, value in self.agent_states[agent.name].items()
                if value is False
            ]
            for fluent in fluents_to_remove:
                del self.agent_states[agent.name][fluent]

    def relative_to_absolute(self, x, y, i, j, room_size):
        """
        Converts relative coordinates (x, y, i, j) to absolute coordinates.
        """
        absolute_x = x * room_size + i
        absolute_y = y * room_size + j
        return absolute_x, absolute_y

    def is_move_blocked(self, action, x, y, i, j, walls, cell_size):
        """
        Checks if the movement specified by the action is blocked by a wall.
        """
        normalized_action = self.normalize_action_name(action)

        if normalized_action not in self.CANONICAL_ACTION_DELTAS:
            raise ValueError(f"Azione non valida: {action}")

        delta_i, delta_j = self.CANONICAL_ACTION_DELTAS[normalized_action]

        absolute_coords = self.relative_to_absolute(x, y, i, j, cell_size)

        new_i = i + delta_i
        new_j = j + delta_j
        absolute_coords_after_action = self.relative_to_absolute(
            x, y, new_i, new_j, cell_size
        )

        is_blocked = (absolute_coords, absolute_coords_after_action) in walls or (
            absolute_coords_after_action,
            absolute_coords,
        ) in walls

        return is_blocked

    def is_move_to_plant(self, action, x, y, i, j, cell_size):
        """
        Checks if the movement specified by the action will move the agent to a plant.
        """
        normalized_action = self.normalize_action_name(action)

        if normalized_action not in self.CANONICAL_ACTION_DELTAS:
            raise ValueError(f"Azione non valida: {action}")

        dx, dy = self.CANONICAL_ACTION_DELTAS[normalized_action]

        absolute_coords = self.relative_to_absolute(x, y, i, j, cell_size)
        next_abs_coords = (absolute_coords[0] + dx, absolute_coords[1] + dy)

        return next_abs_coords in self.plant

    def update_agent_states(self, agent_name: str, ag_state):
        """Updates the state of the specified agent based on new state values."""
        for fluent in list(self.agent_states[agent_name].keys()):
            if fluent not in ag_state:
                del self.agent_states[agent_name][fluent]
        self.agent_states[agent_name].update(ag_state)

    def plants_in_the_office(self, agent_state, agent_name):
        """Checks if the agent is in a cell containing a plant."""
        x = agent_state.get((agent_name, "pos_x"))
        y = agent_state.get((agent_name, "pos_y"))
        i = agent_state.get((agent_name, "pos_i"))
        j = agent_state.get((agent_name, "pos_j"))
        position_abs = self.relative_to_absolute(x, y, i, j, self.cell_size)

        agent_pos = (position_abs[0], position_abs[1])
        return agent_pos in self.plant

    def check_terminations(self):
        """
        Checks if any of the agents have reached their termination state.
        """
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}
        for ag in self.agents:
            if (
                ag.get_reward_machine().get_current_state()
                == ag.get_reward_machine().get_final_state()
            ):
                terminations[ag.name] = True
                truncations[ag.name] = True
            elif self.agent_fail[ag.name]:
                terminations[ag.name] = True
                truncations[ag.name] = True
                self.active_agents[ag.name] = False
            elif self.timestep > self.max_time:
                truncations[ag.name] = True
                terminations[ag.name] = True
                self.active_agents[ag.name] = False
        return terminations, truncations

    def get_state(self, agent):
        """Returns a copy of the current state of the agent."""
        return self.agent_states[agent.name].copy()

    def verifica_condizioni(self, ag, conditions, dic_messaggi, current_location_map):
        """
        Verifies if the given conditions are satisfied for an agent.
        """
        for condition in conditions:
            if isinstance(condition[0], tuple):  # involves another agent
                agent_condition, fluent_condition = condition[0][0], condition[0][1]
                messaggio_chiave = (agent_condition, fluent_condition)
                if (
                    messaggio_chiave not in dic_messaggi
                    or dic_messaggi[messaggio_chiave] != condition[1]
                ):
                    return False
            else:  # local condition
                fluent, value = condition
                if (
                    str(fluent) != current_location_map[0]
                    or value != current_location_map[1]
                ):
                    return False
        return True

    def extract_agent_ids(self, conditions):
        """
        Extracts the identifiers of agents involved in the given conditions.
        """
        agent_ids = set()
        for cond in conditions:
            if (
                isinstance(cond[0], tuple)
                and len(cond[0]) == 2
                and isinstance(cond[0][0], str)
            ):
                agent_id = cond[0][0]
                agent_ids.add(agent_id)
        return list(agent_ids)

    def broadcast_message(self, agents, message):
        """Broadcasts a message to all agents except the sender."""
        for agent in agents:
            if agent != message.sender:
                ag = self.agent(agent)
                ag._receive_message(message)

    def initialize_location_mapping(self, coordinates):
        """
        Initializes the mapping between coordinates and location objects for all agents.
        """
        self.coord_to_location_map = {}
        self.location_to_coord_map = {}

        for agent in self.agents:
            for coord, location_obj in coordinates:
                self.coord_to_location_map[(agent.name, coord)] = location_obj
                self.location_to_coord_map[(agent.name, location_obj)] = coord

    def get_location_by_coordinates(self, agent, x, y):
        """Retrieves the location object corresponding to the given coordinates."""
        return self.coord_to_location_map.get((agent.name, (x, y)))

    def get_coordinates_by_location(self, agent, location):
        """Retrieves the coordinates corresponding to the given location object."""
        return self.location_to_coord_map.get((agent.name, location))

    def update_coordinates(self, agent, action_name, current_location):
        """
        Updates the coordinates of the agent based on the specified action and current location.
        This works at room level (x, y), not at subcell (i, j) level.
        """
        current_coords = self.get_coordinates_by_location(agent, current_location)
        if current_coords is None:
            return None

        # Only actions that move between rooms affect (x, y).
        room_level_deltas = {
            "up": (0, -1),
            "down": (0, +1),
            "left": (-1, 0),
            "right": (1, 0),
            "cross_up": (0, -1),
            "cross_down": (0, +1),
            "cross_left": (-1, 0),
            "cross_right": (1, 0),
            "row_up": (0, -1),
            "row_down": (0, +1),
            "row_left": (-1, 0),
            "row_right": (1, 0),
        }

        normalized_action = self.normalize_action_name(action_name)
        dx, dy = room_level_deltas.get(normalized_action, (0, 0))
        new_coords = (current_coords[0] + dx, current_coords[1] + dy)

        return self.get_location_by_coordinates(agent, *new_coords)

    def preview_new_location(self, agent, action_name):
        """
        Returns (current location, new location) like 'execute_agent_action'
        would, but without actually changing the state of the environment.
        """
        current_state = self.get_state(agent)
        x = current_state[(agent.name, "pos_x")]
        y = current_state[(agent.name, "pos_y")]
        current_location = self.get_location_by_coordinates(agent, x, y)
        if current_location is None:
            return None, None

        new_location = self.update_coordinates(agent, action_name, current_location)
        return current_location, new_location
