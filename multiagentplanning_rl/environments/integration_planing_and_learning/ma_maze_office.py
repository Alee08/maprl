# from algo_rl import RL_algorithms
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagentplanning_rl.multi_agent.reward_machine import RewardMachine

from unified_planning.shortcuts import *

from multiagentplanning_rl.utils.ma_sequential_simulator import (
    UPSequentialSimulatorMA as SequentialSimulatorMA,
)
import cProfile
import json
import pickle
from building_RM import RM_dict, RM_dict_true, RM_dict_true_seq
from multiagentplanning_rl.utils.message import Message
from multiagent_rlrm.multi_agent.base_environment import BaseEnvironment
from collections import defaultdict


class MAP_RL_Env(BaseEnvironment):

    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "multi_agent_planning_with_RL",
    }

    def __init__(self, width, height, walls, plant, cell_size):

        super().__init__(width, height)
        self.walls = walls  # Inserisci le coordinate dei muri qui
        self.grid_width = width  # 10 celle di larghezza
        self.grid_height = height  #  10 celle di altezza
        self.cell_size = cell_size
        # Reimposta epsilon all'inizio di ogni episodio
        self.rewards = 0
        self.current_state = None
        self.position_A = (3, 1)
        self.position_B = (8, 9)
        self.position_C = (4, 7)
        self.position_D = (8, 1)
        self.position_E = (7, 5)
        self.position_F = (1, 8)
        self.new_state = None
        self.num_rm_states = 4  # Aggiorna questo valore in base al tuo specifico caso
        self.Location = UserType("Location")
        self.l33 = Object("l33", self.Location)
        self.l34 = Object("l34", self.Location)
        self.current_state_env = None
        # self.seq_ag = SequentialSimulatorMA(self)
        # self.current_state_env = self.seq_ag.get_initial_state()
        self.initial_states = None
        # self.message_conditions = None
        self.penalty_cells = [
            self.position_A,
            self.position_B,
            self.position_C,
            self.position_D,
            self.position_E,
            self.position_F,
        ]  # Coordinate delle celle con penalità
        self.penalty_amount = 0
        self.rewards = 0
        self.frozen_lake = False
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.bridge_crossings = []
        self.max_time = 1000
        self.plant = plant
        self.centralized = False
        self.no_qrm = False

        self.boat_crossings = []

    @property
    def kind(self) -> "up.model.problem_kind.ProblemKind":
        """
        Calculates and returns the `problem kind` of this `planning problem`.
        If the `Problem` is modified, this method must be called again in order to be reliable.

        IMPORTANT NOTE: this property does a lot of computation, so it should be called as
        seldom as possible.
        """
        self._kind = up.model.problem_kind.ProblemKind(
            version=LATEST_PROBLEM_KIND_VERSION
        )
        self._kind.set_problem_class("ACTION_BASED_MULTI_AGENT")
        """for ag in self.agents:
            for fluent in ag.fluents:
                self._update_problem_kind_fluent(fluent)"""
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
        """
        Initializes the state of the environment.
        Creates a sequential simulator and retrieves the initial state.
        """
        self.seq_ag = SequentialSimulatorMA(self)
        self.initial_states = self.seq_ag.get_initial_state()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a starting point.

        Args:
            seed (int, optional): Seed for random number generation. Default is None.
            options (dict, optional): Additional options for the reset. Default is None.

        Returns:
            tuple: Contains observations and infos dictionary for each agent.
        """
        # Aggiorna epsilon per il controllo dell'esplorazione
        self.rewards = {agent.name: 0 for agent in self.agents}
        #self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        self.timestep = 0
        self.current_state = {}
        # self.seq_ag = SequentialSimulatorMA(self)
        # self.current_state_env = self.seq_ag.get_initial_state()
        self.current_state_env = self.initial_states
        self.agent_states = {agent.name: {} for agent in self.agents}
        # self.message_conditions = None
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
            initial_position = (
                agent.get_state()
            )  # Assumi che get_state ritorni un dizionario con pos_x e pos_y
            if isinstance(l_algo, QLearning):
                l_algo.learn_done_episode()
            self.agent_states[agent.name][(agent.name, "timestep")] = 0

        # Reset the overall environment state
        # self.current_state_env = self.seq_ag.get_initial_state()
        observations = self.agent_states
        # Get dummy infos
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Executes the given actions for each agent in the environment.

        Args:
            actions (dict): A dictionary mapping each agent to its action.

        Returns:
            tuple: Contains observations, rewards, terminations, truncations, and infos dictionaries for each agent.
        """
        terminations, truncations, infos = {}, {}, {}
        self.rewards = {a.name: 0 for a in self.agents}
        infos = {a.name: {} for a in self.agents}

        # ============== 1) Verifica concurrency "row_*" per l13->l14 ==============
        # 1) Costruiamo due dizionari:
        #    - desired_boats[(l_from, l_to)] = [lista_agenti]
        #    - desired_bridges[(l_from, l_to)] = [lista_agenti]
        """desired_boats = defaultdict(list)
        desired_bridges = defaultdict(list)

        # 2) Per ogni agente, se l'azione è row_* => preview (l_from, l_to) => accumulate
        #                   se l'azione è cross_* => preview (l_from, l_to) => accumulate
        for ag in self.agents:
            if not self.active_agents.get(ag.name, True):
                continue  # agente inattivo

            action = actions[ag.name]

            # Calcola (l_from, l_to) "virtualmente"
            if action.name.startswith("row_left"):
                l_from, l_to = self.preview_new_location(ag, action.name)
                if l_from is not None and l_to is not None:
                    desired_boats[(l_from, l_to)].append(ag.name)

            elif action.name.startswith("cross_down"):
                l_from, l_to = self.preview_new_location(ag, action.name)
                if l_from is not None and l_to is not None:
                    desired_bridges[(l_from, l_to)].append(ag.name)

        # 3) Attiviamo la barca se >=2 agenti fanno "row_*" sullo stesso arco
        boat_flu = self.ma_environment.fluent("has_boat")
        for (loc_from, loc_to), agent_list in desired_boats.items():
            if len(agent_list) >= 2:
                # Se vuoi limitare la barca a (l14 <-> l24), controlla:
                if (str(loc_from)=='l14' and str(loc_to)=='l24') or (str(loc_from)=='l24' and str(loc_to)=='l14'):
                    node_bt1 = boat_flu(loc_from, loc_to)
                    node_bt2 = boat_flu(loc_to, loc_from)
                    new_state = self.current_state_env.make_child({node_bt1: Bool(True)})
                    new_state = new_state.make_child({node_bt2: Bool(True)})
                    self.current_state_env = new_state
                    print(f"Abilitato has_boat({loc_from}->{loc_to})=True, agenti concurrency={agent_list}")

        # 4) Attiviamo il ponte se >=3 agenti fanno "cross_*" sullo stesso arco
        bridge_flu = self.ma_environment.fluent("has_bridge")
        for (loc_from, loc_to), agent_list in desired_bridges.items():
            if len(agent_list) >= 3:
                # Se vuoi limitare il ponte a (l13 <-> l14), controlla:
                if (str(loc_from)=='l13' and str(loc_to)=='l14') or (str(loc_from)=='l14' and str(loc_to)=='l13'):
                    node_br1 = bridge_flu(loc_from, loc_to)
                    node_br2 = bridge_flu(loc_to, loc_from)
                    new_state = self.current_state_env.make_child({node_br1: Bool(True)})
                    new_state = new_state.make_child({node_br2: Bool(True)})
                    self.current_state_env = new_state
                    print(f"Abilitato has_bridge({loc_from}->{loc_to})=True, agenti concurrency={agent_list}")"""
        # --- FASE 2: Ora esegui le azioni effettive ---

        for agent in self.agents:
            if not self.centralized:
                if not self.active_agents[agent.name]:
                    # Riempi cmq infos con un default
                    infos[agent.name]["prev_s"] = self.get_state(agent)
                    infos[agent.name]["s"] = self.get_state(agent)
                    infos[agent.name]["Renv"] = 0
                    # E poi magari salti tutto il resto
                    continue  # Salta gli agenti terminati
            current_statee = self.get_state(agent)
            action = actions[agent.name]
            self.execute_agent_action(agent, action, current_statee)
            new_state = self.get_state(agent)

            # reward_env = self.plants_in_the_office(new_state, agent.name)
            reward_env = 0
            self.rewards[agent.name] += reward_env
            self.agent_steps[agent.name] += 1
            infos[agent.name]["prev_s"] = current_statee
            # infos[agent.name]["prev_q"] = state_rm
            infos[agent.name]["s"] = new_state
            # infos[agent.name]["q"] = new_state_rm
            infos[agent.name]["Renv"] = reward_env
            # infos[agent.name]["RQ"] = reward

            # q_learning.update(current_statee, new_state, agent_action, reward, agent, state_rm, new_state_rm)
            # agent.add_to_state("timestep", self.timestep)
        self.timestep += 1

        # 1) Aggiorna bridges e boats dopo che tutti hanno agito
        # self.update_bridges_and_boats()

        terminations, truncations = self.check_terminations()
        # Aggiorna il fluente 'has_bridge' alla fine del timestep

        # breakpoint()
        observations = self.agent_states

        return observations, self.rewards, terminations, truncations, infos

    def update_bridges_and_boats(self):
        """
        Aggiorna 'has_bridge' e 'has_boat' a False se si è verificato l'attraversamento
        (ponte "cade" e barca "scompare").
        """
        # 1) Gestione "bridge_crossings"
        """for crossing in self.bridge_crossings:
            current_location, new_location = crossing
            bridge_flu = self.ma_environment.fluent("has_bridge")
            # Metto False in entrambi i versi (current->new e new->current)
            node_br1 = bridge_flu(current_location, new_location)
            node_br2 = bridge_flu(new_location, current_location)
            
            new_state = self.current_state_env.make_child({node_br1: Bool(False)})
            new_state = new_state.make_child({node_br2: Bool(False)})
            self.current_state_env = new_state

        # Svuota la lista per il prossimo timestep
        self.bridge_crossings.clear()"""

        # 2) Gestione "boat_crossings"
        for crossing in self.boat_crossings:
            current_location, new_location = crossing
            boat_flu = self.ma_environment.fluent("has_boat")
            # Metto False in entrambi i versi
            node_bt1 = boat_flu(current_location, new_location)
            node_bt2 = boat_flu(new_location, current_location)
            # breakpoint()
            new_state = self.current_state_env.make_child({node_bt1: Bool(False)})
            new_state = new_state.make_child({node_bt2: Bool(False)})
            self.current_state_env = new_state

        self.boat_crossings.clear()

    def walls_reward(self, agent, new_state):
        """
        Checks if the agent has moved to a penalty cell and applies the penalty.

        Args:
            agent: The agent being checked.
            new_state (dict): The new state of the agent.
        """
        # Controlla se la nuova posizione è una cella di penalità
        agent_pos = (new_state[(agent.name, "pos_x")], new_state[(agent.name, "pos_y")])
        if agent_pos in self.penalty_cells:
            self.rewards[agent.name] += self.penalty_amount  # Applica la penalità

    def execute_agent_action(self, agent, action, current_state):
        """
        Executes the specified action for the given agent and updates its state.

        Args:
            agent: The agent performing the action.
            action: The action to be executed.
            current_state (dict): The current state of the agent.
        """
        key_x = (agent.name, "pos_x")
        key_y = (agent.name, "pos_y")
        key_i = (agent.name, "pos_i")
        key_j = (agent.name, "pos_j")

        x = current_state[key_x]
        y = current_state[key_y]
        i = current_state[key_i]
        j = current_state[key_j]

        current_location = self.get_location_by_coordinates(agent, x, y)
        new_location = self.update_coordinates(agent, action.name, current_location)
        blocked = False
        blocked = self.is_move_blocked(
            action.name, x, y, i, j, self.walls, self.cell_size
        )
        plant = self.is_move_to_plant(action.name, x, y, i, j, self.cell_size)
        # Se l'agente compie un'azione di "row_*" e l'attraversamento è l24 -> l14 (o viceversa)
        """if not blocked and action.name in ["row_up", "row_down", "row_left", "row_right"]:
            if (str(current_location) == "l24" and str(new_location) == "l14") \
                    or (str(current_location) == "l14" and str(new_location) == "l24"):
                self.boat_crossings.append((current_location, new_location))"""

        if not plant:
            if not blocked:
                # self.current_state_env = self.seq_ag.apply_unsafe(agent, self.current_state_env, action)
                if new_location != None:
                    state = self.seq_ag._apply(
                        self.ma_environment,
                        agent,
                        self.current_state_env,
                        action,
                        (current_location, new_location),
                    )
                else:
                    state = None

                if state != None:
                    self.current_state_env = state

                    # Registra l'attraversamento del ponte se l'azione è di tipo 'cross'
                    if (
                        action.name == "cross_down"
                        and str(current_location) == "l13"
                        and str(new_location) == "l14"
                    ):
                        self.bridge_crossings.append((current_location, new_location))

                ag_state = self.current_state_env.get_dot_values(
                    agent, only_true_values=False
                )
                self.agent_states[agent.name].update(ag_state)

                # Raccogli i fluenti da rimuovere
                fluents_to_remove = [
                    fluent
                    for fluent, value in self.agent_states[agent.name].items()
                    if value is False
                ]
                # Rimuovi i fluenti raccolti dal dizionario
                for fluent in fluents_to_remove:
                    del self.agent_states[agent.name][fluent]

    def relative_to_absolute(self, x, y, i, j, room_size):
        """
        Converts relative coordinates (x, y, i, j) to absolute coordinates.

        Args:
            x, y: Room coordinates.
            i, j: Relative coordinates within the room.
            room_size (int): Size of the room.

        Returns:
            tuple: Absolute coordinates.
        """
        absolute_x = x * room_size + i
        absolute_y = y * room_size + j
        return absolute_x, absolute_y

    def is_move_blocked(self, action, x, y, i, j, walls, cell_size):
        """
        Checks if the movement specified by the action is blocked by a wall.

        Args:
            action (str): The action name ('up', 'down', 'left', 'right', etc.).
            x, y: Room coordinates.
            i, j: Relative coordinates within the room.
            walls (list): List of wall coordinates.
            cell_size (int): Size of the room cells.

        Returns:
            bool: True if the movement is blocked, False otherwise.
        """

        # Funzione per convertire le coordinate relative in assolute
        def relative_to_absolute(x, y, i, j, room_size=cell_size):
            absolute_x = x * room_size + i
            absolute_y = y * room_size + j
            return absolute_x, absolute_y

        # Mappa delle azioni ai cambiamenti di coordinate
        action_to_coord_change = {
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

        # Verifica se l'azione è valida
        if action not in action_to_coord_change:
            raise ValueError(f"Azione non valida: {action}")

        # Ottieni il cambiamento di coordinate per l'azione
        delta_i, delta_j = action_to_coord_change[action]

        # Calcola le coordinate assolute iniziali
        absolute_coords = relative_to_absolute(x, y, i, j, cell_size)

        # Determina le nuove coordinate relative in base all'azione
        new_i = i + delta_i
        new_j = j + delta_j

        # Calcola le nuove coordinate assolute
        absolute_coords_after_action = relative_to_absolute(
            x, y, new_i, new_j, cell_size
        )

        # Verifica se c'è un muro tra le coordinate iniziali e quelle dopo l'azione
        is_blocked = (absolute_coords, absolute_coords_after_action) in walls or (
            absolute_coords_after_action,
            absolute_coords,
        ) in walls

        return is_blocked

    def is_move_to_plant(self, action, x, y, i, j, cell_size):
        """
        Checks if the movement specified by the action will move the agent to a plant.

        Args:
            action (str): The action name.
            x, y: Room coordinates.
            i, j: Relative coordinates within the room.
            cell_size (int): Size of the room cells.

        Returns:
            bool: True if the movement is towards a plant, False otherwise.
        """

        def relative_to_absolute(x, y, i, j, room_size=cell_size):
            absolute_x = x * room_size + i
            absolute_y = y * room_size + j
            return absolute_x, absolute_y

        # Mappa delle azioni ai cambiamenti di coordinate
        action_to_coord_change = {
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

        # Verifica se l'azione è valida
        if action not in action_to_coord_change:
            raise ValueError(f"Azione non valida: {action}")
        absolute_coords = relative_to_absolute(x, y, i, j, cell_size)
        # Ottieni il cambiamento di coordinate per l'azione
        dx, dy = action_to_coord_change[action]

        # Forza le coordinate in interi, nel caso ci fossero discrepanze con il formato di self.plant
        next_abs_coords = (absolute_coords[0] + dx, absolute_coords[1] + dy)

        # Verifica se la nuova posizione è una pianta
        if next_abs_coords in self.plant:
            # print(f"Si muove su una pianta! ({absolute_coords} -> {next_abs_coords}) {action}")
            return True
        else:
            return False

    def update_agent_states(self, agent_name: str, ag_state):
        """
        Updates the state of the specified agent based on the new state values.

        Args:
            agent_name (str): The name of the agent.
            ag_state (dict): The new state of the agent.
        """
        for fluent, value in self.agent_states[
            agent_name
        ].items():  # Usa list() per creare una copia perché modificherai il dizionario durante l'iterazione
            if fluent not in ag_state:
                del self.agent_states[agent_name][fluent]
        self.agent_states[agent_name].update(ag_state)

    def plants_in_the_office(self, agent_state, agent_name):
        """
        Checks if the agent is in a cell containing a plant.

        Args:
            agent_state (dict): The state of the agent.
            agent_name (str): The name of the agent.

        Returns:
            bool: True if the agent is in a plant cell, False otherwise.
        """
        x = agent_state.get((agent_name, "pos_x"))
        y = agent_state.get((agent_name, "pos_y"))
        i = agent_state.get((agent_name, "pos_i"))
        j = agent_state.get((agent_name, "pos_j"))
        position_abs = self.relative_to_absolute(x, y, i, j, self.cell_size)

        agent_pos = (position_abs[0], position_abs[1])
        if agent_pos in self.plant:
            print(agent_pos, self.plant)
            return True
        else:
            # return 0
            return False

    def check_terminations(self):
        """
        Checks if any of the agents have reached their termination state.

        Returns:
            tuple: Dictionaries indicating terminations and truncations for each agent.
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
                self.active_agents[ag.name] = False  # Termina l'agente che ha fallito
            elif self.timestep > self.max_time:
                truncations[ag.name] = True
                terminations[ag.name] = True
                self.active_agents[ag.name] = False  # Termina l'agente per timeout
        return terminations, truncations

    def get_state(self, agent):
        """
        Retrieves the current state of the specified agent.

        Args:
            agent: The agent for which the state is being retrieved.

        Returns:
            dict: A copy of the current state of the agent.
        """
        # Restituisce una copia dello stato corrente dell'agente per evitare modifiche accidentali
        return self.agent_states[agent.name].copy()

    def verifica_condizioni(self, ag, conditions, dic_messaggi, current_location_map):
        """
        Verifies if the given conditions are satisfied for an agent.

        Args:
            ag: The agent being checked.
            conditions (list): List of conditions to verify.
            dic_messaggi (dict): Dictionary of messages received by the agent.
            current_location_map (dict): Mapping of current locations of agents.

        Returns:
            bool: True if all conditions are satisfied, False otherwise.
        """

        for condition in conditions:
            if isinstance(
                condition[0], tuple
            ):  # La condizione coinvolge un altro agente
                agent_condition, fluent_condition = condition[0][0], condition[0][1]
                # Utilizza messaggi_semplificati per il confronto
                messaggio_chiave = (agent_condition, fluent_condition)
                if (
                    messaggio_chiave not in dic_messaggi
                    or dic_messaggi[messaggio_chiave] != condition[1]
                ):
                    return False  # La condizione non è soddisfatta
            else:  # La condizione è locale per l'agente
                fluent, value = condition
                if (
                    str(fluent) != current_location_map[0]
                    or value != current_location_map[1]
                ):
                    return False  # La condizione locale non è soddisfatta

        # Tutte le condizioni sono soddisfatte
        return True

    def extract_agent_ids(self, conditions):
        """
        Extracts the identifiers of agents involved in the given conditions.

        Args:
            conditions (list): List of conditions.

        Returns:
            list: List of agent identifiers.
        """
        agent_ids = set()
        for cond in conditions:
            if (
                isinstance(cond[0], tuple)
                and len(cond[0]) == 2
                and isinstance(cond[0][0], str)
            ):
                agent_id = cond[0][0]  # Estrai l'identificatore dell'agente
                agent_ids.add(agent_id)
        return list(agent_ids)

    def broadcast_message(self, agents, message):
        """
        Broadcasts a message to all agents except the sender.

        Args:
            agents (list): List of all agents in the environment.
            message (Message): The message to be broadcasted.
        """
        for agent in agents:
            if agent != message.sender:
                ag = self.agent(agent)
                ag._receive_message(message)

    def initialize_location_mapping(self, coordinates):
        """
        Initializes the mapping between coordinates and location objects for all agents.

        Args:
            coordinates (list): List of coordinate-location pairs.
        """
        self.coord_to_location_map = {}
        self.location_to_coord_map = {}

        for agent in self.agents:
            for coord, location_obj in coordinates:
                # Usa una tupla (agente, location) come chiave
                self.coord_to_location_map[(agent.name, coord)] = location_obj
                self.location_to_coord_map[(agent.name, location_obj)] = coord

    def get_location_by_coordinates(self, agent, x, y):
        """
        Retrieves the location object corresponding to the given coordinates.

        Args:
            agent: The agent for which the location is being retrieved.
            x (int): The x-coordinate of the location.
            y (int): The y-coordinate of the location.

        Returns:
            Location: The location object for the given coordinates.
        """
        return self.coord_to_location_map.get((agent.name, (x, y)))

    def get_coordinates_by_location(self, agent, location):
        """
        Retrieves the coordinates corresponding to the given location object.

        Args:
            agent: The agent for which the coordinates are being retrieved.
            location (Location): The location object.

        Returns:
            tuple: The coordinates for the given location.
        """
        return self.location_to_coord_map.get((agent.name, location))

    def update_coordinates(self, agent, action_name, current_location):
        """
        Updates the coordinates of the agent based on the specified action and current location.

        Args:
            agent: The agent being moved.
            action_name (str): The action name determining the movement.
            current_location (Location): The current location of the agent.

        Returns:
            Location: The new location of the agent.
        """
        current_coords = self.get_coordinates_by_location(agent, current_location)
        if current_coords is None:
            return None

        action_to_coord_change = {
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

        dx, dy = action_to_coord_change.get(action_name, (0, 0))
        new_coords = (current_coords[0] + dx, current_coords[1] + dy)

        # Ottieni la nuova location basata sull'agente e le nuove coordinate
        return self.get_location_by_coordinates(agent, *new_coords)

    def preview_new_location(self, agent, action_name):
        """
        Ritorna (current_location, new_location) come farebbe 'execute_agent_action'
        ma senza modificare davvero lo stato dell'ambiente.
        """
        current_state = self.get_state(agent)
        x = current_state[(agent.name, "pos_x")]
        y = current_state[(agent.name, "pos_y")]
        current_location = self.get_location_by_coordinates(agent, x, y)
        if current_location is None:
            return None, None

        # Stessa logica di 'update_coordinates(...)' per scoprire la new_location
        new_location = self.update_coordinates(agent, action_name, current_location)
        return (current_location, new_location)
