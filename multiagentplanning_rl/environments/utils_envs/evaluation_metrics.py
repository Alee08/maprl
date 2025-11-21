from multiagent_rlrm.utils.utils import encode_state
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.learning_algorithms.qlearning_lambda import QLearningLambda
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)
from multiagentplanning_rl.environments.integration_planing_and_learning.state_encoder_maze_office import (
    StateEncoderMAPRL,
)
import numpy as np
import copy
import wandb
import os
import string
import copy
import numpy as np


def calcola_media_mobile(valori, finestra):
    medie = np.convolve(valori, np.ones(finestra) / finestra, mode="valid")
    return medie


def test_policy_optima(
    env, episodi_test=100, window_size=100, optimal_steps=30, gamma=0.9
):
    env_test = copy.deepcopy(env)  # Crea una copia dell'ambiente per i test
    successi_per_agente = {ag.name: 0 for ag in env_test.agents}
    successi_per_episodio = {ag.name: [] for ag in env_test.agents}
    ricompense_per_agente = {ag.name: [] for ag in env_test.agents}  # Nuovo
    timestep_per_episodio = []  # Lista per tenere traccia dei timestep per episodio
    arps_per_agente = {
        ag.name: [] for ag in env_test.agents
    }  # Per tenere traccia dell'ARPS

    for episodio in range(episodi_test):
        states, infos = env_test.reset(
            "111"
        )  # Resetta l'ambiente di test per ogni episodio
        done = {ag.name: False for ag in env_test.agents}
        agent_done = {ag.name: False for ag in env_test.agents}
        timestep = 0  # Inizializza il contatore di timestep per l'episodio
        episode_rewards = {ag.name: 0 for ag in env_test.agents}  # Nuovo

        while not all(done.values()):
            actions = {}
            for ag in env_test.agents:
                current_state = env_test.env.get_state(ag)
                azione = ag.select_action(
                    current_state, best=True
                )  # Seleziona l'azione in modalità test
                # print(f"Current State: {current_state}, Select Action: {azione.name}")
                actions[ag.name] = azione

            new_states, rewards, done, truncations, infos = env_test.step(
                actions
            )  # Esegui un passo per tutti gli agenti

            for ag in env_test.agents:
                if not agent_done[ag.name]:
                    episode_rewards[ag.name] += rewards[
                        ag.name
                    ]  # Accumula le ricompense
                    if (
                        done[ag.name]
                        and ag.get_reward_machine().get_current_state()
                        == ag.get_reward_machine().get_final_state()
                    ):
                        successi_per_agente[
                            ag.name
                        ] += 1  # Conta come successo se l'agente raggiunge lo stato finale della RM
                        agent_done[ag.name] = True

            if all(done.values()) or all(truncations.values()):
                break

            states = copy.deepcopy(
                new_states
            )  # Aggiorna lo stato per il prossimo timestep
            timestep += 1  # Incrementa il contatore di timestep per l'episodio
            if timestep > 1000:
                break

        timestep_per_episodio.append(
            timestep
        )  # Aggiungi il conteggio di timestep per l'episodio corrente

        for ag in env_test.agents:
            successi_per_episodio[ag.name].append(1 if agent_done[ag.name] else 0)
            ricompense_per_agente[ag.name].append(episode_rewards[ag.name])  # Nuovo

        # Calcolo dell'ARPS scontato per episodio
        for ag_name, reward in episode_rewards.items():
            discounted_reward = sum(
                reward * (gamma ** t) for t, reward in enumerate(rewards.values())
            )
            if timestep > 0:  # Evita la divisione per zero
                arps = (discounted_reward / timestep) / optimal_steps
                arps_per_agente[ag_name].append(arps)

    success_rate_per_agente = {
        ag_name: (successi / episodi_test) * 100
        for ag_name, successi in successi_per_agente.items()
    }

    # Calcola la media mobile per ogni agente
    moving_averages = {}
    for ag_name in successi_per_episodio:
        moving_averages[ag_name] = (
            np.convolve(successi_per_episodio[ag_name], np.ones(window_size), "valid")
            / window_size
        )

    # Calcola la media dei timestep
    average_timesteps = np.mean(timestep_per_episodio)

    # Calcola la ricompensa media per ogni agente
    avg_reward_per_agente = {
        ag_name: np.mean(rewards) for ag_name, rewards in ricompense_per_agente.items()
    }

    # Calcola l'ARPS medio per ogni agente
    avg_arps_per_agente = {
        ag_name: np.mean(arps) for ag_name, arps in arps_per_agente.items()
    }

    return (
        success_rate_per_agente,
        moving_averages,
        average_timesteps,
        avg_reward_per_agente,
        avg_arps_per_agente,  # Aggiungi ARPS al ritorno
    )


def save_q_tables(agents, directory="data"):
    q_tables_dict = {}
    for agent in agents:
        try:
            q_table = agent.get_learning_algorithm().q_table
        except:
            q_table = agent.get_learning_algorithm().nTSQA
        q_tables_dict[f"q_table_{agent.name}"] = q_table
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed(f"{directory}/q_tables.npz", **q_tables_dict)


def update_actions_log(actions_log, actions, episode):
    """
    Aggiorna il log delle azioni per ogni agente durante un episodio.

    Args:
        actions_log (dict): Dizionario contenente i log delle azioni per ogni agente.
        actions (dict): Dizionario delle azioni eseguite dagli agenti nell'ultimo step.
        episode (int): Numero dell'episodio corrente.
    """
    if episode not in actions_log:
        actions_log[episode] = {}
    for agent_name, action in actions.items():
        if agent_name not in actions_log[episode]:
            actions_log[episode][agent_name] = []
        actions_log[episode][agent_name].append(action.name)


def save_actions_log(actions_log, file_path="data/final_episode_log.json"):
    """
    Salva il log delle azioni in un file JSON.

    Args:
        actions_log (dict): Dizionario contenente i log delle azioni per ogni agente.
        file_path (str): Percorso del file dove salvare il log.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        json.dump(actions_log, f, indent=4)


def update_successes(env, rewards_agents, successi_per_agente, done):
    for agent in env.agents:
        rm_current_state = agent.reward_machine.get_current_state()
        rm_final_state = agent.reward_machine.get_final_state()
        if (
            done[agent.name]
            and rm_current_state == rm_final_state
            and rewards_agents[agent.name] > 0
        ):
            successi_per_agente[agent.name] += 1


def prepare_log_data(
    env,
    episode,
    rewards_agents,
    successi_per_agente,
    ricompense_per_episodio,
    finestra_media_mobile,
):
    log_data = {"epsilon": env.epsilon, "episode": episode, "total_step": env.timestep}

    for agent in env.agents:
        agent_name = agent.name
        """if env.active_agents[agent.name] == False:
            continue"""
        reward = rewards_agents[agent_name]
        steps = env.agent_steps[agent_name]
        success_rate = (successi_per_agente[agent_name] / (episode + 1)) * 100
        ricompense_per_episodio[agent_name].append(reward)

        if len(ricompense_per_episodio[agent_name]) >= finestra_media_mobile:
            media_mobile = calcola_media_mobile(
                ricompense_per_episodio[agent_name], finestra_media_mobile
            )
            log_data[f"media_mobile_{agent_name}"] = media_mobile[-1]

        log_data.update(
            {
                f"reward_{agent_name}": reward,
                f"step_{agent_name}": steps,
                f"success_rate_training_{agent_name}": success_rate,
            }
        )

    return log_data


def get_epsilon_summary(agents):
    epsilon_parts = []
    for agent in agents:
        if isinstance(agent.learning_algorithm, (QLearning, QLearningLambda)):
            epsilon_parts.append(
                f"{agent.name}: {agent.learning_algorithm.epsilon:.2f}"
            )
        else:
            epsilon_parts.append(
                f"{agent.name}: N/A"
            )  # Per algoritmi che non usano epsilon
    return ", ".join(epsilon_parts)


def value_iteration(S, A, T, L, rm, gamma):
    """
    Standard value iteration to compute optimal policies for the grid environments.

    PARAMS
    ----------
    S:     List of states
    A:     List of actions
    T:     Transitions (it is a dictionary from SxA -> S)
    L:     Labeling function (it is a dictionary from states to events)
    rm:    Reward machine
    gamma: Discount factor

    RETURNS
    ----------
    Optimal deterministic policy (dictionary mapping from states (SxU) to actions)
    """
    U = rm.get_all_states()  # All states in the Reward Machine
    V = dict([((s, u), 0) for s in S for u in U])
    V_error = 1

    while V_error > 0.0000001:
        V_error = 0
        for s1 in S:
            for u1 in U:
                q_values = []
                for a in A:
                    s2 = T[(s1, a)]
                    l = "" if s2 not in L else L[s2]
                    u2, r = rm.get_reward_for_non_current_state(u1, l)

                    # Ensure correct transition logic
                    if u2 is None:  # If no transition, continue with the same state
                        u2 = u1
                    if u2 == rm.get_final_state():
                        done = True
                    else:
                        done = False

                    if done:
                        q_values.append(r)
                    else:
                        q_values.append(r + gamma * V[(s2, u2)])

                v_new = max(q_values)
                V_error = max([V_error, abs(v_new - V[(s1, u1)])])
                V[(s1, u1)] = v_new

    # Extracting the optimal policy
    policy = {}
    for s1 in S:
        for u1 in U:
            q_values = []
            for a in A:
                s2 = T[(s1, a)]
                l = "" if s2 not in L else L[s2]
                u2, r = rm.get_reward_for_non_current_state(u1, l)
                if u2 is None:
                    u2 = u1
                if u2 == rm.get_final_state():
                    done = True
                else:
                    done = False

                if done:
                    q_values.append(r)
                else:
                    q_values.append(r + gamma * V[(s2, u2)])

            a_i = max((x, i) for i, x in enumerate(q_values))[
                1
            ]  # argmax over the q-values
            policy[(s1, u1)] = A[a_i]

    return policy


def compute_normalized_arps(gamma, optimal_steps, avg_reward=1.0, target_arps=1.0):
    """
    Calcola l'ARPS normalizzato in modo che il valore finale sia pari a target_arps.

    :param gamma: Fattore di sconto
    :param optimal_steps: Numero di passi ottimali
    :param avg_reward: Ricompensa media per passo, di default 1
    :param target_arps: ARPS desiderato (normalizzato a 1)
    :return: ARPS normalizzato calcolato
    """
    if gamma == 1:
        # Gestire il caso quando gamma è 1, che può risultare in divisione per zero
        return target_arps / optimal_steps

    # Calcola l'ARPS senza normalizzazione
    total_reward = avg_reward * (1 - gamma ** (optimal_steps + 1)) / (1 - gamma)
    arps = total_reward / optimal_steps

    # Calcola il fattore di scala necessario
    scale_factor = target_arps / arps

    # Applica la normalizzazione
    normalized_total_reward = total_reward * scale_factor
    normalized_arps = normalized_total_reward / optimal_steps
    return normalized_arps


##########################################MAPRL################################################

"""def test_policy_optima_MAPRL(
    training_env, MAP_RL_Env, episodi_test=100, window_size=100, optimal_steps=30, gamma=0.9
):
    # Create a new environment instance for testing
    env_test = MAP_RL_Env(
        width=training_env.env.grid_width,
        height=training_env.env.grid_height,
        walls=training_env.env.walls,
        plant=training_env.env.plant,
        cell_size=training_env.env.cell_size,
        # Include other necessary initialization arguments
    )
    env_test.initialize_state()

    # Create new agents for the test environment
    agents_test = []
    #for agent_training in env_test.agents:
        # Create a new agent with the same name and environment
        #agent_test = AgentRL(agent_training.name, env_test)

        # Set the same Reward Machine
        #agent_test.set_reward_machine(agent_training.get_reward_machine())

        # Set the same learning algorithm with shared Q-table
        #learning_algorithm = agent_training.get_learning_algorithm()
        #agent_test.set_learning_algorithm(learning_algorithm)  # Share the same instance

        # Add the agent to the test environment
        #env_test.add_agent(agent_training)

        #agents_test.append(agent_test)
    agents_test = env_test.agents
    #breakpoint()
    # Wrap env_test with RMEnvironmentWrapper
    rm_env_test = RMEnvironmentWrapper(env_test, agents_test)

    # Initialize metrics
    successi_per_agente = {ag.name: 0 for ag in agents_test}
    timestep_per_episodio = []
    ricompense_per_agente = {ag.name: [] for ag in agents_test}
    arps_per_agente = {ag.name: [] for ag in agents_test}

    for episodio in range(episodi_test):
        states, infos = rm_env_test.reset()
        done = {ag.name: False for ag in agents_test}
        agent_done = {ag.name: False for ag in agents_test}
        timestep = 0
        episode_rewards = {ag.name: 0 for ag in agents_test}

        while not all(done.values()):
            actions = {}
            for ag in agents_test:
                current_state = rm_env_test.env.get_state(ag)
                action = ag.select_action(current_state, best=True)
                actions[ag.name] = action

            new_states, rewards, done, truncations, infos = rm_env_test.step(actions)

            for ag in agents_test:
                if not agent_done[ag.name]:
                    episode_rewards[ag.name] += rewards[ag.name]
                    if (
                        done[ag.name]
                        and ag.get_reward_machine().get_current_state()
                        == ag.get_reward_machine().get_final_state()
                    ):
                        successi_per_agente[ag.name] += 1
                        agent_done[ag.name] = True

            if all(done.values()) or all(truncations.values()):
                break

            states = copy.deepcopy(new_states)
            timestep += 1
            if timestep > 100:
                break

        timestep_per_episodio.append(timestep)
        for ag in agents_test:
            ricompense_per_agente[ag.name].append(episode_rewards[ag.name])

        # Calculate ARPS for each agent
        for ag_name, reward in episode_rewards.items():
            discounted_reward = sum(
                reward * (gamma ** t) for t in range(timestep)
            )
            arps = (discounted_reward / timestep) / optimal_steps
            arps_per_agente[ag_name].append(arps)

    success_rate_per_agente = {
        ag_name: (successi / episodi_test) * 100
        for ag_name, successi in successi_per_agente.items()
    }

    average_timesteps = np.mean(timestep_per_episodio)
    avg_reward_per_agente = {
        ag_name: np.mean(rewards) for ag_name, rewards in ricompense_per_agente.items()
    }
    avg_arps_per_agente = {
        ag_name: np.mean(arps) for ag_name, arps in arps_per_agente.items()
    }

    return (
        success_rate_per_agente,
        None,  # You can adjust this if needed
        average_timesteps,
        avg_reward_per_agente,
        avg_arps_per_agente,
    )"""


def test_policy_optima_MAPRL(
    rm_env_test, episodi_test=100, window_size=100, optimal_steps=30, gamma=0.9
):
    """
    Tests the optimal policy for MAPRL (Multi-Agent Reinforcement Learning) and calculates metrics for agent performance.

    :param rm_env_test: Reward Machine environment for testing
    :param episodi_test: Number of test episodes
    :param window_size: Window size for moving average
    :param optimal_steps: Optimal number of steps for completing the task
    :param gamma: Discount factor
    :return: Success rate per agent, moving averages, average timesteps, average reward per agent, average ARPS per agent
    """
    # Inizializza le metriche
    successi_per_agente = {ag.name: 0 for ag in rm_env_test.agents}
    successi_per_episodio = {ag.name: [] for ag in rm_env_test.agents}
    ricompense_per_agente = {ag.name: [] for ag in rm_env_test.agents}
    timestep_per_episodio = []
    arps_per_agente = {ag.name: [] for ag in rm_env_test.agents}

    for episodio in range(episodi_test):
        states, infos = rm_env_test.reset("111")
        done = {ag.name: False for ag in rm_env_test.agents}
        agent_done = {ag.name: False for ag in rm_env_test.agents}
        timestep = 0
        episode_rewards = {ag.name: 0 for ag in rm_env_test.agents}
        per_step_rewards = {ag.name: [] for ag in rm_env_test.agents}

        while not all(done.values()):
            actions = {}
            for ag in rm_env_test.agents:
                if done[ag.name]:
                    continue  # Salta gli agenti che hanno terminato
                current_state = rm_env_test.env.get_state(ag)
                azione = ag.select_action(current_state, best=True)
                actions[ag.name] = azione

            new_states, rewards, done, truncations, infos = rm_env_test.step(actions)

            for ag in rm_env_test.agents:
                if not agent_done[ag.name]:
                    episode_rewards[ag.name] += rewards[ag.name]
                    per_step_rewards[ag.name].append(rewards[ag.name])
                    if (
                        done[ag.name]
                        and ag.get_reward_machine().get_current_state()
                        == ag.get_reward_machine().get_final_state()
                    ):
                        successi_per_agente[ag.name] += 1
                        agent_done[ag.name] = True

            if all(done.values()) or all(truncations.values()):
                break

            states = copy.deepcopy(new_states)
            timestep += 1
            if timestep > 100:
                break

        timestep_per_episodio.append(timestep)

        for ag in rm_env_test.agents:
            successi_per_episodio[ag.name].append(1 if agent_done[ag.name] else 0)
            ricompense_per_agente[ag.name].append(episode_rewards[ag.name])

        # Calcolo dell'ARPS scontato per episodio
        for ag_name in episode_rewards.keys():
            rewards_list = per_step_rewards[ag_name]
            discounted_reward = sum(
                reward * (gamma ** t) for t, reward in enumerate(rewards_list)
            )
            if timestep > 0:
                arps = (discounted_reward / timestep) / optimal_steps
                arps_per_agente[ag_name].append(arps)
            else:
                arps_per_agente[ag_name].append(0)

    # Calcola il tasso di successo per agente
    success_rate_per_agente = {
        ag_name: (successi / episodi_test) * 100
        for ag_name, successi in successi_per_agente.items()
    }

    # Calcola la media mobile per ogni agente
    moving_averages = {}
    for ag_name in successi_per_episodio:
        data = successi_per_episodio[ag_name]
        if len(data) >= window_size:
            moving_avg = np.convolve(data, np.ones(window_size), "valid") / window_size
            moving_averages[ag_name] = moving_avg
        else:
            moving_averages[ag_name] = np.array([])

    # Calcola la media dei timestep
    average_timesteps = np.mean(timestep_per_episodio)

    # Calcola la ricompensa media per ogni agente
    avg_reward_per_agente = {
        ag_name: np.mean(rews) for ag_name, rews in ricompense_per_agente.items()
    }

    # Calcola l'ARPS medio per ogni agente
    avg_arps_per_agente = {
        ag_name: np.mean(arps) for ag_name, arps in arps_per_agente.items()
    }

    return (
        success_rate_per_agente,
        moving_averages,
        average_timesteps,
        avg_reward_per_agente,
        avg_arps_per_agente,
    )


def _sanitize_office_world_lines(office_world):
    """Remove comments and empty rows from a map string."""

    sanitized_lines = []
    for raw_line in office_world.strip().split("\n"):
        line = raw_line.split("#", 1)[0].strip()
        if line:
            sanitized_lines.append(line)
    return sanitized_lines


def parse_office_world_(office_world):
    """
    Parses an office world representation into coordinates, goals, walls, rooms, and connections.

    :param office_world: String representation of the office world
    :return: Parsed coordinates, goals, walls, rooms, and connections
    """
    symbol_mapping = {"🟩": "empty_cell", "🪴": "plant", "🥤": "coffee", "✉️": "letter"}
    # Parse the office world into a list of lists, ignoring ⛔ and 🚪
    office_lines = [
        line.replace("⛔", "").replace("🚪", "").strip().split()
        for line in _sanitize_office_world_lines(office_world)
    ]
    filtered_list_of_lists = [lst for lst in office_lines if lst]
    goals = {
        char: [] for char in string.ascii_uppercase + string.digits
    }  # Dictionary for alphabetic and numeric goals

    # Initialize dictionaries to hold the coordinates for each symbol
    coordinates = {"plant": [], "coffee": [], "letter": [], "empty_cell": []}

    # Iterate through the office lines and collect the coordinates
    for y, row in enumerate(filtered_list_of_lists):
        for x, cell in enumerate(row):
            if cell in symbol_mapping:
                coordinates[symbol_mapping[cell]].append((x, y))
            elif cell in string.ascii_uppercase + string.digits:
                goals[cell].append((x, y))

    # Filter out empty goal lists
    goals = {k: v[0] for k, v in goals.items() if v}

    walls = find_disconnected_pairs(office_world)

    # Parse the rooms and connections
    rooms, connections = find_rooms_and_connections(office_world)

    return (
        coordinates,
        goals,
        walls,
        rooms,
        connections,
    )


def parse_office_grid(office_world):
    """
    Parses the office world into a list of lists representing the grid.

    :param office_world: String representation of the office world
    :return: Parsed grid as a list of lists
    """

    # Parsing the grid into a list of lists
    grid = [line.strip().split() for line in _sanitize_office_world_lines(office_world)]
    return grid


def find_disconnected_pairs(office_world):
    """
    Parses the office world into a list of lists representing the grid.

    :param office_world: String representation of the office world
    :return: Parsed grid as a list of lists
    """

    grid = parse_office_grid(office_world)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    disconnected_pairs = []

    y_offsets = [0] * rows
    y_adjusted = 0
    for y in range(rows):
        if all(cell == "⛔" or cell == "🚪" for cell in grid[y]):
            y_offsets[y] = 1
        else:
            y_offsets[y] = y_adjusted
            y_adjusted += 1

    x_offsets = [0] * cols
    x_adjusted = 0
    for x in range(cols):
        if all(grid[y][x] == "⛔" or grid[y][x] == "🚪" for y in range(rows)):
            x_offsets[x] = 1
        else:
            x_offsets[x] = x_adjusted
            x_adjusted += 1

    for y in range(rows):
        for x in range(cols):
            if grid[y][x] == "⛔" or grid[y][x] == "🚪":
                continue
            # Check horizontal pairs
            if (
                x < cols - 2  # Check two cells ahead
                and grid[y][x + 1] == "⛔"
                and grid[y][x + 2] != "⛔"
            ):
                disconnected_pairs.append(
                    ((x_offsets[x], y_offsets[y]), (x_offsets[x + 2], y_offsets[y]))
                )
            # Check vertical pairs
            if (
                y < rows - 2  # Check two cells below
                and grid[y + 1][x] == "⛔"
                and grid[y + 2][x] != "⛔"
            ):
                disconnected_pairs.append(
                    ((x_offsets[x], y_offsets[y]), (x_offsets[x], y_offsets[y + 2]))
                )

    return disconnected_pairs


def find_rooms_and_connections(office_world):
    """
    Identifies rooms and their connections from the office world representation.

    :param office_world: String representation of the office world
    :return: Rooms and their connections as dictionaries
    """
    grid = parse_office_grid(office_world)
    visited = set()
    rooms = {}
    connections = {}
    room_counter = 1

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Calculate x_offsets and y_offsets similar to find_disconnected_pairs
    y_offsets = [0] * rows
    y_adjusted = 0
    for y in range(rows):
        if all(cell == "⛔" or cell == "🚪" for cell in grid[y]):
            y_offsets[y] = -1
        else:
            y_offsets[y] = y_adjusted
            y_adjusted += 1

    x_offsets = [0] * cols
    x_adjusted = 0
    for x in range(cols):
        if all(grid[y][x] == "⛔" or grid[y][x] == "🚪" for y in range(rows)):
            x_offsets[x] = -1
        else:
            x_offsets[x] = x_adjusted
            x_adjusted += 1

    # Create an inverse map for adjusted to original coordinates
    inverse_adjusted_map = {}
    for y in range(rows):
        for x in range(cols):
            if x_offsets[x] != -1 and y_offsets[y] != -1:
                inverse_adjusted_map[(x_offsets[x], y_offsets[y])] = (x, y)

    def is_valid(x, y):
        return (
            0 <= x < cols
            and 0 <= y < rows
            and (x, y) not in visited
            and grid[y][x] != "⛔"
            and grid[y][x] != "🚪"
        )

    def dfs(x, y):
        stack = [(x, y)]
        room = []

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            room.append((x_offsets[cx], y_offsets[cy]))

            # Check adjacent cells (right, down, left, up)
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if is_valid(nx, ny):
                    stack.append((nx, ny))

        return room

    # Traverse the grid and identify rooms
    row_col_mapping = {}
    room_row_col_mapping = {}

    for y in range(rows):
        for x in range(cols):
            if is_valid(x, y):
                room = dfs(x, y)
                if room:
                    # Determine the top-left corner to use as the base for row/column naming
                    min_x = min(pos[0] for pos in room)
                    min_y = min(pos[1] for pos in room)

                    # Determine the logical row and column based on minimum coordinates
                    if min_y not in row_col_mapping:
                        row_number = len(row_col_mapping) + 1
                        row_col_mapping[min_y] = row_number
                    else:
                        row_number = row_col_mapping[min_y]

                    if min_x not in room_row_col_mapping:
                        col_number = (
                            len(
                                [
                                    col
                                    for row, col in room_row_col_mapping.values()
                                    if row == row_number
                                ]
                            )
                            + 1
                        )
                        room_row_col_mapping[min_x] = (row_number, col_number)
                    else:
                        col_number = room_row_col_mapping[min_x][1]

                    room_name = f"l{col_number}{row_number}"
                    rooms[room_name] = room

                    # Initialize connection list for the room
                    connections[room_name] = []

                    # Check for connections with adjacent rooms
                    check_room_connections(
                        grid, room_name, room, rooms, connections, inverse_adjusted_map
                    )

    return rooms, connections


def check_room_connections(
    grid, room_name, room, rooms, connections, inverse_adjusted_map
):
    """
    Checks for room connections to adjacent rooms and updates the connections dictionary.

    :param grid: Parsed office grid
    :param room_name: Name of the current room
    :param room: Coordinates of the current room
    :param rooms: Dictionary of all rooms and their coordinates
    :param connections: Dictionary of room connections
    :param inverse_adjusted_map: Map of adjusted to original coordinates
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    for (ax, ay) in room:
        # Ottieni le coordinate originali usando la mappa inversa
        cx, cy = inverse_adjusted_map[(ax, ay)]

        # Controlla a destra (East)
        if cx + 1 < cols and grid[cy][cx + 1] == "🚪" and cx + 2 < cols:
            adjacent_room = find_room_by_coord(cx + 2, cy, rooms, inverse_adjusted_map)
            if adjacent_room and adjacent_room != room_name:
                if adjacent_room not in connections[room_name]:
                    connections[room_name].append(adjacent_room)
                if room_name not in connections[adjacent_room]:
                    connections[adjacent_room].append(room_name)

        # Controlla a sinistra (West)
        if cx - 1 >= 0 and grid[cy][cx - 1] == "🚪" and cx - 2 >= 0:
            adjacent_room = find_room_by_coord(cx - 2, cy, rooms, inverse_adjusted_map)
            if adjacent_room and adjacent_room != room_name:
                if adjacent_room not in connections[room_name]:
                    connections[room_name].append(adjacent_room)
                if room_name not in connections[adjacent_room]:
                    connections[adjacent_room].append(room_name)

        # Controlla in basso (South)
        if cy + 1 < rows and grid[cy + 1][cx] == "🚪" and cy + 2 < rows:
            adjacent_room = find_room_by_coord(cx, cy + 2, rooms, inverse_adjusted_map)
            if adjacent_room and adjacent_room != room_name:
                if adjacent_room not in connections[room_name]:
                    connections[room_name].append(adjacent_room)
                if room_name not in connections[adjacent_room]:
                    connections[adjacent_room].append(room_name)

        # Controlla in alto (North)
        if cy - 1 >= 0 and grid[cy - 1][cx] == "🚪" and cy - 2 >= 0:
            adjacent_room = find_room_by_coord(cx, cy - 2, rooms, inverse_adjusted_map)
            if adjacent_room and adjacent_room != room_name:
                if adjacent_room not in connections[room_name]:
                    connections[room_name].append(adjacent_room)
                if room_name not in connections[adjacent_room]:
                    connections[adjacent_room].append(room_name)


def find_room_by_coord(x, y, rooms, inverse_adjusted_map):
    """
    Finds the room name corresponding to a given coordinate.

    :param x: X-coordinate to find the room
    :param y: Y-coordinate to find the room
    :param rooms: Dictionary of all rooms and their coordinates
    :param inverse_adjusted_map: Map of adjusted to original coordinates
    :return: Name of the room at the given coordinate, or None if not found
    """
    for room_name, coords in rooms.items():
        if (x, y) in [inverse_adjusted_map[(ax, ay)] for ax, ay in coords]:
            return room_name
    return None


def log_test_episode_data(rm_env, total_steps_per_agent, rewards_agents, episode):
    """
    Logs test episode data to Weights & Biases.

    :param rm_env: The Reward Machine environment instance.
    :param total_steps_per_agent: Dictionary containing the total steps taken by each agent.
    :param rewards_agents: Dictionary containing the rewards obtained by each agent.
    :param episode: The current episode number.
    """
    # Calcola il tasso di successo per agente
    success_rates = {}
    for agent in rm_env.agents:
        # L'agente ha successo se ha raggiunto lo stato finale
        if (
            agent.get_reward_machine().get_current_state()
            == agent.get_reward_machine().get_final_state()
        ):
            success_rates[agent.name] = 100.0
        else:
            success_rates[agent.name] = 0.0

    # Logga i dati su wandb
    log_data = {}
    for agent in rm_env.agents:
        log_data[f"test/steps_{agent.name}"] = total_steps_per_agent[agent.name]
        log_data[f"test/success_rate_{agent.name}"] = success_rates[agent.name]
        log_data[f"test/reward_{agent.name}"] = rewards_agents[agent.name]

    wandb.log(log_data, step=episode)


def test_policy_ottima(rm_env_test, central_agent, num_test_episodes=20):
    """
    Esegue episodi di test usando la policy appresa dal central_agent
    (selezionando le azioni con best=True) e calcola metriche di performance.
    """
    episodi_con_successo = 0
    sum_timesteps = 0
    sum_total_reward = 0.0
    rm_env_test.env.centralized = True
    rm_env_test.env.initialize_state()
    for ep in range(num_test_episodes):
        # (1) Reset del wrapper di test
        central_agent.central_rm.reset_to_initial_state()  # la RM centralizzata
        states, infos = rm_env_test.reset("111")
        states = copy.deepcopy(states)
        done = {ag.name: False for ag in rm_env_test.agents}
        timesteps = 0
        episode_reward = 0.0

        while any(rm_env_test.env.active_agents.values()):
            # Costruiamo global_state
            global_state = {}
            for ag in rm_env_test.agents:
                global_state[ag.name] = rm_env_test.env.get_state(ag)

            # Azioni “best”
            actions = central_agent.select_actions(global_state, best=True)

            new_states, rewards, done, truncations, infos = rm_env_test.step(actions)

            event = central_agent.central_rm.event_detector.detect_event(
                new_states, central_agent.central_rm.current_state
            )
            reward_from_rm = central_agent.central_rm.step(
                new_states, central_agent.central_rm.current_state
            )

            # Ricompensa cumulata
            total_r_step = sum(rewards.values())
            episode_reward += reward_from_rm

            timesteps += 1

            """if all(done.values()) or all(truncations.values()):
                break"""

            # Se la RM centralizzata è in final_state => successo
            if (
                central_agent.central_rm.get_current_state()
                == central_agent.central_rm.get_final_state()
            ):
                episodi_con_successo += 1
                break

        sum_timesteps += timesteps
        sum_total_reward += episode_reward
        print(sum_timesteps, sum_total_reward)
    # Calcoliamo metriche
    success_rate = (episodi_con_successo / num_test_episodes) * 100.0
    avg_timesteps = sum_timesteps / num_test_episodes
    avg_reward = sum_total_reward / num_test_episodes

    return success_rate, avg_timesteps, avg_reward
