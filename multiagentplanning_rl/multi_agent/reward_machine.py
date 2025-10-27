from typing import Optional, List, Union, Iterable


class RewardMachine:
    def __init__(self, transitions, event_detector):
        """
        Initializes the Reward Machine with a set of transitions and an event detector.

        Args:
            transitions (dict): Dictionary of transitions in the form {(current_state, event): (new_state, reward)}.
            event_detector (EventDetector): An instance of EventDetector for detecting events in the environment.
        """
        self.transitions = transitions  # {(current_state, event): (new_state, reward)}
        self.initial_state = self._get_start_state()  # Memorizza lo stato iniziale
        self.current_state = self.initial_state
        self.state_indices = self._generate_state_indices()
        self.event_detector = event_detector
        self.potentials = None  # Aggiungi questa linea per memorizzare i potenziali

        # Crea una mappatura da nome stato a indice e viceversa
        self.state_to_idx = self.state_indices  # Basato su _generate_state_indices
        self.idx_to_state = {idx: state for state, idx in self.state_indices.items()}

    def _generate_state_indices(self):
        """
        Generates unique indices for each state in the Reward Machine.

        Returns:
            dict: Dictionary mapping each state to a unique index.
        """
        unique_states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            unique_states.add(from_state)
            unique_states.add(to_state)

        # Assicurati che lo stato iniziale sia incluso e mappato a zero
        unique_states.add(self.current_state)
        sorted_states = sorted(unique_states)
        sorted_states.remove(self.current_state)
        sorted_states.insert(0, self.current_state)
        # breakpoint()
        # Assegna un indice univoco a ciascuno stato
        return {state: i for i, state in enumerate(sorted_states)}

    def get_state_index(self, rm_state):
        """
        Retrieves the index of a given state in the Reward Machine.

        Args:
            rm_state (str): The state for which to retrieve the index.

        Returns:
            int: Index corresponding to the given state.
        """
        return self.state_indices[rm_state]

    def step(self, current_state, state_rm):
        """
        Detects the current event, executes the state transition, and returns the reward.

        Args:
            current_state (dict): The current state of the environment.
            state_rm (str): The current state of the Reward Machine.

        Returns:
            int: The reward obtained after the state transition.
        """
        event = self.event_detector.detect_event(current_state, state_rm)
        # print(f"Detected event: {event} for current state: {current_state}")
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state
            return reward
        return 0

    def get_reward(self, event):
        """
        Returns the reward associated with a specific event without executing the state transition.

        Args:
            event (tuple): The event for which to retrieve the reward.

        Returns:
            int: The reward associated with the given event.
        """
        if (self.current_state, event) in self.transitions:
            new_state, reward = self.transitions[(self.current_state, event)]
            self.current_state = new_state

            return reward

        return 0

    def get_reward_for_non_current_state(self, state_rm, event):
        """
        Retrieves the reward for a specific event from a non-current state.

        Args:
            state_rm (str): The state from which to check the reward.
            event (tuple): The event to check.

        Returns:
            tuple: (new_state, reward) for the given state and event.
        """
        if isinstance(event, list):
            print("It's a list!")
            breakpoint()
            event = tuple(event)

        if (state_rm, event) in self.transitions:
            new_state, reward = self.transitions[(state_rm, event)]
            return new_state, reward
        else:
            return None, 0

    def get_all_states(self):
        """
        Returns all unique states present in the Reward Machine.

        Returns:
            list: List of all states in the Reward Machine in order of appearance.
        """
        seen_states = set()
        all_states = []

        # Aggiungere gli stati in ordine di apparizione
        for (from_state, _), (to_state, _) in self.transitions.items():
            if from_state not in seen_states:
                all_states.append(from_state)
                seen_states.add(from_state)
            if to_state not in seen_states:
                all_states.append(to_state)
                seen_states.add(to_state)

        return all_states

    def get_possible_events(self, state_rm):
        """
        Retrieves all possible events from a given state in the Reward Machine.

        Args:
            state_rm (str): The state for which to retrieve possible events.

        Returns:
            list: List of possible events from the given state.
        """
        possible_events = []
        for (current_state, event), (new_state, _) in self.transitions.items():
            if current_state == state_rm:
                possible_events.append(event)
        return possible_events

    def get_current_state(self):
        """
        Returns the current state of the Reward Machine.

        Returns:
            str: The current state.
        """
        return self.current_state

    def numbers_state(self):
        """
        Counts the total number of unique states in the Reward Machine.

        Returns:
            int: Number of unique states.
        """
        states = set()
        for (from_state, _), (to_state, _) in self.transitions.items():
            states.add(from_state)
            states.add(to_state)
        return len(states)

    @property
    def get_transitions(self):
        """
        Counts the total number of unique states in the Reward Machine.

        Returns:
            int: Number of unique states.
        """
        return self.transitions

    def reset_to_initial_state(self):
        """
        Resets the Reward Machine to its initial state.

        Returns:
            str: The initial state.
        """
        self.current_state = (
            self.initial_state
        )  # Assumi che 'initial_state' sia memorizzato come attributo
        return self.initial_state

    def get_final_state(self):
        """
        Retrieves the final state of the Reward Machine.

        Returns:
            str: The final state in the Reward Machine.
        """
        if self.transitions:  # Assicurati che ci siano transizioni
            last_to_state = next(reversed(self.transitions.values()))[0]
            return last_to_state
        else:
            # Nessuna transizione definita, gestisci come preferisci
            return None

    def get_initial_state(self):
        """
        Retrieves the initial state of the Reward Machine.

        Returns:
            str: The initial state in the Reward Machine.
        """
        if self.transitions:  # Assicurati che ci siano transizioni
            transitions_list = list(self.transitions.values())
            initial_state = transitions_list[0][
                0
            ]  # Prendi solo il primo elemento della tupla
            return initial_state
        else:
            # Nessuna transizione definita, gestisci come preferisci
            return None

    def _get_start_state(self):
        """
        Retrieves the start state of the Reward Machine based on the defined transitions.

        Returns:
            str: The start state.
        """
        if not self.transitions:
            return None

        # Prendi il primo stato di partenza dalla prima transizione della lista delle transizioni
        first_transition = next(iter(self.transitions))
        start_state = first_transition[0]

        return start_state

    def get_state_name_from_idx(self, idx):
        """
        Restituisce il nome dello stato RM dato l'indice dello stato.

        :param idx: Indice dello stato RM.
        :return: Nome dello stato RM.
        """
        return self.idx_to_state.get(idx, None)

    def add_transitions(self, new_transitions, position="after"):
        """
        Adds new transitions to the existing Reward Machine transitions.

        Args:
            new_transitions (dict): Dictionary of new transitions to be added.
            position (str, optional): Position to add transitions, either 'before' or 'after'. Defaults to 'after'.
        """
        if position == "before":
            # Le nuove transizioni vengono aggiunte prima
            self.transitions = {**new_transitions, **self.transitions}
        elif position == "after":
            # Le nuove transizioni vengono aggiunte dopo
            self.transitions.update(new_transitions)
        else:
            raise ValueError("La posizione deve essere 'before' o 'after'.")

        # Ricostruisci gli indici degli stati
        self.state_indices = self._generate_state_indices()

        # Aggiorna lo stato iniziale
        self.initial_state = self._get_start_state()

        # Resetta lo stato corrente all'iniziale
        self.current_state = self.initial_state

    def extract_events(self):
        """
        Extracts all unique events from the Reward Machine's transitions.

        Returns:
            set: Set of unique events.
        """
        eventi_unici = set()
        for (_, conditions) in self.transitions.keys():
            for condition in conditions:
                eventi_unici.add(condition)
        return eventi_unici

    def add_transitions_with_merge(
        self, new_transitions, position="after", prefix="new"
    ):
        """
        Adds new transitions to the Reward Machine, renaming states and merging the last state
        of the new transitions with the first state of existing transitions.

        Args:
            new_transitions (dict): Dictionary of new transitions to be added.
            position (str, optional): Position to add transitions, either 'before' or 'after'. Defaults to 'after'.
            prefix (str, optional): Prefix for renaming states. Defaults to 'new'.
        """
        # Rinominare gli stati nelle nuove transizioni
        renamed_transitions = {}
        state_mapping_new = {}

        for (from_state, conditions), (to_state, reward) in new_transitions.items():
            # Rinominare lo stato di partenza
            if from_state not in state_mapping_new:
                state_mapping_new[from_state] = f"{prefix}_{from_state}"
            new_from_state = state_mapping_new[from_state]

            # Rinominare lo stato di arrivo
            if to_state not in state_mapping_new:
                state_mapping_new[to_state] = f"{prefix}_{to_state}"
            new_to_state = state_mapping_new[to_state]

            # Aggiungere la transizione rinominata
            renamed_transitions[(new_from_state, conditions)] = (new_to_state, reward)

        # Identificare l'ultimo stato delle nuove transizioni
        to_states_new = set(
            state_mapping_new[to_state]
            for (_, _), (to_state, _) in new_transitions.items()
        )
        from_states_new = set(
            state_mapping_new[from_state]
            for (from_state, _), _ in new_transitions.items()
        )
        last_new_states = to_states_new - from_states_new
        if not last_new_states:
            raise ValueError(
                "Non è stato possibile identificare l'ultimo stato delle nuove transizioni."
            )
        last_new_state = next(iter(last_new_states))

        # Identificare il primo stato delle transizioni esistenti
        from_states_existing = set(
            from_state for (from_state, _) in self.transitions.keys()
        )
        to_states_existing = set(
            to_state for (_, (to_state, _)) in self.transitions.items()
        )
        initial_states_existing = from_states_existing - to_states_existing
        if not initial_states_existing:
            raise ValueError(
                "Non è stato possibile identificare il primo stato delle transizioni esistenti."
            )
        first_existing_state = next(iter(initial_states_existing))

        # Unire gli stati
        # Sostituire il nome del primo stato delle transizioni esistenti con l'ultimo stato delle nuove transizioni
        state_mapping_existing = {first_existing_state: last_new_state}

        # Aggiornare le transizioni esistenti con il nuovo nome dello stato
        updated_existing_transitions = {}
        for (from_state, conditions), (to_state, reward) in self.transitions.items():
            # Aggiornare lo stato di partenza
            new_from_state = state_mapping_existing.get(from_state, from_state)
            # Aggiornare lo stato di arrivo
            new_to_state = state_mapping_existing.get(to_state, to_state)
            # Aggiungere la transizione aggiornata
            updated_existing_transitions[(new_from_state, conditions)] = (
                new_to_state,
                reward,
            )

        # Aggiornare le transizioni della RM
        if position == "before":
            self.transitions = {**renamed_transitions, **updated_existing_transitions}
            # Aggiorna lo stato iniziale
            self.initial_state = state_mapping_new[
                next(iter(new_transitions.keys()))[0]
            ]
            self.current_state = self.initial_state
        elif position == "after":
            # Questa logica per 'after' dovrebbe unire il primo stato delle nuove transizioni con l'ultimo stato delle transizioni esistenti
            # Implementazione simile a quanto sopra, ma per 'after'
            # Non richiesto in base alla tua specifica, ma possiamo implementarlo se necessario
            pass
        else:
            raise ValueError("La posizione deve essere 'before' o 'after'.")

        # Aggiornare gli indici degli stati
        self.state_indices = self._generate_state_indices()

        # Aggiornare l'Event Detector
        if self.event_detector:
            new_events = self.extract_events()
            self.event_detector.add_events(new_events)

    def add_reward_shaping(self, gamma, rs_gamma):
        """
        Adds new transitions to the Reward Machine, renaming states and merging the last state
        of the new transitions with the first state of existing transitions.

        Args:
            new_transitions (dict): Dictionary of new transitions to be added.
            position (str, optional): Position to add transitions, either 'before' or 'after'. Defaults to 'after'.
            prefix (str, optional): Prefix for renaming states. Defaults to 'new'.
        """
        self.gamma = gamma
        self.potentials = self.value_iteration(
            list(self.state_indices.keys()),
            self.get_delta_u(),
            self.get_delta_r(),
            self.get_final_state(),
            rs_gamma,
        )
        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]

    def get_delta_u(self):
        """
        Retrieves a dictionary representing state transitions and their events.

        Returns:
            dict: Dictionary of state transitions and events.
        """
        delta_u = {}
        for (u1, event), (u2, _) in self.transitions.items():
            if u1 not in delta_u:
                delta_u[u1] = {}
            if u2 not in delta_u:
                delta_u[u2] = {}  # Ensure u2 is also present
            delta_u[u1][u2] = event
        return delta_u

    def get_delta_r(self):
        """
        Retrieves a dictionary representing state transitions and their rewards.

        Returns:
            dict: Dictionary of state transitions and rewards.
        """
        delta_r = {}
        for (u1, event), (u2, reward) in self.transitions.items():
            if u1 not in delta_r:
                delta_r[u1] = {}
            if u2 not in delta_r:
                delta_r[u2] = {}  # Ensure u2 is also present
            delta_r[u1][u2] = ConstantRewardFunction(reward)
        return delta_r

    def value_iteration(self, U, delta_u, delta_r, terminal_u, gamma):
        """
        Performs value iteration for reward shaping.

        Args:
            U (list): List of states in the Reward Machine.
            delta_u (dict): Dictionary of state transitions and events.
            delta_r (dict): Dictionary of state transitions and rewards.
            terminal_u (str): The terminal state in the Reward Machine.
            gamma (float): Discount factor for value iteration.

        Returns:
            dict: Dictionary mapping each state to its value.
        """
        V = dict([(u, 0) for u in U])
        V[terminal_u] = 0
        V_error = 1

        # Debugging print statements
        print("Initial V:", V)
        print("Delta U:", delta_u)
        print("Delta R:", delta_r)

        while V_error > 0.0000001:
            V_error = 0
            for u1 in U:
                if not delta_u[u1]:  # Check if there are no outgoing transitions
                    continue
                q_u2 = []
                for u2 in delta_u[u1]:
                    if delta_r[u1][u2].get_type() == "constant":
                        r = delta_r[u1][u2].get_reward(None)
                    else:
                        r = 0  # Se la funzione di ricompensa non è costante, assume che ritorni zero
                    q_u2.append(r + gamma * V[u2])
                if q_u2:  # Ensure q_u2 is not empty
                    v_new = max(q_u2)
                    V_error = max([V_error, abs(v_new - V[u1])])
                    V[u1] = v_new
        return V


# Definizione di ConstantRewardFunction
class RewardFunction:
    def __init__(self):
        pass

    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c
