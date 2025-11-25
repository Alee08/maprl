from multiagent_rlrm.multi_agent.event_detector import EventDetector


class PositionEventDetector(EventDetector):
    """
    Detects events based on the agent's current position, supporting both fluent
    conditions and absolute coordinates.
    """

    def __init__(self, positions, agent):
        """
        Initializes the event detector with relevant positions.

        Args:
            positions (set): A set of tuples representing relevant positions.
            agent (AgentRL): The agent associated with this detector.
        """
        self.positions = positions
        self.agent = agent

    def detect_event(self, current_state, state_rm):
        """
        Detects events based on the agent's current position and Reward Machine state.

        Args:
            current_state (dict): The current state of the agent.
            state_rm (str): The current state of the Reward Machine.

        Returns:
            conditions (list or None): The conditions that triggered an event, or None if no event is detected.
        """
        current_location_map = self.get_current_location_map(current_state)
        # Retrieve the agent's current Reward Machine state
        state_rm_current = self.agent.reward_machine.get_current_state()
        transitions = self.agent.reward_machine.get_transitions

        # Iterate through the possible transitions
        for (state, conditions), (next_state, reward) in transitions.items():
            if state != state_rm:
                continue  # Skip if the state does not match the provided RM state

            # Handle transitions that include state "X"
            if "X" in next_state:
                self.agent.message_conditions = conditions

            # If the agent is in state "X", manage the communication
            if "X" in state_rm_current:
                self.handle_state_X(conditions, current_location_map)
                if self.check_conditions(
                    conditions, current_location_map, current_state
                ):
                    return conditions  # Event detected
            else:
                # Check whether the local conditions are satisfied
                if self.check_local_conditions(
                    conditions, current_location_map, current_state
                ):
                    return conditions  # Event detected

        return None  # No event detected

    def get_current_location_map(self, current_state):
        """
        Retrieves the current location map based on the agent's position.

        Args:
            current_state (dict): The current state of the agent.

        Returns:
            tuple: A tuple representing the fluent and its value (fluent, True).
        """
        pos_x_key = (self.agent.name, "pos_x")
        pos_y_key = (self.agent.name, "pos_y")
        pos_x = current_state[pos_x_key]
        pos_y = current_state[pos_y_key]
        current_location = self.get_location_by_coordinates(self.agent, pos_x, pos_y)
        fluent = f"pos({current_location})"
        return (fluent, True)

    def handle_state_X(self, conditions, current_location_map):
        """
        Handles communication when the agent is in state 'X' of the Reward Machine.

        Args:
            conditions (list): The conditions that must be satisfied.
            current_location_map (dict): The current mapping of the agent's location.
        """
        list_agents_mess = self.extract_agent_ids(conditions)
        self.agent._send_message(list_agents_mess, self.agent.message_conditions)
        # Non impostare message_sent qui; gestiscilo nell'agente se necessario

    def check_conditions(self, conditions, current_location_map, current_state):
        """
        Checks if all conditions (local and message-based) are satisfied.

        Args:
            conditions (list): The conditions to be verified.
            current_location_map (dict): The current mapping of the agent's location.
            current_state (dict): The current state of the agent.

        Returns:
            bool: True if all conditions are satisfied, False otherwise.
        """
        if not self.check_local_conditions(
            conditions, current_location_map, current_state
        ):
            return False
        if not self.check_message_conditions(conditions):
            return False
        return True

    def check_local_conditions(self, conditions, current_location_map, current_state):
        """
        Checks if the local conditions are satisfied.

        Args:
            conditions (list): The local conditions to verify.
            current_location_map (dict): The current mapping of the agent's location.
            current_state (dict): The current state of the agent.

        Returns:
            bool: True if all local conditions are satisfied, False otherwise.
        """
        for condition in conditions:
            if not isinstance(condition[0], tuple) or isinstance(
                condition[0][0], int
            ):  # Condizione locale
                if not self.check_local_condition(
                    condition, current_location_map, current_state
                ):
                    return False
        return True

    def check_local_condition(self, condition, current_location_map, current_state):
        """
        Checks a single local condition.

        Args:
            condition (tuple): A single condition to be verified.
            current_location_map (dict): The current mapping of the agent's location.
            current_state (dict): The current state of the agent.

        Returns:
            bool: True if the condition is satisfied, False otherwise.
        """
        key, expected_value = condition

        # If the condition is an absolute coordinate (tuple of integers)
        if isinstance(key, tuple) and all(isinstance(coord, int) for coord in key):
            # Retrieve the agent's current absolute position
            absolute_position = self.get_absolute_position(current_state)
            return key == absolute_position and expected_value == True
        else:
            # The condition is a fluent (for example, pos(l13))
            fluent = str(key)
            actual_fluent, actual_value = current_location_map
            return fluent == actual_fluent and expected_value == actual_value

    def get_absolute_position(self, current_state):
        """
        Calculates the absolute position of the agent on the map.

        Args:
            current_state (dict): The current state of the agent.

        Returns:
            tuple: The absolute (x, y) coordinates of the agent.
        """
        pos_x_key = (self.agent.name, "pos_x")
        pos_y_key = (self.agent.name, "pos_y")
        pos_i_key = (self.agent.name, "pos_i")
        pos_j_key = (self.agent.name, "pos_j")
        pos_x = current_state[pos_x_key]
        pos_y = current_state[pos_y_key]
        pos_i = current_state[pos_i_key]
        pos_j = current_state[pos_j_key]
        absolute_x = pos_x * self.agent.ma_problem.cell_size + pos_i
        absolute_y = pos_y * self.agent.ma_problem.cell_size + pos_j
        return (absolute_x, absolute_y)

    def check_message_conditions(self, conditions):
        """
        Checks if the message-based conditions are satisfied.

        Args:
            conditions (list): The conditions to be verified.

        Returns:
            bool: True if all message-based conditions are satisfied, False otherwise.
        """
        dic_messaggi = self.agent.return_messages()
        if not dic_messaggi:
            return False  # No messages received, conditions not satisfied
        for condition in conditions:
            if isinstance(condition[0], tuple):  # Message-based condition
                if not self.check_message_condition(condition, dic_messaggi):
                    return False
        return True

    def check_message_condition(self, condition, dic_messaggi):
        """
        Checks a single message-based condition.

        Args:
            condition (tuple): A condition involving an agent and its expected value.
            dic_messaggi (dict): The dictionary of messages received by the agent.

        Returns:
            bool: True if the condition is satisfied, False otherwise.
        """
        (agent_name, key), expected_value = condition
        actual_value = dic_messaggi.get((agent_name, key))
        return actual_value == expected_value

    def get_location_by_coordinates(self, agent, x, y):
        """
        Retrieves the location associated with the given (x, y) coordinates for a specific agent.

        Args:
            agent (AgentRL): The agent whose location is being retrieved.
            x (int): The x-coordinate of the location.
            y (int): The y-coordinate of the location.

        Returns:
            Location: The location object corresponding to the given coordinates.
        """
        return self.agent.ma_problem.coord_to_location_map.get((agent.name, (x, y)))

    def extract_agent_ids(self, conditions):
        """
        Extracts the identifiers of agents involved in the given conditions.

        Args:
            conditions (list): The conditions from which to extract agent IDs.

        Returns:
            list: A list of agent identifiers.
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

    def add_events(self, new_events):
        """
        Adds new events to the EventDetector.

        Args:
            new_events (iterable): An iterable containing the new events to be added.
        """
        self.positions.update(new_events)
