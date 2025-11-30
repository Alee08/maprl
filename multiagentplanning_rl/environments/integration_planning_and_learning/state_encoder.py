from multiagent_rlrm.multi_agent.state_encoder import StateEncoder


class StateEncoderMAPRL(StateEncoder):
    """Encodes and decodes agent positions combined with Reward Machine state."""

    def encode(self, state, state_rm=None):
        """
        Codifies the current state, the Reward Machine state, the timestep, and returns the necessary info.

        :param agent: The agent instance to access necessary agent-specific configurations.
        :param state: Dictionary representing the agent's state, including position and timestep.
        :return: A tuple (encoded_state, info) where info is a supplementary information dictionary.
        """

        num_rm_states = self.agent.get_reward_machine().numbers_state()
        # pos_x, pos_y, timestep = state["pos_x"], state["pos_y"], state["timestep"]
        # pos_x, pos_y = state["pos_x"], state["pos_y"]
        # Extract agent-specific state properties
        pos_x = state[(self.agent.name, "pos_x")]
        pos_y = state[(self.agent.name, "pos_y")]
        pos_i = state[(self.agent.name, "pos_i")]
        pos_j = state[(self.agent.name, "pos_j")]
        # timestep = state[(self.agent.name, "timestep")]

        rm_state_index = self.encode_rm_state(state_rm)

        max_x_value, max_y_value = (
            self.agent.ma_problem.grid_width,
            self.agent.ma_problem.grid_height,
        )
        max_i_value, max_j_value = (
            self.agent.ma_problem.cell_size,
            self.agent.ma_problem.cell_size,
        )
        # global_pos_index = pos_y * max_x_value + pos_x
        # local_pos_index = pos_j * max_i_value + pos_i

        # max_timestep_value = self.agent.ma_problem.max_time

        global_pos_index = pos_y * max_x_value + pos_x
        local_pos_index = pos_j * max_i_value + pos_i

        # Combina gli indici di posizione globale, locale, il tempo e lo stato della reward machine
        s = global_pos_index * max_i_value * max_j_value + local_pos_index
        encoded_state = s * num_rm_states + rm_state_index  # TODO deccomentare
        # * max_timestep_value + timestep        )

        # Calcola il numero totale di stati possibili
        total_states = (
            max_x_value
            * max_y_value
            * max_i_value
            * max_j_value
            * num_rm_states  # * max_timestep_value #TODO deccomentare
        )

        # encode_s = pos_index  # * num_timesteps + timestep
        # encode_q = rm_state_index

        # total_states = max_x_value * max_y_value * num_rm_states  # * num_timesteps
        if encoded_state >= total_states:
            print(encoded_state, total_states)
            raise ValueError("Encoded state index exceeds total state space size.")

        # Costruzione delle info
        info = {
            "global_s": global_pos_index,
            "local_s": local_pos_index,
            "s": s,
            "q": rm_state_index,
        }
        return encoded_state, info

    def decode(self, encoded_state):  # , include_rm=True):
        """
        Decodifica uno stato codificato in un dizionario di stato originale.

        :param encoded_state: Stato codificato come intero.
        :param include_rm: Booleano che indica se includere anche lo stato della Reward Machine.
        :return: Tuple (state_dict, info) dove state_dict è il dizionario dello stato originale e info contiene informazioni aggiuntive.
        """
        num_rm_states = self.agent.get_reward_machine().numbers_state()

        # Estrai lo stato RM dall'encoded_state
        s = encoded_state // num_rm_states
        rm_state_index = encoded_state % num_rm_states

        # Decodifica s in global_pos_index e local_pos_index
        max_i_value, max_j_value = (
            self.agent.ma_problem.cell_size,
            self.agent.ma_problem.cell_size,
        )
        local_pos_index = s % (max_i_value * max_j_value)
        global_pos_index = s // (max_i_value * max_j_value)

        # Decodifica global_pos_index in pos_x e pos_y
        max_x_value, max_y_value = (
            self.agent.ma_problem.grid_width,
            self.agent.ma_problem.grid_height,
        )
        pos_x = global_pos_index % max_x_value
        pos_y = global_pos_index // max_x_value

        # Decodifica local_pos_index in pos_i e pos_j
        pos_i = local_pos_index % max_i_value
        pos_j = local_pos_index // max_j_value

        """# Ottieni il nome dello stato RM a partire dall'indice
        if include_rm:
            rm_state_str = self.agent.get_reward_machine().get_state_name_from_idx(rm_state_index)
        else:
            rm_state_str = None"""

        # Costruisci il dizionario dello stato originale
        state_dict = {
            (self.agent.name, "pos_x"): pos_x,
            (self.agent.name, "pos_y"): pos_y,
            (self.agent.name, "pos_i"): pos_i,
            (self.agent.name, "pos_j"): pos_j,
        }

        # Costruisci le informazioni aggiuntive
        info = {
            "global_s": global_pos_index,
            "local_s": local_pos_index,
            "s": s,
            "q": rm_state_index,
        }

        """if include_rm:
            state_dict["rm_state"] = rm_state_str
            info["rm_state"] = rm_state_str"""

        return state_dict, info
