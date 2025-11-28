"""A utility for building transition rules for multi-agent scheduling."""

import pickle
from typing import Any, Dict, List, Optional, Set, Union, cast

import networkx as nx
import unified_planning as up
import unified_planning.model.walkers as walkers
import unified_planning.plans as plans
from unified_planning.exceptions import UPUsageError, UPValueError
from unified_planning.model import FNode, InstantaneousAction, Expression
from unified_planning.model.operators import OperatorKind
from unified_planning.plot import show_partial_order_plan


# import maze 5 agents plan
with open("planning_utils/result_maze_5_agents.pkl", "rb") as file:
    pop_plan = pickle.load(file)
with open("planning_utils/problem_maze_5_agents.pkl", "rb") as file:
    problem = pickle.load(file)

# show_partial_order_plan(pop_plan.plan, "pop_plan")

######################################Direct Dependecies Between Actions######################################
# The function that establishes dependencies between actions
def directly_depends_on(partial_order_plan, problem):
    subs = partial_order_plan._environment.substituter
    simp = partial_order_plan._environment.simplifier
    eqr = walkers.ExpressionQuantifiersRemover(partial_order_plan._environment)
    fve = partial_order_plan._environment.free_vars_extractor

    graph = partial_order_plan._graph
    last_modifier = (
        {}
    )  # Mapping of (fluent, agent_name) to action that last modified it
    last_modifier_env = (
        {}
    )  # Mapping of fluent to action that last modified it if fluent is environment-owned
    direct_dependencies = {}
    dependency_data = []

    for action_instance in graph.nodes():
        inst_action = cast(InstantaneousAction, action_instance.action)
        required_fluents: Set[FNode] = set()
        lifted_required_fluents: Set[FNode] = set()

        # Gathering fluents required by the action's preconditions
        for prec in inst_action.preconditions:
            lifted_required_fluents |= fve.get(eqr.remove_quantifiers(prec, problem))

        # Assign actual parameters to the lifted fluents
        assignments_action = dict(
            zip(inst_action.parameters, action_instance.actual_parameters)
        )
        for lifted_fluent in lifted_required_fluents:
            required_fluents |= {
                simp.simplify(subs.substitute(lifted_fluent, assignments_action))
            }

        # Determine dependencies for each required fluent
        for required_fluent in required_fluents:
            try:
                # Check if fluent is owned by the environment
                if problem.ma_environment.fluent(required_fluent.fluent().name):
                    required_fluent_last_modifier = last_modifier_env.get(
                        required_fluent, None
                    )
            except UPValueError:
                # If fluent is not environment-owned, treat it as agent-specific
                required_fluent_last_modifier = last_modifier.get(
                    (required_fluent, action_instance.agent.name), None
                )

            if required_fluent_last_modifier is not None:
                # Add the action that last modified the required fluent as a dependency
                direct_dependencies.setdefault(action_instance, []).append(
                    required_fluent_last_modifier
                )

                # Collect dependency data
                dependency_info = {
                    "current_action": action_instance,
                    "dependent_agent": required_fluent_last_modifier.agent.name
                    if hasattr(required_fluent_last_modifier, "agent")
                    else None,
                    "dependent_action": required_fluent_last_modifier,
                    "matching_condition": required_fluent,
                }
                dependency_data.append(dependency_info)

        # Update last modifiers for each effect of the action
        for effect in inst_action.effects:
            for eff in effect.expand_effect(problem):
                grounded_fluent = simp.simplify(
                    subs.substitute(eff.fluent, assignments_action)
                )
                try:
                    # Check if the fluent is environment-owned
                    if problem.ma_environment.fluent(grounded_fluent.fluent().name):
                        last_modifier_env[grounded_fluent] = action_instance
                except UPValueError:
                    # If not, record it as agent-specific
                    last_modifier[
                        (grounded_fluent, action_instance.agent.name)
                    ] = action_instance

    return direct_dependencies, dependency_data


# pop_plan.plan._graph.nodes()
direct_dep, data_dep = directly_depends_on(pop_plan.plan, problem)
# direct_dep #dependencies of each action in the plan
# The code examines actions in a partial-order multi-agent planning problem, establishing dependencies between actions
# based on the latest modifications to fluents. It records for each action the previous actions required to fulfill its
# preconditions (direct dependencies), differentiating between fluents modified by specific agents and those modified by
# the shared environment.

# Last Modifier Management: The code keeps two separate maps, last_modifier for agent-specific fluents and
# last_modifier_env for environment fluents. This is necessary because an environment fluent can be modified by any agent
# and therefore needs global tracking, while agent-specific fluents are modified only by the agent they belong to and
# therefore the tracking is specific to that agent.
##################################################################################################################

##########################################Identify concurrent actions##########################################


def compute_required_fluents(
    inst_action: InstantaneousAction,
    action_instance: up.plans.plan.ActionInstance,
    problem: Any,
    subs: Any,
    simp: Any,
    eqr: walkers.ExpressionQuantifiersRemover,
    fve: walkers.FreeVarsExtractor,
) -> tuple[set[FNode], dict]:
    """Returns the requested fluents and assignments for the specific action."""

    lifted_required_fluents: Set[FNode] = set()
    for prec in inst_action.preconditions:
        lifted_required_fluents |= fve.get(eqr.remove_quantifiers(prec, problem))

    assignments_action = dict(
        zip(inst_action.parameters, action_instance.actual_parameters)
    )
    required_fluents: Set[FNode] = {
        simp.simplify(subs.substitute(lifted_fluent, assignments_action))
        for lifted_fluent in lifted_required_fluents
    }
    return required_fluents, assignments_action


def compute_grounded_fluents(
    inst_action: InstantaneousAction,
    assignments_action: dict,
    problem: Any,
    simp: Any,
    subs: Any,
) -> set[FNode]:
    """Calculate the fluents resulting from the effects of the action."""

    grounded_fluents: Set[FNode] = set()
    for effect in inst_action.effects:
        for eff in effect.expand_effect(problem):
            grounded_fluents.add(
                simp.simplify(subs.substitute(eff.fluent, assignments_action))
            )
    return grounded_fluents


sequenza = {}
# actions = seq_plan_.actions

# Topological ordering of nodes in the plane graph
ordered_nodes = nx.topological_sort(pop_plan.plan._graph)
ordered_nodes_list = list(ordered_nodes)

# Assume these are functions or classes imported from your environment
subs = pop_plan.plan._environment.substituter
simp = pop_plan.plan._environment.simplifier
eqr = walkers.ExpressionQuantifiersRemover(pop_plan.plan._environment)
fve = pop_plan.plan._environment.free_vars_extractor


def remove_fluents_if_owned_by_environment(fluents, environment):
    """
    Removes fluents from the given set if they are owned by the environment.

    :param fluents: A set of fluents to be cleaned.
    :param environment: The environment to check for fluent ownership.
    :return: A new set with the owned fluents removed.
    """
    to_remove = set()
    for fluent in fluents:
        try:
            if environment.fluent(fluent.fluent().name):
                to_remove.add(fluent)
        except UPValueError:
            # Ignore the error and move on to the next fluent
            pass
    # print(fluents, "rimuovooooooooooooooo", to_remove)
    return fluents - to_remove


def some_comparison_function(
    current_required_fluents,
    current_grounded_fluent,
    next_required_fluents,
    next_grounded_fluent,
    problem,
):
    """
    Compares sets of current and next required and grounded fluents for equality,
    after removing those owned by the environment.

    :param current_required_fluents: Set of current required fluents.
    :param current_grounded_fluent: Set of current grounded fluents.
    :param next_required_fluents: Set of next required fluents.
    :param next_grounded_fluent: Set of next grounded fluents.
    :param problem: The problem context containing the environment.
    :return: True if the cleaned sets of current and next fluents are equal, False otherwise.
    """
    # Remove fluents owned by the environment from the current and next fluent sets
    current_grounded_clean = remove_fluents_if_owned_by_environment(
        current_grounded_fluent, problem.ma_environment
    )
    next_grounded_clean = remove_fluents_if_owned_by_environment(
        next_grounded_fluent, problem.ma_environment
    )
    current_required_clean = remove_fluents_if_owned_by_environment(
        current_required_fluents, problem.ma_environment
    )
    next_required_clean = remove_fluents_if_owned_by_environment(
        next_required_fluents, problem.ma_environment
    )

    # Compare the cleaned sets of fluents for equality
    return (
        current_required_clean == next_required_clean
        and current_grounded_clean == next_grounded_clean
    )


# Now you can call the function with your fluents and the problem environment
# result = some_comparison_function(current_required_fluents, current_grounded_fluent, next_required_fluents, next_grounded_fluent, problem)


in_sequence = False
current_sequence = []
previous_action = None  # Questa è l'azione che precede la sequenza corrente
next_sequence_action = (
    None  # Questa sarà l'azione che segue l'ultima azione della sequenza corrente
)


for i in range(len(ordered_nodes_list) - 1):
    current_action = ordered_nodes_list[i]
    next_action = ordered_nodes_list[i + 1] if i + 1 < len(ordered_nodes_list) else None

    current_inst_action = cast(InstantaneousAction, current_action.action)
    next_action_inst_action = cast(InstantaneousAction, next_action.action)

    current_required_fluents, current_assignments_action = compute_required_fluents(
        current_inst_action, current_action, problem, subs, simp, eqr, fve
    )
    next_required_fluents, next_assignments_action = compute_required_fluents(
        next_action_inst_action, next_action, problem, subs, simp, eqr, fve
    )

    current_grounded_fluent = compute_grounded_fluents(
        current_inst_action, current_assignments_action, problem, simp, subs
    )
    next_grounded_fluent = compute_grounded_fluents(
        next_action_inst_action, next_assignments_action, problem, simp, subs
    )

    # Check if the current and next actions are in sequence
    if next_action and some_comparison_function(
        current_required_fluents,
        current_grounded_fluent,
        next_required_fluents,
        next_grounded_fluent,
        problem,
    ):
        if not in_sequence:
            # If we are not in a sequence, we start a new sequence with the current action
            in_sequence = True
            current_sequence = [current_action]
        else:
            # If we are already in a sequence, we add the current action to the sequence
            current_sequence.append(current_action)
    else:
        # If the sequence is interrupted or we are at the last action
        if in_sequence:
            # If we were in a sequence, we add the current action to the sequence
            current_sequence.append(current_action)
            if next_action:
                # If there is a subsequent action, we also include it in the sequence
                current_sequence.append(next_action)
            # Save the sequence with the previous action as the key
            sequenza[previous_action] = current_sequence
            # Let's prepare for the next sequence
            current_sequence = []
            in_sequence = False
        previous_action = current_action  # We set the current action as the precedent for the next iteration

# Print or return the sequence dictionary
print(sequenza)


# *   Initialization: Sets up an empty dictionary to track sequences of actions, sorts the actions topologically from a multi-agent action plan, and establishes necessary functions for fluents manipulation and actions comparison.
# *   Actions Iteration: Sequentially iterates through the ordered actions, determining whether each action belongs to a sequence based on its relationship with the subsequent action.

# * Actions Iteration: Sequentially iterates through the ordered actions, determining whether each action belongs to a sequence based on its relationship with the subsequent action.
# * Actions Comparison: Uses a comparison function to decide if two consecutive actions should be grouped in the same sequence by examining the fluents required and modified by each action.
# * Sequences Building: Groups actions into sequences based on this dependency relationship and assigns them to a key in the dictionary, which is the action that precedes the start of the sequence.
# * Next Action Inclusion: Includes the action immediately following the end of each sequence in the value associated with its corresponding key in the dictionary.
# * End of Sequences Handling: When a sequence ends or the last action is reached, the code saves the current sequence in the dictionary before moving to the next one or finishing if the end of the action list has been reached.

##################################I find the dependencies of concurrent actions##################################

dipendenze_dirette = direct_dep
nuovo_piano = {}
azioni_gia_modificate = set()
azioni_end = [
    sequenza_azioni[-1] for sequenza_azioni in sequenza.values() if sequenza_azioni
]


def filtra_fluenti_ambientali(azione, environment):
    """Removes environmental influences from preconditions and effects of a cloned action."""
    nuove_precondizioni = []
    nuovi_effetti = []

    for prec in azione.preconditions:
        try:
            if not environment.fluent(prec.fluent().name):
                nuove_precondizioni.append(prec)
        except UPValueError:
            nuove_precondizioni.append(prec)

    for eff in azione.effects:
        try:
            if not environment.fluent(eff.fluent.fluent().name):
                nuovi_effetti.append(eff)
        except UPValueError:
            nuovi_effetti.append(eff)

    azione.clear_preconditions()
    azione.clear_effects()
    for prec in nuove_precondizioni:
        azione.add_precondition(prec)
    for eff in nuovi_effetti:
        azione.add_effect(eff.fluent, eff.value, eff.condition)


def deve_essere_aggiunta(azione):
    """Determines whether the action should be added to the new plan."""
    return azione not in sequenza.keys() and azione not in azioni_end


# Funzione per filtrare le dipendenze
def filtra_dipendenze(dipendenze):
    """Remove dependencies already covered by final sequences or actions."""
    return [
        dip
        for dip in dipendenze
        if dip not in sequenza.keys() and dip not in azioni_end and dip
    ]


dipendenze_aggregate = {}

for chiave, azioni in sequenza.items():
    if chiave not in dipendenze_aggregate:
        dipendenze_aggregate[chiave] = []

    for azione in azioni:
        if azione in dipendenze_dirette:
            dipendenze_filtrate = filtra_dipendenze(dipendenze_dirette[azione])
            for dipendenza in dipendenze_filtrate:
                if (
                    dipendenza not in dipendenze_aggregate[chiave]
                    and dipendenza not in sequenza[chiave]
                ):
                    dipendenze_aggregate[chiave].append(dipendenza)

for azione, dipendenze in dipendenze_dirette.items():
    if deve_essere_aggiunta(azione) and any(
        azione in sequenza_azioni for sequenza_azioni in sequenza.values()
    ):
        nome_concorrente = f"{azione.action.name}_concurrent"
        if nome_concorrente not in azioni_gia_modificate:
            nuova_azione = azione.action.clone()
            nuova_azione.name = nome_concorrente
            azioni_gia_modificate.add(nome_concorrente)

            # Filtra i fluenti ambientali dalle precondizioni e dagli effetti
            filtra_fluenti_ambientali(nuova_azione, problem.ma_environment)
            azioni_gia_modificate.add(nome_concorrente)

        chiave_di_sequenza = next((k for k, v in sequenza.items() if azione in v), None)
        dipendenze_da_usare = filtra_dipendenze(
            dipendenze_aggregate.get(chiave_di_sequenza, [])
        )

        nuova_azione_instance = up.plans.plan.ActionInstance(
            nuova_azione, azione.actual_parameters, agent=azione.agent
        )
        nuovo_piano[nuova_azione_instance] = dipendenze_da_usare
    elif deve_essere_aggiunta(azione):
        nuovo_piano[azione] = filtra_dipendenze(dipendenze)

print("New plan:", nuovo_piano)

################################ SequentialValidator ################################
# Applied Definition 8 of:
# Macros, Reactive Plans and Compact Representations-Christer Bäckström and Anders Jonssonand Peter Jonsson
# [Articolo](https://www.researchgate.net/publication/287321934_Macros_reactive_plans_and_compact_representations)


class SequentialPlanValidator:
    def __init__(self, pop_plan):
        self.pop_plan = pop_plan
        self.subs = pop_plan.plan._environment.substituter
        self.simp = pop_plan.plan._environment.simplifier
        self.eqr = walkers.ExpressionQuantifiersRemover(pop_plan.plan._environment)
        self.fve = pop_plan.plan._environment.free_vars_extractor
        self.eff_cumulativi_per_agente = {}
        self.eff_cumulativi_environment = {}

    def verifica_conflitti(self):
        effetti_environment = self.eff_cumulativi_environment.get("env", {})
        effetti_environment_set = set(effetti_environment.items())

        # Check environmental conflicts
        if any(
            self._verifica_conflitto(
                fluente, valore, self.precondizioni_per_agente.get(agente, {}), agente
            )
            for fluente, valore in effetti_environment_set
            for agente, precondizioni in self.precondizioni_per_agente.items()
        ):
            return True

        # Check for conflicts for each agent
        for agente, effetti_agente in self.eff_cumulativi_per_agente.items():
            effetti_agente_set = set(effetti_agente.items())
            if any(
                self._verifica_conflitto(
                    fluente,
                    valore,
                    self.precondizioni_per_agente.get(agente, {}),
                    agente,
                )
                for fluente, valore in effetti_agente_set
            ):
                return True

        return False

    def _verifica_conflitto(self, fluente, valore, precondizioni, agente):
        valore_precondizione = precondizioni.get(fluente)
        if valore_precondizione is not None and valore_precondizione != valore:
            print(
                f"Conflict detected: fluent {fluente.fluent().name} for agent {agente}"
            )
            return True
        return False

    def validate(self, actions, problem):
        found_contradition = False
        for i in range(len(actions) - 1):
            action = actions[i]
            next_action = actions[i + 1]

            # Update effects for the current agent
            effetti_agente = self.eff_cumulativi_per_agente.setdefault(
                action.agent.name, {}
            )
            effetti_environment = self.eff_cumulativi_environment.setdefault("env", {})
            assignments_action = dict(
                zip(action.action.parameters, action.actual_parameters)
            )
            for effect in action.action.effects:
                for eff in effect.expand_effect(problem):
                    grounded_fluent = self.simp.simplify(
                        self.subs.substitute(eff.fluent, assignments_action)
                    )
                    effetti_agente[grounded_fluent] = eff.value.bool_constant_value()
                    if grounded_fluent.fluent() not in action.agent.fluents:
                        effetti_environment[
                            grounded_fluent
                        ] = eff.value.bool_constant_value()

            # Prepare preconditions for the next action agent
            self.precondizioni_per_agente = {
                next_action.agent.name: self._prepare_preconditions(
                    next_action, problem
                )
            }

            # Compare the effects with the preconditions of the next action
            contraddizione = self.verifica_conflitti()
            if contraddizione:
                found_contradition = True
            self._print_results(action, next_action, contraddizione, i)
        if not found_contradition:
            print("Valid Plan!")

    def _prepare_preconditions(self, next_action, problem):
        prec_agente = {}
        for prec in next_action.action.preconditions:
            lifted_fluents = self.fve.get(self.eqr.remove_quantifiers(prec, problem))
            assignments_next_action = dict(
                zip(next_action.action.parameters, next_action.actual_parameters)
            )
            for lifted_fluent in lifted_fluents:
                grounded_fluent = self.simp.simplify(
                    self.subs.substitute(lifted_fluent, assignments_next_action)
                )
                prec_agente[grounded_fluent] = not (prec.is_not())
        return prec_agente

    def _print_results(self, action, next_action, contradiction, index):

        if contradiction:
            print("Current action: ", action)
            print("Next action: ", next_action)
            print("Cumulative effects per agent: ", self.eff_cumulativi_per_agente)
            print("Preconditions per agent: ", self.precondizioni_per_agente)
            print(
                f"Contradiction between the effects of action {index} (agent {action.agent.name}) "
                f"and the preconditions of action {index + 1} (agent {next_action.agent.name})\n"
            )


# Uso della classe
liste_di_sequenze = [[chiave] + valori for chiave, valori in sequenza.items()]
first_sequenza = liste_di_sequenze[0]

validator = SequentialPlanValidator(pop_plan)
validator.validate(first_sequenza, problem)


################################## estrai_cambiamenti_fluenti ##################################
def estrai_cambiamenti_fluenti_per_rm(azione_con_dipendenze, problem):
    cambiamenti_per_rm = {}
    for azione, dipendenze in azione_con_dipendenze.items():
        inst_action = cast(InstantaneousAction, azione.action)

        # Inizializza la lista dei cambiamenti dell'azione corrente e delle dipendenze
        cambiamenti_azione_corrente = []
        cambiamenti_dipendenze = []

        # Estrai i cambiamenti dell'azione corrente
        assignments_action = dict(zip(inst_action.parameters, azione.actual_parameters))
        for effect in inst_action.effects:
            for eff in effect.expand_effect(problem):
                grounded_fluent = simp.simplify(
                    subs.substitute(eff.fluent, assignments_action)
                )
                nuovo_valore = (
                    eff.value.bool_constant_value()
                    if isinstance(eff.value, FNode)
                    else eff.value
                )
                cambiamenti_azione_corrente.append(
                    (grounded_fluent, azione.agent.name, nuovo_valore)
                )

        # Estrai i cambiamenti delle azioni dipendenti
        for dipendenza in dipendenze:
            inst_dipendenza = cast(InstantaneousAction, dipendenza.action)
            assignments_dipendenza = dict(
                zip(inst_dipendenza.parameters, dipendenza.actual_parameters)
            )
            for effect in inst_dipendenza.effects:
                for eff in effect.expand_effect(problem):
                    grounded_fluent = simp.simplify(
                        subs.substitute(eff.fluent, assignments_dipendenza)
                    )
                    nuovo_valore = (
                        eff.value.bool_constant_value()
                        if isinstance(eff.value, FNode)
                        else eff.value
                    )
                    cambiamenti_dipendenze.append(
                        (grounded_fluent, dipendenza.agent.name, nuovo_valore)
                    )

        # Aggiungi i cambiamenti al dizionario per RM
        cambiamenti_per_rm[inst_action.name] = {
            "current_action": cambiamenti_azione_corrente,
            "dependencies": cambiamenti_dipendenze,
        }

    return cambiamenti_per_rm


# Esempio di utilizzo
risultati = {}  # Inizializza come dizionario
for k, v in nuovo_piano.items():
    if k.agent.name not in risultati:
        risultati[
            k.agent.name
        ] = []  # Crea una nuova lista per questo agente se non esiste già
    risultati[k.agent.name].append({k: v})  # Aggiungi l'azione alla lista dell'agente

# Now results is a dictionary with actions grouped by agent.
# Usage example
cambiamenti_per_rm = estrai_cambiamenti_fluenti_per_rm(risultati["a2"][4], problem)
print(cambiamenti_per_rm)
# cambiamenti_per_rm

acts_ = []
dic = {}
for j, k in risultati.items():
    acts_ = []
    for i in k:
        # print(j, i, k)
        acts_.append(estrai_cambiamenti_fluenti_per_rm(i, problem))
    dic[j] = acts_


def aggiorna_dipendenze_concurrent(acts, sequenza):
    for nome_sequenza, azioni_sequenza in sequenza.items():
        # Map the base names of the actions to their respective agents
        mappa_azioni_agenti = {
            azione.action.name.split("_")[0]: azione.agent.name
            for azione in azioni_sequenza[:-1]
        }  # Exclude last action

        for agent, azioni in acts.items():
            for azione in azioni:
                nome_azione = list(azione.keys())[0]
                nome_base = nome_azione.split("_")[0]

                # Check if the action is part of the sequence and needs to be updated
                if nome_base in mappa_azioni_agenti and "_concurrent" in nome_azione:
                    azione_corrente = azione[nome_azione]["current_action"]
                    dipendenze_attuali = set(
                        tuple(d[:2]) for d in azione[nome_azione]["dependencies"]
                    )

                    # Check if there are fluent dependencies of all the agents in the sequence
                    for altro_nome_base, altro_agent in mappa_azioni_agenti.items():
                        fluente_manca = all(
                            (fl[0], altro_agent) not in dipendenze_attuali
                            for fl in azione_corrente
                        )
                        if fluente_manca:
                            print(
                                f"A fluent is missing {altro_agent} nelle dependencies of {nome_azione} of {agent}"
                            )
                            # Add missing fluents to dependencies
                            for azione_altra in acts[altro_agent]:
                                if (
                                    list(azione_altra.keys())[0]
                                    == altro_nome_base + "_concurrent"
                                ):
                                    for fluente in azione_altra[
                                        altro_nome_base + "_concurrent"
                                    ]["current_action"]:
                                        if fluente[2] == False:
                                            nuovo_fluente = (
                                                fluente[0],
                                                altro_agent,
                                                True,
                                            )
                                            if (
                                                nuovo_fluente[0],
                                                nuovo_fluente[1],
                                            ) not in dipendenze_attuali:
                                                azione[nome_azione][
                                                    "dependencies"
                                                ].append(nuovo_fluente)

    return acts


# Applica la funzione ai dati di esempio
nuovo_acts_aggiornato = aggiorna_dipendenze_concurrent(dic, sequenza)
print(nuovo_acts_aggiornato)
# nuovo_acts_aggiornato


##################################Build RM##################################
nuvo_dic = nuovo_acts_aggiornato
transitions = {}
current_state = "state1"
reward = 10
state_counter = 2
RM_dict = {}
for agent, actions in nuvo_dic.items():
    current_state = "state1"
    reward = 10
    state_counter = 2
    transitions = {}

    for i, action in enumerate(actions):
        action_key = list(action.keys())[0]  # Ottieni il nome dell'azione
        action_fluents = action[action_key]["current_action"]
        dependencies = action[action_key]["dependencies"]

        # If the action is concurrent and the first of the agent
        if "_concurrent" in action_key and i == 0:
            # Use action dependencies as a condition for transitioning from state1 to state1X
            condition_for_state1X = tuple(
                (fluent, value)
                for fluent, agent_id, value in dependencies
                if agent_id == agent
            )
            # Add transition from state1 to state1X
            transitions[(current_state, condition_for_state1X)] = ("state1X", reward)
            current_state = "state1X"
            reward += 10

        if "_concurrent" not in action_key:
            # Non-competing actions
            condition = tuple(
                (fluent, value) if agent_id != agent else (fluent, value)
                for fluent, agent_id, value in action_fluents
            )
            next_state = (
                f"state{state_counter}X"
                if i < len(actions) - 1
                and "_concurrent" in list(actions[i + 1].keys())[0]
                else f"state{state_counter}"
            )
            transitions[(current_state, condition)] = (next_state, reward)
            current_state = next_state
        else:
            # Competing actions
            dependency_condition = tuple(
                ((agent_id, fluent), value) if agent_id != agent else (fluent, value)
                for fluent, agent_id, value in dependencies
            )
            final_fluents = tuple(
                ((agent_id, fluent), value) if agent_id != agent else (fluent, value)
                for fluent, agent_id, value in action_fluents
            )
            next_state = f"state{state_counter}"

            transitions[(current_state, dependency_condition)] = (next_state, reward)
            transitions[(next_state, final_fluents)] = (
                f"state{state_counter + 1}",
                reward + 10,
            )
            current_state = f"state{state_counter + 1}"

        state_counter += 1
        reward += 10

    RM_dict[agent] = transitions


print("\n\n\n", RM_dict)


def rimuovi_fluenti_falsi(RM_dict):
    """Scarta le condizioni con valore booleano falso dalle transizioni."""
    RM_dict_pulito = {}
    for agente, transizioni in RM_dict.items():
        transizioni_pulite = {}
        for (stato_corrente, condizioni), (
            stato_successivo,
            ricompensa,
        ) in transizioni.items():
            # Filter the conditions to keep only the true ones
            condizioni_vere = tuple(
                condizione for condizione in condizioni if condizione[1]
            )
            if (
                condizioni_vere
            ):  # If there are true conditions, add them to the transitions
                transizioni_pulite[(stato_corrente, condizioni_vere)] = (
                    stato_successivo,
                    ricompensa,
                )
        RM_dict_pulito[agente] = transizioni_pulite
    return RM_dict_pulito


# Apply the function to RM_dict
RM_dict_true = rimuovi_fluenti_falsi(RM_dict)


def rm_concurrent_sequence(rm_dictionary):
    """Stores transactions related to concurrent states and those immediately following them."""
    new_dict = {}
    for agent, transitions in rm_dictionary.items():
        new_transitions = {}
        x_state_found = False
        transitions_list = list(transitions.items())

        for index, ((current_state, conditions), (next_state, reward)) in enumerate(
            transitions_list
        ):
            # If the current state ends with 'X', add the transition and mark that we found a state 'X'
            if "X" in current_state:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = True
            # If we found a state 'X' and this is the immediately following transition, add it
            elif x_state_found:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = False  # Resetta il flag dopo aver aggiunto la transizione successiva
            # If the next state ends with 'X', add the transition and mark that we have found a state 'X'
            elif "X" in next_state:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = True

        # Add the new filtered transitions for the current agent to the new dictionary
        new_dict[agent] = new_transitions
    return new_dict


RM_dict_true_seq = rm_concurrent_sequence(RM_dict_true)
