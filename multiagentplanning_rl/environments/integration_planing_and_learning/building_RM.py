"""import subprocess

def download_file(url, file_name):
    try:
        # Costruisce il comando curl
        command = ['curl', '-L', '-o', file_name, url]

        # Esegue il comando
        subprocess.run(command, check=True)
        print(f"File scaricato con successo: {file_name}")
    except subprocess.CalledProcessError:
        print(f"Errore durante il download del file: {file_name}")

# URL e nomi file da scaricare
urls_and_files = [
    ("https://drive.google.com/uc?export=download&id=1zljnr76rj6-dJUnkgfhZrt0LTfxsU9UG", "result_maze_5_agents.pkl"),
    ("https://drive.google.com/uc?export=download&id=17eJ--w0Khwm-oxQxz885Irhc3EkPXxaR", "problem_maze_5_agents.pkl")
]

# Esegue il download per ogni coppia URL/nome file
for url, file_name in urls_and_files:
    download_file(url, file_name)"""

import pickle
import networkx as nx
import unified_planning
import unified_planning.model.fluent
import collections
from unified_planning.environment import Environment
from unified_planning.model.operators import OperatorKind
from typing import Dict, List, Optional, Set, Union
from fractions import Fraction
from unified_planning.model.fnode import FNode
from collections import defaultdict
from unified_planning.io.ma_pddl_writer import MAPDDLWriter


# Apri il file in modalità di lettura binaria


# import maze 5 agents plan
with open("planning_utils/result_maze_5_agents.pkl", "rb") as file:
    pop_plan = pickle.load(file)
# import maze 5 agents problem
with open("planning_utils/problem_maze_5_agents.pkl", "rb") as file:
    problem = pickle.load(file)


from unified_planning.plot import show_partial_order_plan

# show_partial_order_plan(pop_plan.plan, "pop_plan")

import unified_planning as up
import unified_planning.plans as plans
import unified_planning.model.walkers as walkers
from unified_planning.environment import Environment
from unified_planning.exceptions import UPUsageError
from unified_planning.model import FNode, InstantaneousAction, Expression
from typing import Dict, List, Any, Set, cast
from unified_planning.exceptions import UPValueError

######################################Direct Dependecies Between Actions######################################
# La funzione che stabilisce le dipendenze tra azioni
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
from typing import Dict, List, Any, Set, cast
from unified_planning.model import InstantaneousAction

# Inizializzazione delle strutture dati
sequenza = {}
# azioni = seq_plan_.actions

# Ordinamento topologico dei nodi nel grafico del piano
ordered_nodes = nx.topological_sort(pop_plan.plan._graph)
ordered_nodes_list = list(ordered_nodes)

# Assumi che queste siano funzioni o classi importate dal tuo ambiente
subs = pop_plan.plan._environment.substituter
simp = pop_plan.plan._environment.simplifier
eqr = walkers.ExpressionQuantifiersRemover(pop_plan.plan._environment)
fve = pop_plan.plan._environment.free_vars_extractor

"""def some_comparison_function(current_required_fluents, current_grounded_fluent, next_required_fluents, next_grounded_fluent):
    # Implementa la tua logica specifica qui
    # Questa funzione determina se c'è una relazione sequenziale tra le azioni basata sui fluenti
    fluents_set = problem.ma_environment.fluents
    to_remove = set()
    to_remove2 = set()"""


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
    # print("\n\n quiiiiiiiii", current_grounded_clean, next_grounded_clean, current_required_clean, next_required_clean)
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
    current_required_fluents = set()
    current_lifted_required_fluents = set()
    next_action_inst_action = cast(InstantaneousAction, next_action.action)
    next_required_fluents = set()
    next_lifted_required_fluents = set()

    current_grounded_fluent = set()
    next_grounded_fluent = set()

    # Raccogliere fluenti richiesti dalle precondizioni delle azioni
    for prec in current_inst_action.preconditions:
        current_lifted_required_fluents |= fve.get(
            eqr.remove_quantifiers(prec, problem)
        )
    for prec in next_action_inst_action.preconditions:
        next_lifted_required_fluents |= fve.get(eqr.remove_quantifiers(prec, problem))

    # Assegnare i parametri attuali ai fluenti sollevati
    current_assignments_action = dict(
        zip(current_inst_action.parameters, current_action.actual_parameters)
    )
    for lifted_fluent in current_lifted_required_fluents:
        current_required_fluents |= {
            simp.simplify(subs.substitute(lifted_fluent, current_assignments_action))
        }
    next_assignments_action = dict(
        zip(next_action_inst_action.parameters, next_action.actual_parameters)
    )
    for lifted_fluent in next_lifted_required_fluents:
        next_required_fluents |= {
            simp.simplify(subs.substitute(lifted_fluent, next_assignments_action))
        }

    # Aggiornare i modificatori per ogni effetto dell'azione
    for effect in current_inst_action.effects:
        for eff in effect.expand_effect(problem):
            current_grounded_fluent |= {
                simp.simplify(subs.substitute(eff.fluent, current_assignments_action))
            }
    for effect in next_action_inst_action.effects:
        for eff in effect.expand_effect(problem):
            next_grounded_fluent |= {
                simp.simplify(subs.substitute(eff.fluent, next_assignments_action))
            }

    # Controlliamo se le azioni corrente e successiva sono in sequenza
    if next_action and some_comparison_function(
        current_required_fluents,
        current_grounded_fluent,
        next_required_fluents,
        next_grounded_fluent,
        problem,
    ):
        if not in_sequence:
            # Se non siamo in una sequenza, iniziamo una nuova sequenza con l'azione corrente
            in_sequence = True
            current_sequence = [current_action]
        else:
            # Se siamo già in una sequenza, aggiungiamo l'azione corrente alla sequenza
            current_sequence.append(current_action)
    else:
        # Se la sequenza si interrompe o siamo all'ultima azione
        if in_sequence:
            # Se eravamo in una sequenza, aggiungiamo l'azione corrente alla sequenza
            current_sequence.append(current_action)
            if next_action:
                # Se esiste un'azione successiva, la includiamo anche nella sequenza
                current_sequence.append(next_action)
            # Salviamo la sequenza con l'azione precedente come chiave
            sequenza[previous_action] = current_sequence
            # Prepariamo per la prossima sequenza
            current_sequence = []
            in_sequence = False
        previous_action = current_action  # Impostiamo l'azione corrente come precedente per la prossima iterazione

# Stampa o restituisci il dizionario delle sequenze
print(sequenza)


"""current_seq = []
for i, v in sequenza.items():
  current_seq.append(v)
  print(i, v)"""

# *   Initialization: Sets up an empty dictionary to track sequences of actions, sorts the actions topologically from a multi-agent action plan, and establishes necessary functions for fluents manipulation and actions comparison.
# *   Actions Iteration: Sequentially iterates through the ordered actions, determining whether each action belongs to a sequence based on its relationship with the subsequent action.

# * Actions Iteration: Sequentially iterates through the ordered actions, determining whether each action belongs to a sequence based on its relationship with the subsequent action.
# * Actions Comparison: Uses a comparison function to decide if two consecutive actions should be grouped in the same sequence by examining the fluents required and modified by each action.
# * Sequences Building: Groups actions into sequences based on this dependency relationship and assigns them to a key in the dictionary, which is the action that precedes the start of the sequence.
# * Next Action Inclusion: Includes the action immediately following the end of each sequence in the value associated with its corresponding key in the dictionary.
# * End of Sequences Handling: When a sequence ends or the last action is reached, the code saves the current sequence in the dictionary before moving to the next one or finishing if the end of the action list has been reached.

##################################I find the dependencies of concurrent actions##################################
from unified_planning.plans.plan import ActionInstance
from unified_planning.model.action import InstantaneousAction

dipendenze_dirette = direct_dep
nuovo_piano = {}
azioni_gia_modificate = set()
azioni_end = [
    sequenza_azioni[-1] for sequenza_azioni in sequenza.values() if sequenza_azioni
]


def filtra_fluenti_ambientali(azione, environment):
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
    return azione not in sequenza.keys() and azione not in azioni_end


# Funzione per filtrare le dipendenze
def filtra_dipendenze(dipendenze):
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

print("Nuovo Piano:", nuovo_piano)

################################ SequentialValidator ################################
# Applicata Definizione 8 di:
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

        # Verifica conflitti ambientali
        if any(
            self._verifica_conflitto(
                fluente, valore, self.precondizioni_per_agente.get(agente, {}), agente
            )
            for fluente, valore in effetti_environment_set
            for agente, precondizioni in self.precondizioni_per_agente.items()
        ):
            return True

        # Verifica conflitti per ogni agente
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
                f"Conflitto rilevato: fluente {fluente.fluent().name} per l'agente {agente}"
            )
            return True
        return False

    def validate(self, actions, problem):
        found_contradition = False
        for i in range(len(actions) - 1):
            action = actions[i]
            next_action = actions[i + 1]

            # Aggiorna gli effetti per l'agente corrente
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

            # Prepara le precondizioni per l'agente dell'azione successiva
            self.precondizioni_per_agente = {
                next_action.agent.name: self._prepare_preconditions(
                    next_action, problem
                )
            }

            # Confronta gli effetti con le precondizioni dell'azione successiva
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

    def _print_results(self, action, next_action, contraddizione, index):

        if contraddizione:
            print("Azione corrente: ", action)
            print("Azione successiva: ", next_action)
            print("Effetti cumulativi per agente: ", self.eff_cumulativi_per_agente)
            print("Precondizioni per agente: ", self.precondizioni_per_agente)
            print(
                f"Contraddizione tra gli effetti dell'azione {index} (agente {action.agent.name}) e le precondizioni dell'azione {index + 1} (agente {next_action.agent.name})\n"
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
            "azione_corrente": cambiamenti_azione_corrente,
            "dipendenze": cambiamenti_dipendenze,
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

# Ora risultati è un dizionario con le azioni raggruppate per agente
# Esempio di utilizzo
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
        # Mappa i nomi base delle azioni ai loro rispettivi agenti
        mappa_azioni_agenti = {
            azione.action.name.split("_")[0]: azione.agent.name
            for azione in azioni_sequenza[:-1]
        }  # Escludi l'ultima azione

        for agent, azioni in acts.items():
            for azione in azioni:
                nome_azione = list(azione.keys())[0]
                nome_base = nome_azione.split("_")[0]

                # Verifica se l'azione è parte della sequenza e necessita di aggiornamento
                if nome_base in mappa_azioni_agenti and "_concurrent" in nome_azione:
                    azione_corrente = azione[nome_azione]["azione_corrente"]
                    dipendenze_attuali = set(
                        tuple(d[:2]) for d in azione[nome_azione]["dipendenze"]
                    )

                    # Verifica se nelle dipendenze sono presenti fluenti di tutti gli agenti della sequenza
                    for altro_nome_base, altro_agent in mappa_azioni_agenti.items():
                        fluente_manca = all(
                            (fl[0], altro_agent) not in dipendenze_attuali
                            for fl in azione_corrente
                        )
                        if fluente_manca:
                            print(
                                f"Manca un fluente di {altro_agent} nelle dipendenze di {nome_azione} di {agent}"
                            )
                            # Aggiungi i fluenti mancanti alle dipendenze
                            for azione_altra in acts[altro_agent]:
                                if (
                                    list(azione_altra.keys())[0]
                                    == altro_nome_base + "_concurrent"
                                ):
                                    for fluente in azione_altra[
                                        altro_nome_base + "_concurrent"
                                    ]["azione_corrente"]:
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
                                                    "dipendenze"
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
        action_fluents = action[action_key]["azione_corrente"]
        dependencies = action[action_key]["dipendenze"]

        # Se l'azione è concorrente e la prima dell'agente
        if "_concurrent" in action_key and i == 0:
            # Usa le dipendenze dell'azione come condizione per la transizione da state1 a state1X
            condition_for_state1X = tuple(
                (fluent, value)
                for fluent, agent_id, value in dependencies
                if agent_id == agent
            )
            # Aggiungi la transizione da state1 a state1X
            transitions[(current_state, condition_for_state1X)] = ("state1X", reward)
            current_state = "state1X"
            reward += 10

        if "_concurrent" not in action_key:
            # Azioni non concorrenti
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
            # Azioni concorrenti
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


# print(transitions)
# transitions
# RM_dict
print("\n\n\n", RM_dict)


def rimuovi_fluenti_falsi(RM_dict):
    RM_dict_pulito = {}
    for agente, transizioni in RM_dict.items():
        transizioni_pulite = {}
        for (stato_corrente, condizioni), (
            stato_successivo,
            ricompensa,
        ) in transizioni.items():
            # Filtra le condizioni per mantenere solo quelle vere
            condizioni_vere = tuple(
                condizione for condizione in condizioni if condizione[1]
            )
            if (
                condizioni_vere
            ):  # Se ci sono condizioni vere, aggiungile alle transizioni
                transizioni_pulite[(stato_corrente, condizioni_vere)] = (
                    stato_successivo,
                    ricompensa,
                )
        RM_dict_pulito[agente] = transizioni_pulite
    return RM_dict_pulito


# Applica la funzione a RM_dict
RM_dict_true = rimuovi_fluenti_falsi(RM_dict)


def rm_concurrent_sequence(rm_dictionary):
    new_dict = {}
    for agent, transitions in rm_dictionary.items():
        new_transitions = {}
        x_state_found = False
        transitions_list = list(transitions.items())

        for index, ((current_state, conditions), (next_state, reward)) in enumerate(
            transitions_list
        ):
            # Se lo stato corrente termina con 'X', aggiungi la transizione e segna che abbiamo trovato uno stato 'X'
            if "X" in current_state:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = True
            # Se abbiamo trovato uno stato 'X' e questa è la transizione immediatamente successiva, aggiungila
            elif x_state_found:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = False  # Resetta il flag dopo aver aggiunto la transizione successiva
            # Se lo stato successivo termina con 'X', aggiungi la transizione e segna che abbiamo trovato uno stato 'X'
            elif "X" in next_state:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = True

        # Aggiungi le nuove transizioni filtrate per l'agente corrente al nuovo dizionario
        new_dict[agent] = new_transitions
    return new_dict


RM_dict_true_seq = rm_concurrent_sequence(RM_dict_true)
# Supponendo che RM_dict_true_seq sia il dizionario che vuoi modificare

"""# Per ogni agente in RM_dict_true_seq
for agent, transitions in RM_dict_true_seq.items():
    # Prima, troviamo gli agenti con cui l'agente corrente ha interagito nello stato X
    agents_involved = set()
    for (from_state, conditions), (to_state, reward) in transitions.items():
        if 'X' in from_state:
            # Questo è lo stato concorrente
            for condition in conditions:
                if isinstance(condition[0], tuple) and condition[0][0] != agent:
                    # Questa è una condizione che coinvolge un altro agente
                    agents_involved.add(condition[0][0])

    # Ora, modifichiamo l'ultima transizione
    # Troviamo lo stato finale (quello che non appare mai come from_state)
    from_states = set()
    to_states = set()
    for (from_state, conditions), (to_state, reward) in transitions.items():
        from_states.add(from_state)
        to_states.add(to_state)
    final_states = to_states - from_states
    if len(final_states) != 1:
        print(f"Attenzione: L'agente {agent} ha stati finali multipli: {final_states}")
        continue
    final_state = final_states.pop()
    # Troviamo la transizione che porta allo stato finale
    transitions_to_final = [((from_state, conditions), (to_state, reward)) for ((from_state, conditions), (to_state, reward)) in transitions.items() if to_state == final_state]
    if len(transitions_to_final) != 1:
        print(f"Attenzione: L'agente {agent} ha transizioni multiple verso lo stato finale")
        continue
    ((from_state, conditions), (to_state, reward)) = transitions_to_final[0]
    # Ora, modifichiamo le condizioni per aggiungere le condizioni che gli altri agenti sono nello stesso stato finale
    # Troviamo la condizione di posizione dell'agente corrente
    agent_pos_conditions = [cond for cond in conditions if not isinstance(cond[0], tuple) and 'pos(' in str(cond[0])]
    if not agent_pos_conditions:
        print(f"L'agente {agent} non ha una propria posizione nelle condizioni")
        continue
    agent_pos_condition = agent_pos_conditions[0]
    # Otteniamo la posizione (es. 'pos(l14)')
    agent_position = agent_pos_condition[0]
    # Ora, per ogni altro agente, aggiungiamo una condizione che l'altro agente sia nella stessa posizione
    new_conditions = list(conditions)
    for other_agent in agents_involved:
        other_agent_pos_condition = ((other_agent, agent_position), True)
        if other_agent_pos_condition not in new_conditions:
            new_conditions.append(other_agent_pos_condition)
    # Aggiorniamo la transizione
    # Rimuoviamo la vecchia transizione
    transitions.pop((from_state, conditions))
    # Aggiungiamo la nuova transizione
    transitions[(from_state, tuple(new_conditions))] = (to_state, reward)"""

# Ora, RM_dict_true_seq è stato aggiornato con le nuove condizioni
# breakpoint()
