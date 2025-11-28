"""A utility for building transition rules for multi-agent scheduling."""

import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import networkx as nx
import unified_planning as up
import unified_planning.model.walkers as walkers
import unified_planning.plans as plans
from unified_planning.exceptions import UPUsageError, UPValueError
from unified_planning.model import Expression, FNode, InstantaneousAction
from unified_planning.model.operators import OperatorKind
from unified_planning.plot import show_partial_order_plan


def directly_depends_on(
    partial_order_plan: plans.PartialOrderPlan, problem: Any
) -> tuple[
    Dict[plans.ActionInstance, List[plans.ActionInstance]], List[Dict[str, Any]]
]:
    """
    Establish direct dependencies between actions in a partial order plan.

    :param partial_order_plan: The partial order plan containing actions and dependencies.
    :param problem: The planning problem containing environment information.
    :return: A tuple with a mapping from action to its direct dependencies and a list of
        dependency metadata.
    """
    subs = partial_order_plan._environment.substituter
    simp = partial_order_plan._environment.simplifier
    eqr = walkers.ExpressionQuantifiersRemover(partial_order_plan._environment)
    fve = partial_order_plan._environment.free_vars_extractor

    graph = partial_order_plan._graph
    last_modifier: Dict[tuple[FNode, str], plans.ActionInstance] = {}
    last_modifier_env: Dict[FNode, plans.ActionInstance] = {}
    direct_dependencies: Dict[plans.ActionInstance, List[plans.ActionInstance]] = {}
    dependency_data: List[Dict[str, Any]] = []

    for action_instance in graph.nodes():
        inst_action = cast(InstantaneousAction, action_instance.action)
        required_fluents: Set[FNode] = set()
        lifted_required_fluents: Set[FNode] = set()

        for prec in inst_action.preconditions:
            lifted_required_fluents |= fve.get(eqr.remove_quantifiers(prec, problem))

        assignments_action = dict(
            zip(inst_action.parameters, action_instance.actual_parameters)
        )
        for lifted_fluent in lifted_required_fluents:
            required_fluents |= {
                simp.simplify(subs.substitute(lifted_fluent, assignments_action))
            }

        for required_fluent in required_fluents:
            try:
                if problem.ma_environment.fluent(required_fluent.fluent().name):
                    required_fluent_last_modifier = last_modifier_env.get(
                        required_fluent, None
                    )
            except UPValueError:
                required_fluent_last_modifier = last_modifier.get(
                    (required_fluent, action_instance.agent.name), None
                )

            if required_fluent_last_modifier is not None:
                direct_dependencies.setdefault(action_instance, []).append(
                    required_fluent_last_modifier
                )

                dependency_info = {
                    "current_action": action_instance,
                    "dependent_agent": required_fluent_last_modifier.agent.name
                    if hasattr(required_fluent_last_modifier, "agent")
                    else None,
                    "dependent_action": required_fluent_last_modifier,
                    "matching_condition": required_fluent,
                }
                dependency_data.append(dependency_info)

        for effect in inst_action.effects:
            for eff in effect.expand_effect(problem):
                grounded_fluent = simp.simplify(
                    subs.substitute(eff.fluent, assignments_action)
                )
                try:
                    if problem.ma_environment.fluent(grounded_fluent.fluent().name):
                        last_modifier_env[grounded_fluent] = action_instance
                except UPValueError:
                    last_modifier[
                        (grounded_fluent, action_instance.agent.name)
                    ] = action_instance

    return direct_dependencies, dependency_data


def compute_required_fluents(
    inst_action: InstantaneousAction,
    action_instance: up.plans.plan.ActionInstance,
    problem: Any,
    subs: Any,
    simp: Any,
    eqr: walkers.ExpressionQuantifiersRemover,
    fve: walkers.FreeVarsExtractor,
) -> tuple[set[FNode], dict]:
    """Return fluents and parameter assignments required for an action.

    :param inst_action: The instantaneous action being considered.
    :param action_instance: The grounded action instance.
    :param problem: The planning problem.
    :param subs: The substituter from the planning environment.
    :param simp: The simplifier from the planning environment.
    :param eqr: Expression quantifiers remover.
    :param fve: Free variables extractor.
    :return: A tuple containing required fluents and the assignments applied.
    """
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
    """Calculate grounded fluents resulting from action effects.

    :param inst_action: The instantaneous action being processed.
    :param assignments_action: Parameter assignments for the action.
    :param problem: The planning problem.
    :param simp: The simplifier from the planning environment.
    :param subs: The substituter from the planning environment.
    :return: A set of grounded fluents produced by the action.
    """
    grounded_fluents: Set[FNode] = set()
    for effect in inst_action.effects:
        for eff in effect.expand_effect(problem):
            grounded_fluents.add(
                simp.simplify(subs.substitute(eff.fluent, assignments_action))
            )
    return grounded_fluents


def remove_fluents_if_owned_by_environment(
    fluents: Set[FNode], environment: Any
) -> Set[FNode]:
    """Remove fluents owned by the environment from the provided set.

    :param fluents: A set of fluents to clean.
    :param environment: The environment to check for fluent ownership.
    :return: A set with environment-owned fluents removed.
    """
    to_remove: Set[FNode] = set()
    for fluent in fluents:
        try:
            if environment.fluent(fluent.fluent().name):
                to_remove.add(fluent)
        except UPValueError:
            pass
    return fluents - to_remove


def some_comparison_function(
    current_required_fluents,
    current_grounded_fluent,
    next_required_fluents,
    next_grounded_fluent,
    problem,
) -> bool:
    """Compare fluent requirements/effects between two actions.

    :param current_required_fluents: Set of required fluents for current action.
    :param current_grounded_fluent: Set of grounded fluents for current action.
    :param next_required_fluents: Set of required fluents for next action.
    :param next_grounded_fluent: Set of grounded fluents for next action.
    :param problem: Planning problem with environment definition.
    :return: ``True`` when cleaned fluent sets match, else ``False``.
    """
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

    return (
        current_required_clean == next_required_clean
        and current_grounded_clean == next_grounded_clean
    )


def filtra_fluenti_ambientali(azione: InstantaneousAction, environment: Any) -> None:
    """Remove environment-owned fluents from an action's preconditions and effects.

    :param azione: The action to sanitize.
    :param environment: The environment containing shared fluents.
    """
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


def deve_essere_aggiunta(
    azione: plans.ActionInstance, sequenza: Dict[Any, List[Any]], azioni_end: List[Any]
) -> bool:
    """Determine if an action should be included in the new plan.

    :param azione: The action instance under consideration.
    :param sequenza: Mapping of sequence starts to their actions.
    :param azioni_end: List of actions that end sequences.
    :return: ``True`` if the action is not already covered by a sequence, else ``False``.
    """
    return azione not in sequenza.keys() and azione not in azioni_end


def filtra_dipendenze(
    dipendenze: List[plans.ActionInstance],
    sequenza: Dict[Any, List[Any]],
    azioni_end: List[Any],
) -> List[plans.ActionInstance]:
    """Filter out dependencies already represented by sequences or endings.

    :param dipendenze: Dependencies to filter.
    :param sequenza: Mapping of sequence starts to their actions.
    :param azioni_end: Actions that end sequences.
    :return: Filtered dependency list.
    """
    return [
        dip
        for dip in dipendenze
        if dip not in sequenza.keys() and dip not in azioni_end and dip
    ]


class SequentialPlanValidator:
    """Validate sequential consistency between actions in a plan."""

    def __init__(self, pop_plan: plans.Plan):
        """Initialize validator utilities from the planning environment."""
        self.pop_plan = pop_plan
        self.subs = pop_plan.plan._environment.substituter
        self.simp = pop_plan.plan._environment.simplifier
        self.eqr = walkers.ExpressionQuantifiersRemover(pop_plan.plan._environment)
        self.fve = pop_plan.plan._environment.free_vars_extractor
        self.eff_cumulativi_per_agente: Dict[str, Dict[FNode, Any]] = {}
        self.eff_cumulativi_environment: Dict[str, Dict[FNode, Any]] = {}

    def verifica_conflitti(self) -> bool:
        """Check for conflicts between effects and upcoming preconditions."""
        effetti_environment = self.eff_cumulativi_environment.get("env", {})
        effetti_environment_set = set(effetti_environment.items())

        if any(
            self._verifica_conflitto(
                fluente, valore, self.precondizioni_per_agente.get(agente, {}), agente
            )
            for fluente, valore in effetti_environment_set
            for agente, precondizioni in self.precondizioni_per_agente.items()
        ):
            return True

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

    def _verifica_conflitto(
        self, fluente: FNode, valore: Any, precondizioni: Dict[FNode, Any], agente: str
    ) -> bool:
        """Check if a fluent value contradicts stored preconditions."""
        valore_precondizione = precondizioni.get(fluente)
        if valore_precondizione is not None and valore_precondizione != valore:
            print(
                f"Conflict detected: fluent {fluente.fluent().name} for agent {agente}"
            )
            return True
        return False

    def validate(self, actions: List[plans.ActionInstance], problem: Any) -> None:
        """Validate that cumulative effects align with the next action's preconditions."""
        found_contradition = False
        for i in range(len(actions) - 1):
            action = actions[i]
            next_action = actions[i + 1]

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

            self.precondizioni_per_agente = {
                next_action.agent.name: self._prepare_preconditions(
                    next_action, problem
                )
            }

            contraddizione = self.verifica_conflitti()
            if contraddizione:
                found_contradition = True
            self._print_results(action, next_action, contraddizione, i)
        if not found_contradition:
            print("Valid Plan!")

    def _prepare_preconditions(
        self, next_action: plans.ActionInstance, problem: Any
    ) -> Dict[FNode, bool]:
        """Ground preconditions for a specific action instance."""
        prec_agente: Dict[FNode, bool] = {}
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

    def _print_results(
        self,
        action: plans.ActionInstance,
        next_action: plans.ActionInstance,
        contradiction: bool,
        index: int,
    ) -> None:
        """Display validation step outcomes."""
        if contradiction:
            print("Current action: ", action)
            print("Next action: ", next_action)
            print("Cumulative effects per agent: ", self.eff_cumulativi_per_agente)
            print("Preconditions per agent: ", self.precondizioni_per_agente)
            print(
                f"Contradiction between the effects of action {index} (agent {action.agent.name}) "
                f"and the preconditions of action {index + 1} (agent {next_action.agent.name})\n"
            )


def estrai_cambiamenti_fluenti_per_rm(
    azione_con_dipendenze: Dict[plans.ActionInstance, List[plans.ActionInstance]],
    problem: Any,
    simp: Any,
    subs: Any,
) -> Dict[str, Dict[str, List[Tuple[FNode, str, Any]]]]:
    """Extract fluent changes for an action and its dependencies for RM building.

    :param azione_con_dipendenze: Mapping of action to its dependencies.
    :param problem: The planning problem.
    :param simp: Simplifier from the planning environment.
    :param subs: Substituter from the planning environment.
    :return: A dictionary with current action changes and dependency changes.
    """
    cambiamenti_per_rm: Dict[str, Dict[str, List[Tuple[FNode, str, Any]]]] = {}
    for azione, dipendenze in azione_con_dipendenze.items():
        inst_action = cast(InstantaneousAction, azione.action)

        cambiamenti_azione_corrente: List[Tuple[FNode, str, Any]] = []
        cambiamenti_dipendenze: List[Tuple[FNode, str, Any]] = []

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

        cambiamenti_per_rm[inst_action.name] = {
            "current_action": cambiamenti_azione_corrente,
            "dependencies": cambiamenti_dipendenze,
        }

    return cambiamenti_per_rm


def aggiorna_dipendenze_concurrent(
    acts: Dict[str, List[Dict[str, Dict[str, List[Tuple[FNode, str, Any]]]]]],
    sequenza: Dict[Any, List[Any]],
) -> Dict[str, List[Dict[str, Dict[str, List[Tuple[FNode, str, Any]]]]]]:
    """Ensure concurrent actions include all dependency fluents across agents.

    :param acts: Actions grouped by agent with fluent changes and dependencies.
    :param sequenza: Mapping of sequence starts to the actions in the sequence.
    :return: Updated actions with completed dependencies.
    """
    for nome_sequenza, azioni_sequenza in sequenza.items():
        mappa_azioni_agenti = {
            azione.action.name.split("_")[0]: azione.agent.name
            for azione in azioni_sequenza[:-1]
        }

        for agent, azioni in acts.items():
            for azione in azioni:
                nome_azione = list(azione.keys())[0]
                nome_base = nome_azione.split("_")[0]

                if nome_base in mappa_azioni_agenti and "_concurrent" in nome_azione:
                    azione_corrente = azione[nome_azione]["current_action"]
                    dipendenze_attuali = set(
                        tuple(d[:2]) for d in azione[nome_azione]["dependencies"]
                    )

                    for altro_nome_base, altro_agent in mappa_azioni_agenti.items():
                        fluente_manca = all(
                            (fl[0], altro_agent) not in dipendenze_attuali
                            for fl in azione_corrente
                        )
                        if fluente_manca:
                            print(
                                f"A fluent is missing {altro_agent} nelle dependencies of {nome_azione} of {agent}"
                            )
                            for azione_altra in acts[altro_agent]:
                                if (
                                    list(azione_altra.keys())[0]
                                    == altro_nome_base + "_concurrent"
                                ):
                                    for fluente in azione_altra[
                                        altro_nome_base + "_concurrent"
                                    ]["current_action"]:
                                        if fluente[2] is False:
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


def rimuovi_fluenti_falsi(
    RM_dict: Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]]
) -> Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]]:
    """Discard transitions where all conditions evaluate to ``False``.

    :param RM_dict: Transition dictionary per agent.
    :return: Filtered transition dictionary containing only true conditions.
    """
    RM_dict_pulito: Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]] = {}
    for agente, transizioni in RM_dict.items():
        transizioni_pulite: Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]] = {}
        for (stato_corrente, condizioni), (
            stato_successivo,
            ricompensa,
        ) in transizioni.items():
            condizioni_vere = tuple(
                condizione for condizione in condizioni if condizione[1]
            )
            if condizioni_vere:
                transizioni_pulite[(stato_corrente, condizioni_vere)] = (
                    stato_successivo,
                    ricompensa,
                )
        RM_dict_pulito[agente] = transizioni_pulite
    return RM_dict_pulito


def rm_concurrent_sequence(
    rm_dictionary: Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]]
) -> Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]]:
    """Store transitions for concurrent states and their immediate successors.

    :param rm_dictionary: Transition dictionary per agent.
    :return: Filtered transition dictionary capturing concurrent sequences.
    """
    new_dict: Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]] = {}
    for agent, transitions in rm_dictionary.items():
        new_transitions: Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]] = {}
        x_state_found = False
        transitions_list = list(transitions.items())

        for index, ((current_state, conditions), (next_state, reward)) in enumerate(
            transitions_list
        ):
            if "X" in current_state:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = True
            elif x_state_found:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = False
            elif "X" in next_state:
                new_transitions[(current_state, conditions)] = (next_state, reward)
                x_state_found = True

        new_dict[agent] = new_transitions
    return new_dict


def load_plan_and_problem(
    plan_path: str,
    problem_path: str,
) -> tuple[Any, Any]:
    """Load the default plan and problem definitions."""

    with open(plan_path, "rb") as file:
        pop_plan = pickle.load(file)
    with open(problem_path, "rb") as file:
        problem = pickle.load(file)

    return pop_plan, problem


def build_sequences(
    ordered_nodes_list: List[plans.ActionInstance],
    problem: Any,
    subs: Any,
    simp: Any,
    eqr: walkers.ExpressionQuantifiersRemover,
    fve: walkers.FreeVarsExtractor,
) -> Dict[Optional[plans.ActionInstance], List[plans.ActionInstance]]:
    """Identify action sequences based on fluent comparisons."""

    sequenza: Dict[Optional[plans.ActionInstance], List[plans.ActionInstance]] = {}
    in_sequence = False
    current_sequence: List[plans.ActionInstance] = []
    previous_action: Optional[plans.ActionInstance] = None

    for i in range(len(ordered_nodes_list) - 1):
        current_action = ordered_nodes_list[i]
        next_action = (
            ordered_nodes_list[i + 1] if i + 1 < len(ordered_nodes_list) else None
        )

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

        if next_action and some_comparison_function(
            current_required_fluents,
            current_grounded_fluent,
            next_required_fluents,
            next_grounded_fluent,
            problem,
        ):
            if not in_sequence:
                in_sequence = True
                current_sequence = [current_action]
            else:
                current_sequence.append(current_action)
        else:
            if in_sequence:
                current_sequence.append(current_action)
                if next_action:
                    current_sequence.append(next_action)
                sequenza[previous_action] = current_sequence
                current_sequence = []
                in_sequence = False
            previous_action = current_action

    return sequenza


def aggregate_dependencies(
    dipendenze_dirette: Dict[plans.ActionInstance, List[plans.ActionInstance]],
    sequenza: Dict[Optional[plans.ActionInstance], List[plans.ActionInstance]],
    azioni_end: List[plans.ActionInstance],
) -> Dict[Any, List[plans.ActionInstance]]:
    """Collect unique dependencies per sequence."""

    dipendenze_aggregate: Dict[Any, List[plans.ActionInstance]] = {}
    for chiave, azioni in sequenza.items():
        if chiave not in dipendenze_aggregate:
            dipendenze_aggregate[chiave] = []

        for azione in azioni:
            if azione in dipendenze_dirette:
                dipendenze_filtrate = filtra_dipendenze(
                    dipendenze_dirette[azione], sequenza, azioni_end
                )
                for dipendenza in dipendenze_filtrate:
                    if (
                        dipendenza not in dipendenze_aggregate[chiave]
                        and dipendenza not in sequenza[chiave]
                    ):
                        dipendenze_aggregate[chiave].append(dipendenza)

    return dipendenze_aggregate


def crea_nuovo_piano(
    dipendenze_dirette: Dict[plans.ActionInstance, List[plans.ActionInstance]],
    sequenza: Dict[Optional[plans.ActionInstance], List[plans.ActionInstance]],
    azioni_end: List[plans.ActionInstance],
    dipendenze_aggregate: Dict[Any, List[plans.ActionInstance]],
    problem: Any,
) -> Dict[plans.ActionInstance, List[plans.ActionInstance]]:
    """Assemble the updated plan inserting concurrent actions where needed."""

    nuovo_piano: Dict[plans.ActionInstance, List[plans.ActionInstance]] = {}
    azioni_gia_modificate: Set[str] = set()

    for azione, dipendenze in dipendenze_dirette.items():
        if deve_essere_aggiunta(azione, sequenza, azioni_end) and any(
            azione in sequenza_azioni for sequenza_azioni in sequenza.values()
        ):
            nome_concorrente = f"{azione.action.name}_concurrent"
            if nome_concorrente not in azioni_gia_modificate:
                nuova_azione = azione.action.clone()
                nuova_azione.name = nome_concorrente
                azioni_gia_modificate.add(nome_concorrente)

                filtra_fluenti_ambientali(nuova_azione, problem.ma_environment)
                azioni_gia_modificate.add(nome_concorrente)

            chiave_di_sequenza = next(
                (k for k, v in sequenza.items() if azione in v), None
            )
            dipendenze_da_usare = filtra_dipendenze(
                dipendenze_aggregate.get(chiave_di_sequenza, []), sequenza, azioni_end
            )

            nuova_azione_instance = up.plans.plan.ActionInstance(
                nuova_azione, azione.actual_parameters, agent=azione.agent
            )
            nuovo_piano[nuova_azione_instance] = dipendenze_da_usare
        elif deve_essere_aggiunta(azione, sequenza, azioni_end):
            nuovo_piano[azione] = filtra_dipendenze(dipendenze, sequenza, azioni_end)

    return nuovo_piano


def build_reward_machine(pop_plan: Any, problem: Any) -> Dict[str, Any]:
    """Execute the RM building workflow for a given plan and problem."""

    direct_dep, data_dep = directly_depends_on(pop_plan.plan, problem)
    ordered_nodes = nx.topological_sort(pop_plan.plan._graph)
    ordered_nodes_list = list(ordered_nodes)

    subs = pop_plan.plan._environment.substituter
    simp = pop_plan.plan._environment.simplifier
    eqr = walkers.ExpressionQuantifiersRemover(pop_plan.plan._environment)
    fve = pop_plan.plan._environment.free_vars_extractor

    sequenza = build_sequences(ordered_nodes_list, problem, subs, simp, eqr, fve)
    print(sequenza)

    dipendenze_dirette = direct_dep
    azioni_end = [
        sequenza_azioni[-1] for sequenza_azioni in sequenza.values() if sequenza_azioni
    ]
    dipendenze_aggregate = aggregate_dependencies(
        dipendenze_dirette, sequenza, azioni_end
    )
    nuovo_piano = crea_nuovo_piano(
        dipendenze_dirette, sequenza, azioni_end, dipendenze_aggregate, problem
    )

    print("New plan:", nuovo_piano)

    liste_di_sequenze = [[chiave] + valori for chiave, valori in sequenza.items()]
    first_sequenza = liste_di_sequenze[0]

    validator = SequentialPlanValidator(pop_plan)
    validator.validate(first_sequenza, problem)

    risultati: Dict[
        str, List[Dict[plans.ActionInstance, List[plans.ActionInstance]]]
    ] = {}
    for k, v in nuovo_piano.items():
        if k.agent.name not in risultati:
            risultati[k.agent.name] = []
        risultati[k.agent.name].append({k: v})

    cambiamenti_per_rm = estrai_cambiamenti_fluenti_per_rm(
        risultati["a2"][4], problem, simp, subs
    )
    print(cambiamenti_per_rm)

    acts_: List[Dict[str, Dict[str, List[Tuple[FNode, str, Any]]]]] = []
    dic: Dict[str, List[Dict[str, Dict[str, List[Tuple[FNode, str, Any]]]]]] = {}
    for j, k in risultati.items():
        acts_ = []
        for i in k:
            acts_.append(estrai_cambiamenti_fluenti_per_rm(i, problem, simp, subs))
        dic[j] = acts_

    nuovo_acts_aggiornato = aggiorna_dipendenze_concurrent(dic, sequenza)
    print(nuovo_acts_aggiornato)

    nuvo_dic = nuovo_acts_aggiornato
    transitions: Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]] = {}
    current_state = "state1"
    reward = 10
    state_counter = 2
    RM_dict: Dict[str, Dict[Tuple[str, Tuple[Any, ...]], Tuple[str, int]]] = {}
    for agent, actions in nuvo_dic.items():
        current_state = "state1"
        reward = 10
        state_counter = 2
        transitions = {}

        for i, action in enumerate(actions):
            action_key = list(action.keys())[0]
            action_fluents = action[action_key]["current_action"]
            dependencies = action[action_key]["dependencies"]

            if "_concurrent" in action_key and i == 0:
                condition_for_state1X = tuple(
                    (fluent, value)
                    for fluent, agent_id, value in dependencies
                    if agent_id == agent
                )
                transitions[(current_state, condition_for_state1X)] = (
                    "state1X",
                    reward,
                )
                current_state = "state1X"
                reward += 10

            if "_concurrent" not in action_key:
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
                dependency_condition = tuple(
                    ((agent_id, fluent), value)
                    if agent_id != agent
                    else (fluent, value)
                    for fluent, agent_id, value in dependencies
                )
                final_fluents = tuple(
                    ((agent_id, fluent), value)
                    if agent_id != agent
                    else (fluent, value)
                    for fluent, agent_id, value in action_fluents
                )
                next_state = f"state{state_counter}"

                transitions[(current_state, dependency_condition)] = (
                    next_state,
                    reward,
                )
                transitions[(next_state, final_fluents)] = (
                    f"state{state_counter + 1}",
                    reward + 10,
                )
                current_state = f"state{state_counter + 1}"

            state_counter += 1
            reward += 10

        RM_dict[agent] = transitions

    print("\n\n\n", RM_dict)

    RM_dict_true = rimuovi_fluenti_falsi(RM_dict)
    RM_dict_true_seq = rm_concurrent_sequence(RM_dict_true)

    return {
        "direct_dependencies": dipendenze_dirette,
        "dependency_metadata": data_dep,
        "sequenza": sequenza,
        "nuovo_piano": nuovo_piano,
        "risultati": risultati,
        "nuovo_acts_aggiornato": nuovo_acts_aggiornato,
        "RM_dict": RM_dict,
        "RM_dict_true": RM_dict_true,
        "RM_dict_true_seq": RM_dict_true_seq,
    }


def main() -> None:
    """Entrypoint that loads sample data and builds the reward machine."""
    pop_plan, problem = load_plan_and_problem(
        "planning_utils/result_maze_5_agents.pkl",
        "planning_utils/problem_maze_5_agents.pkl",
    )
    build_reward_machine(pop_plan, problem)


# The functions above can be imported without triggering any side effects.
# The following guard only executes the default pipeline when the module is
# run directly (e.g., `python building_RM.py`). When the module is imported
# elsewhere, `main` is not executed, allowing callers to invoke
# `load_plan_and_problem` or `build_reward_machine` explicitly.
if __name__ == "__main__":
    main()
