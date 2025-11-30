from enum import Enum, auto
from fractions import Fraction
from warnings import warn
import unified_planning as up

from unified_planning.engines.engine import Engine
from unified_planning.engines.mixins.sequential_simulator import (
    SequentialSimulatorMixin,
)
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.model.problem_kind_versioning import LATEST_PROBLEM_KIND_VERSION
from unified_planning.exceptions import (
    UPUsageError,
    UPConflictingEffectsException,
    UPInvalidActionError,
    UPUnreachableCodeError,
    UPProblemDefinitionError,
)
from unified_planning.model import (
    Fluent,
    FNode,
    ExpressionManager,
    Problem,
    MinimizeActionCosts,
    MinimizeExpressionOnFinalState,
    MaximizeExpressionOnFinalState,
    Oversubscription,
)
from unified_planning.model.walkers import ExpressionQuantifiersRemover
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from unified_planning.model.multi_agent import MultiAgentProblem, Agent

# --- Custom tools from your project --- #
from multiagentplanning_rl.utils.grounder import Grounder, GrounderHelper
from multiagentplanning_rl.utils.state_evaluator import StateEvaluator
from multiagentplanning_rl.utils.up_state import UPState


class InapplicabilityReasons(Enum):
    """
    Represents the possible reasons for an action being inapplicable after the
    ``SequentialSimulator.is_applicable`` method returns ``True`` but then the
    ``SequentialSimulator.apply_unsafe`` returns ``None``.
    """

    VIOLATES_CONDITIONS = auto()
    CONFLICTING_EFFECTS = auto()
    VIOLATES_STATE_INVARIANTS = auto()


class UPSequentialSimulatorMA(Engine, SequentialSimulatorMixin):
    """
    Sequential SequentialSimulatorMixin implementation for MultiAgent problems.

    This SequentialSimulator, when considering if a state is goal or not, ignores the
    quality metrics.
    """

    def __init__(
        self,
        problem: "MultiAgentProblem",
        error_on_failed_checks: bool = True,
        **kwargs,
    ):
        Engine.__init__(self)
        SequentialSimulatorMixin.__init__(self, problem, error_on_failed_checks)

        pk = problem.kind
        if not Grounder.supports(pk):
            msg = (
                f"The Grounder used in the {type(self).__name__} "
                f"does not support the given problem"
            )
            if self.error_on_failed_checks:
                raise UPUsageError(msg)
            else:
                warn(msg)

        assert isinstance(self._problem, MultiAgentProblem)
        self._grounder = GrounderHelper(problem)
        self._se = StateEvaluator(self._problem)
        self._initial_state: Optional[UPState] = None

        # NOTE: state invariants check is currently disabled for performance reasons.
        # If you want to re-enable them, you can reconstruct the original logic
        # using ExpressionQuantifiersRemover and bounded numeric types.
        qrm = ExpressionQuantifiersRemover(self._problem.environment)
        # self._state_invariants: List[FNode] = [
        #     qrm.remove_quantifiers(si, self._problem).simplify()
        #     for si in self._problem.state_invariants
        # ]
        # self._fluent_exps_in_state_invariants: Set[FNode] = set()
        # for si in self._state_invariants:
        #     self._fluent_exps_in_state_invariants |= (
        #         si.environment.free_vars_extractor.get(si)
        #     )
        #
        # em = self._problem.environment.expression_manager
        # for f in self.agent.fluents:
        #     lower_bound, upper_bound = None, None
        #     f_type = f.type
        #     if f_type.is_int_type() or f_type.is_real_type():

    #         f_type = cast(_RealType, f_type)
    #         lower_bound, upper_bound = f_type.lower_bound, f_type.upper_bound
    #     if lower_bound is not None:
    #         for f_e in get_all_fluent_exp(self._problem, f):
    #             self._fluent_exps_in_state_invariants.add(f_e)
    #             self._state_invariants.append(em.LE(lower_bound, f_e))
    #     if upper_bound is not None:
    #         for f_e in get_all_fluent_exp(self._problem, f):
    #             self._fluent_exps_in_state_invariants.add(f_e)
    #             self._state_invariants.append(em.LE(f_e, upper_bound))
    #
    # self._fluents_in_state_invariants: Set[Fluent] = set(
    #     (fe.fluent() for fe in self._fluent_exps_in_state_invariants)
    # )

    def _ground_action(
        self,
        agent: "Agent",
        action: "up.model.Action",
        params: Tuple["up.model.FNode", ...],
    ) -> Optional["up.model.InstantaneousAction"]:
        """
        Utility method to ground an action and do the basic checks.

        :param action: The action to ground.
        :param params: The parameters used to ground the action.
        :return: The grounded action. None if the action grounds to an
            invalid action.
        """
        grounded_act = self._grounder.ground_action(action, params)
        assert (
            isinstance(grounded_act, up.model.InstantaneousAction)
            or grounded_act is None
        ), "Supported_kind not respected"
        return grounded_act

    def _get_initial_state(self) -> "up.model.State":
        """
        Returns the problem's initial state.

        NOTE: Every method that requires a state assumes that it's the same class
        of the state given here, therefore an up.model.UPState.
        """
        assert isinstance(
            self._problem, MultiAgentProblem
        ), "supported_kind not respected"

        if self._initial_state is None:
            self._initial_state = UPState(self._problem.initial_values)

            # If you re-enable state invariants above, also re-enable this:
            # for si in self._state_invariants:
            #     if not self._se.evaluate(si, self._initial_state).bool_constant_value():
            #         raise UPProblemDefinitionError(
            #             "The initial state of the problem already violates the state invariants"
            #         )
        assert self._initial_state is not None
        return self._initial_state

    def _is_applicable(
        self,
        agent: "Agent",
        state: "up.model.State",
        action: "up.model.Action",
        parameters: Tuple["up.model.FNode", ...],
    ) -> bool:
        """
        Returns `True` if the given `action conditions` are evaluated as `True` in the given `state`;
        returns `False` otherwise.
        """
        try:
            _, reason = self.get_unsatisfied_conditions(
                agent,
                state,
                action,
                parameters,
                early_termination=True,
                full_check=True,
            )
            is_applicable = reason is None
        except UPInvalidActionError:
            is_applicable = False
        return is_applicable

    def _apply(
        self,
        env,
        agent: "Agent",
        state: "up.model.State",
        action: "up.model.Action",
        parameters: Tuple["up.model.FNode", ...],
    ) -> Optional["up.model.State"]:
        """
        Returns `None` if the given `action` is not applicable in the given `state`,
        otherwise returns a new `State` with the effects applied.
        """
        _, reason = self.get_unsatisfied_conditions(
            agent, state, action, parameters, early_termination=True, full_check=False
        )
        if reason is not None:
            return None

        try:
            return self.apply_unsafe(env, agent, state, action, parameters)
        except (UPInvalidActionError, UPConflictingEffectsException):
            return None

    def apply_unsafe(
        self,
        env,
        agent: "Agent",
        state: "up.model.State",
        action_or_action_instance: Union["up.model.Action", "up.plans.ActionInstance"],
        parameters: Optional[Sequence["up.model.Expression"]] = None,
    ) -> "up.model.State":
        """
        Returns a new `State` with the applicable `effects` of the `action` applied.
        IMPORTANT NOTE: Assumes that `self.is_applicable(state, event)` returns `True`.
        """
        action, params = self._get_action_and_parameters(
            action_or_action_instance, parameters
        )

        grounded_action = self._ground_action(agent, action, params)
        if grounded_action is None:
            raise UPInvalidActionError("Apply_unsafe got an inapplicable action.")
        assert isinstance(action, up.model.InstantaneousAction)

        updated_values: Dict["up.model.FNode", "up.model.FNode"] = {}
        assigned_fluent: Set["up.model.FNode"] = set()
        em = self._problem.environment.expression_manager

        if grounded_action.simulated_effect is not None:
            for f, v in zip(
                grounded_action.simulated_effect.fluents,
                grounded_action.simulated_effect.function(self._problem, state, {}),
            ):
                updated_values[f] = v
                assigned_fluent.add(f)

        for e in grounded_action.effects:
            for effect in e.expand_effect(
                cast(up.model.mixins.ObjectsSetMixin, self._problem)
            ):
                fluent, value = self._evaluate_effect(
                    env, agent, effect, state, updated_values, assigned_fluent, em
                )
                if fluent is not None:
                    assert value is not None
                    agent_fluent = fluent
                    updated_values[agent_fluent] = value

        new_state = state.make_child(updated_values)

        # If you re-enable state invariants, re-enable this:
        # for si in self._state_invariants:
        #     if not self._se.evaluate(si, new_state).bool_constant_value():
        #         raise UPInvalidActionError(
        #             "The given action is not applicable because it violates state invariants.",
        #             "Bounded numeric types are checked as state invariants.",
        #         )

        return new_state

    def _evaluate_effect(
        self,
        env,
        agent: "Agent",
        effect: "up.model.Effect",
        state: "up.model.State",
        updated_values: Dict["up.model.FNode", "up.model.FNode"],
        assigned_fluent: Set["up.model.FNode"],
        em: ExpressionManager,
        evaluated_fluent: Optional[FNode] = None,
        evaluated_condition: Optional[bool] = None,
    ) -> Tuple[Optional[FNode], Optional[FNode]]:
        def evaluate(exp: FNode) -> FNode:
            return self._se.evaluate(agent, exp, state)

        # Compute the fluent instance
        if evaluated_fluent is not None:
            fluent = evaluated_fluent
        else:
            fluent = effect.fluent.fluent()(*(map(evaluate, effect.fluent.args)))

        # Decide whether this is an environment fluent or an agent fluent
        if fluent.fluent() in env.fluents:
            agent_fluent = fluent  # Caso in cui il fluente è dell'environment
        else:
            agent_fluent = em.Dot(agent, fluent)

        if evaluated_condition is None:
            evaluated_condition = (
                not effect.is_conditional() or evaluate(effect.condition).is_true()
            )

        if not evaluated_condition:
            return None, None

        new_value = evaluate(effect.value)

        if effect.is_assignment():
            old_value = updated_values.get(agent_fluent, None)

            if (
                old_value is not None
                and new_value.constant_value() != old_value.constant_value()
            ):
                if not fluent.type.is_bool_type():
                    raise UPConflictingEffectsException(
                        f"The fluent {fluent} is modified by 2 different assignments in the same action."
                    )
                elif not old_value.bool_constant_value():
                    return agent_fluent, new_value
                else:
                    return None, None
            elif old_value is not None and agent_fluent not in assigned_fluent:
                raise UPConflictingEffectsException(
                    f"The fluent {fluent} is modified by 1 assignments and an increase/decrease in the same action."
                )
            else:
                assigned_fluent.add(agent_fluent)
                return agent_fluent, new_value
        else:
            if fluent in assigned_fluent:
                raise UPConflictingEffectsException(
                    f"The fluent {fluent} is modified by an assignment and an increase/decrease in the same action."
                )

            f_eval = updated_values.get(agent_fluent, evaluate(agent_fluent))

            if effect.is_increase():
                return (
                    agent_fluent,
                    em.auto_promote(
                        f_eval.constant_value() + new_value.constant_value()
                    )[0],
                )
            elif effect.is_decrease():
                return (
                    agent_fluent,
                    em.auto_promote(
                        f_eval.constant_value() - new_value.constant_value()
                    )[0],
                )
            else:
                raise NotImplementedError

    def _get_applicable_actions(
        self, agent: "Agent", state: "up.model.State"
    ) -> Iterator[Tuple["up.model.Action", Tuple["up.model.FNode", ...]]]:
        """
        Returns a view over all the `action + parameters` that are applicable in the given `State`.
        """
        for original_action, params, _ in self._grounder.get_grounded_actions():
            if self._is_applicable(agent, state, original_action, params):
                yield (original_action, params)

    def get_unsatisfied_conditions(
        self,
        agent: "Agent",
        state: "up.model.State",
        action_or_action_instance: Union["up.model.Action", "up.plans.ActionInstance"],
        parameters: Optional[Sequence["up.model.Expression"]] = None,
        early_termination: bool = False,
        full_check: bool = False,
    ) -> Tuple[List["up.model.FNode"], Optional[InapplicabilityReasons]]:
        """
        Returns the list of `unsatisfied action's conditions` evaluated in the given `state`,
        together with an optional reason.
        """
        action, params = self._get_action_and_parameters(
            action_or_action_instance,
            parameters,
        )
        g_action = self._ground_action(agent, action, params)
        if g_action is None:
            raise UPInvalidActionError(
                "The given action grounded with the given parameters does not create a valid action."
            )

        def evaluate(exp: FNode) -> FNode:
            return self._se.evaluate(agent, exp, state)

        reason: Optional[InapplicabilityReasons] = None
        unsatisfied_conditions: List[FNode] = []

        # Check preconditions
        for c in g_action.preconditions:
            evaluated_cond = evaluate(c)
            if (
                not evaluated_cond.is_bool_constant()
                or not evaluated_cond.bool_constant_value()
            ):
                unsatisfied_conditions.append(c)
                reason = InapplicabilityReasons.VIOLATES_CONDITIONS
                if early_termination:
                    return unsatisfied_conditions, reason

        updated_values: Dict["up.model.FNode", "up.model.FNode"] = {}
        assigned_fluent: Set["up.model.FNode"] = set()
        em = self._problem.environment.expression_manager

        if full_check:
            # Add simulated effects first
            sim_eff = g_action.simulated_effect
            if sim_eff is not None:
                for f, v in zip(
                    sim_eff.fluents,
                    sim_eff.function(self._problem, state, {}),
                ):
                    updated_values[f] = v
                    assigned_fluent.add(f)

            # Conditional effects
            for effect in g_action.conditional_effects:
                for e in effect.expand_effect(
                    cast(up.model.mixins.ObjectsSetMixin, self._problem)
                ):
                    if not e.fluent.type.is_bool_type():
                        evaluated_condition = evaluate(
                            e.condition
                        ).bool_constant_value()
                        if evaluated_condition:
                            try:
                                fluent, value = self._evaluate_effect(
                                    agent,
                                    e,
                                    state,
                                    updated_values,
                                    assigned_fluent,
                                    em,
                                    evaluated_condition=evaluated_condition,
                                )
                                assert fluent is not None and value is not None
                                updated_values[fluent] = value
                            except UPConflictingEffectsException:
                                reason = InapplicabilityReasons.CONFLICTING_EFFECTS
                                if early_termination:
                                    return unsatisfied_conditions, reason

            # Unconditional effects that interact with updated_values
            if updated_values:
                for effect in g_action.unconditional_effects:
                    for e in effect.expand_effect(
                        cast(up.model.mixins.ObjectsSetMixin, self._problem)
                    ):
                        ev_fluent = e.fluent.fluent()(*(map(evaluate, e.fluent.args)))
                        values = updated_values.get(ev_fluent, None)
                        if values is not None:
                            try:
                                fluent, value = self._evaluate_effect(
                                    agent,
                                    e,
                                    state,
                                    updated_values,
                                    assigned_fluent,
                                    em,
                                    evaluated_fluent=ev_fluent,
                                    evaluated_condition=True,
                                )
                                assert fluent is not None and value is not None
                                updated_values[fluent] = value
                            except UPConflictingEffectsException:
                                reason = InapplicabilityReasons.CONFLICTING_EFFECTS
                                if early_termination:
                                    return unsatisfied_conditions, reason

            # At this point you could also check state invariants on the partial new state
            new_partial_state = state.make_child(updated_values)
            # for si in self._state_invariants:
            #     if not self._se.evaluate(si, new_partial_state).bool_constant_value():
            #         unsatisfied_conditions.append(si)
            #         if reason is None:
            #             reason = InapplicabilityReasons.VIOLATES_STATE_INVARIANTS
            #         if early_termination:
            #             break

        return unsatisfied_conditions, reason

    def get_unsatisfied_goals(
        self, state: "up.model.State", early_termination: bool = False
    ) -> List["up.model.FNode"]:
        """
        Returns the list of `unsatisfied goals` evaluated in the given `state`.
        """
        unsatisfied_goals: List[FNode] = []
        for g in cast(up.model.Problem, self._problem).goals:
            g_eval = self._se.evaluate(g, state).bool_constant_value()
            if not g_eval:
                unsatisfied_goals.append(g)
                if early_termination:
                    break
        return unsatisfied_goals

    def _is_goal(self, state: "up.model.State") -> bool:
        """
        is_goal implementation
        """
        return len(self.get_unsatisfied_goals(state, early_termination=True)) == 0

    @property
    def name(self) -> str:
        return "ma_sequential_simulator"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        supported_kind = up.model.ProblemKind(version=LATEST_PROBLEM_KIND_VERSION)
        supported_kind.set_problem_class("ACTION_BASED_MULTI_AGENT")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_numbers("BOUNDED_TYPES")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FORALL_EFFECTS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        supported_kind.set_constraints_kind("STATE_INVARIANTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_quality_metrics("OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("TEMPORAL_OVERSUBSCRIPTION")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_oversubscription_kind("INT_NUMBERS_IN_OVERSUBSCRIPTION")
        supported_kind.set_oversubscription_kind("REAL_NUMBERS_IN_OVERSUBSCRIPTION")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UPSequentialSimulatorMA.supported_kind()


def evaluate_quality_metric(
    simulator: SequentialSimulatorMixin,
    quality_metric: "up.model.PlanQualityMetric",
    metric_value: Union[Fraction, int],
    state: "up.model.State",
    action: "up.model.Action",
    parameters: Tuple["up.model.FNode", ...],
    next_state: "up.model.State",
) -> Union[Fraction, int]:
    """
    Evaluates the value of the given metric.
    """
    if not isinstance(simulator._problem, up.model.Problem):
        raise NotImplementedError(
            "Currently this method is implemented only for classical and numeric problems."
        )
    se = StateEvaluator(simulator._problem)
    if quality_metric.is_minimize_action_costs():
        assert isinstance(quality_metric, MinimizeActionCosts)
        action_cost = quality_metric.get_action_cost(action)
        if action_cost is None:
            raise UPUsageError(
                "Can't evaluate Action cost when the cost is not set.",
                "You can explicitly set a default in the MinimizeActionCost constructor.",
            )
        if len(action.parameters) != len(parameters):
            raise UPUsageError(
                "The parameters length is different than the action's parameters length."
            )
        action_cost = action_cost.substitute(dict(zip(action.parameters, parameters)))
        assert isinstance(action_cost, up.model.FNode)
        return se.evaluate(action_cost, state).constant_value() + metric_value
    elif quality_metric.is_minimize_sequential_plan_length():
        return metric_value + 1
    elif (
        quality_metric.is_minimize_expression_on_final_state()
        or quality_metric.is_maximize_expression_on_final_state()
    ):
        assert isinstance(
            quality_metric,
            (MinimizeExpressionOnFinalState, MaximizeExpressionOnFinalState),
        )
        return se.evaluate(quality_metric.expression, next_state).constant_value()
    elif quality_metric.is_oversubscription():
        assert isinstance(quality_metric, Oversubscription)
        total_gain: Union[Fraction, int] = 0
        for goal, gain in quality_metric.goals.items():
            if se.evaluate(goal, next_state).bool_constant_value():
                total_gain += gain
        return total_gain
    else:
        raise NotImplementedError(
            f"QualityMetric {quality_metric} not supported by the UPSequentialSimulatorMA."
        )


def evaluate_quality_metric_in_initial_state(
    simulator: SequentialSimulatorMixin,
    quality_metric: "up.model.PlanQualityMetric",
) -> Union[Fraction, int]:
    """
    Returns the evaluation of the given metric in the initial state.
    """
    if not isinstance(simulator._problem, up.model.Problem):
        raise NotImplementedError(
            "Currently this method is implemented only for classical and numeric problems."
        )
    se = StateEvaluator(simulator._problem)
    initial_state = simulator.get_initial_state()
    if quality_metric.is_minimize_action_costs():
        return 0
    elif quality_metric.is_minimize_sequential_plan_length():
        return 0
    elif (
        quality_metric.is_minimize_expression_on_final_state()
        or quality_metric.is_maximize_expression_on_final_state()
    ):
        assert isinstance(
            quality_metric,
            (MinimizeExpressionOnFinalState, MaximizeExpressionOnFinalState),
        )
        return se.evaluate(quality_metric.expression, initial_state).constant_value()
    elif quality_metric.is_oversubscription():
        assert isinstance(quality_metric, Oversubscription)
        total_gain: Union[Fraction, int] = 0
        for goal, gain in quality_metric.goals.items():
            if se.evaluate(goal, initial_state).bool_constant_value():
                total_gain += gain
        return total_gain
    else:
        raise NotImplementedError(
            f"QualityMetric {quality_metric} not supported by the UPSequentialSimulatorMA."
        )
