from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
from collections import namedtuple
from unified_planning.io.ma_pddl_writer import MAPDDLWriter

# TYPEs
Location = UserType("Location")
button = UserType("button")
door = UserType("door")
employer_door = UserType("employer_door")
manager_door = UserType("manager_door")
# door = UserType("door")
problem = MultiAgentProblem("RM_examaple")
s1 = Object("s1", button)
s2 = Object("s2", button)
s3 = Object("s3", button)
d1 = Object("d1", door)
d2 = Object("d2", door)
d3 = Object("d3", door)
br1 = Object("br1", employer_door)
br2 = Object("br2", employer_door)
br3 = Object("br3", employer_door)
bo1 = Object("bo1", manager_door)
bo2 = Object("bo2", manager_door)
bo3 = Object("bo3", manager_door)

# Righe/Colonne
l11 = Object("l11", Location)
l12 = Object("l12", Location)
l13 = Object("l13", Location)
l14 = Object("l14", Location)
l21 = Object("l21", Location)
l22 = Object("l22", Location)
l23 = Object("l23", Location)
l24 = Object("l24", Location)
l31 = Object("l31", Location)
l32 = Object("l32", Location)
l33 = Object("l33", Location)
l34 = Object("l34", Location)
l41 = Object("l41", Location)
l42 = Object("l42", Location)
l43 = Object("l43", Location)
l44 = Object("l44", Location)

problem.add_objects(
    [l11, l12, l13, l14, l21, l22, l23, l24, l31, l32, l33, l34, l41, l42, l43, l44]
)
problem.add_object(s1)
problem.add_object(s2)
problem.add_object(s3)
problem.add_object(d1)
problem.add_object(d2)
problem.add_object(d3)
problem.add_object(br1)
problem.add_object(br2)
problem.add_object(br3)
problem.add_object(bo1)
problem.add_object(bo2)
problem.add_object(bo3)

# FLUENTS
activeButton = Fluent("activeButton", button=button)
pressButton = Fluent(
    "pressButton",
    BoolType(),
    button=button,
    position=Location,
    connect_from=Location,
    connect_to=Location,
)
# has_door =  Fluent("has_door", BoolType(), door=door, connect_from=Location, connect_to=Location)
has_door_manager = Fluent(
    "has_door_manager",
    BoolType(),
    employer_door=employer_door,
    connect_from=Location,
    connect_to=Location,
)
has_door_manager_manager = Fluent(
    "has_door_manager_manager",
    BoolType(),
    manager_door=manager_door,
    connect_from=Location,
    connect_to=Location,
)
free = Fluent("free", BoolType())
open_door_employed = Fluent(
    "open_door_employed",
    BoolType(),
    employer_door=employer_door,
    connect_from=Location,
    connect_to=Location,
)
open_manager_door0 = Fluent(
    "open_manager_door0",
    BoolType(),
    manager_door=manager_door,
    connect_from=Location,
    connect_to=Location,
)
open_manager_door1 = Fluent(
    "open_manager_door1",
    BoolType(),
    manager_door=manager_door,
    connect_from=Location,
    connect_to=Location,
)
open_manager_door2 = Fluent(
    "open_manager_door2",
    BoolType(),
    manager_door=manager_door,
    connect_from=Location,
    connect_to=Location,
)

# AGENTs
a1 = Agent("a1", problem)
a2 = Agent("a2", problem)
a3 = Agent("a3", problem)
a4 = Agent("a4", problem)
a5 = Agent("a5", problem)

is_connected = Fluent("is_connected", BoolType(), l1=Location, l2=Location)
problem.ma_environment.add_fluent(is_connected, default_initial_value=False)
problem.ma_environment.add_fluent(free, default_initial_value=False)
problem.ma_environment.add_fluent(open_door_employed, default_initial_value=False)
problem.ma_environment.add_fluent(open_manager_door0, default_initial_value=False)
problem.ma_environment.add_fluent(open_manager_door1, default_initial_value=False)
problem.ma_environment.add_fluent(open_manager_door2, default_initial_value=False)


pos = Fluent("pos", position=Location)
# move_now = Fluent("move_now", l_from=Location, l_to=Location)
a1.add_public_fluent(pos, default_initial_value=False)
a2.add_public_fluent(pos, default_initial_value=False)
a3.add_public_fluent(pos, default_initial_value=False)
a4.add_public_fluent(pos, default_initial_value=False)
a5.add_public_fluent(pos, default_initial_value=False)


problem.ma_environment.add_fluent(activeButton, default_initial_value=False)
problem.ma_environment.add_fluent(pressButton, default_initial_value=False)
# problem.ma_environment.add_fluent(has_door, default_initial_value=False)
problem.ma_environment.add_fluent(has_door_manager, default_initial_value=False)
problem.ma_environment.add_fluent(has_door_manager_manager, default_initial_value=False)


# ACTIONS
move = InstantaneousAction("move", l_from=Location, l_to=Location)
l_from = move.parameter("l_from")
l_to = move.parameter("l_to")
move.add_precondition(pos(l_from))
move.add_precondition(free)
move.add_precondition(is_connected(l_from, l_to))
move.add_effect(pos(l_to), True)
move.add_effect(pos(l_from), False)


# open_employer_door########################################################################################
start_open_employer_door = InstantaneousAction(
    "start_open_employer_door", br=employer_door, l_from=Location, l_to=Location
)
l_from = start_open_employer_door.parameter("l_from")
l_to = start_open_employer_door.parameter("l_to")
br = start_open_employer_door.parameter("br")
start_open_employer_door.add_precondition(free)
start_open_employer_door.add_precondition(pos(l_from))
start_open_employer_door.add_precondition(has_door_manager(br, l_from, l_to))
start_open_employer_door.add_effect((free), False)
start_open_employer_door.add_effect(open_door_employed(br, l_from, l_to), True)

open_employer_door = InstantaneousAction(
    "open_employer_door", br=employer_door, l_from=Location, l_to=Location
)
l_from = open_employer_door.parameter("l_from")
l_to = open_employer_door.parameter("l_to")
br = open_employer_door.parameter("br")
open_employer_door.add_precondition(pos(l_from))
open_employer_door.add_precondition(open_door_employed(br, l_from, l_to))
open_employer_door.add_effect(pos(l_to), True)
open_employer_door.add_effect(pos(l_from), False)

end_open_employer_door = InstantaneousAction(
    "end_open_employer_door", br=employer_door, l_from=Location, l_to=Location
)
l_from = end_open_employer_door.parameter("l_from")
l_to = end_open_employer_door.parameter("l_to")
br = end_open_employer_door.parameter("br")
end_open_employer_door.add_precondition(open_door_employed(br, l_from, l_to))
end_open_employer_door.add_effect(free, True)
end_open_employer_door.add_effect(open_door_employed(br, l_from, l_to), False)
end_open_employer_door.add_effect(has_door_manager(br, l_from, l_to), False)
end_open_employer_door.add_effect(has_door_manager(br, l_from, l_to), False)
end_open_employer_door.add_effect(is_connected(l_to, l_from), False)
# open_employer_door########################################################################################


start_open_manager_door = InstantaneousAction(
    "start_open_manager_door", bo=manager_door, l_from=Location, l_to=Location
)
l_from = start_open_manager_door.parameter("l_from")
l_to = start_open_manager_door.parameter("l_to")
bo = start_open_manager_door.parameter("bo")
start_open_manager_door.add_precondition(free)
start_open_manager_door.add_precondition(pos(l_from))
start_open_manager_door.add_precondition(has_door_manager_manager(bo, l_from, l_to))
# row.add_precondition(is_connected(l_from, l_to))
start_open_manager_door.add_effect(free, False)
start_open_manager_door.add_effect(open_manager_door0(bo, l_from, l_to), True)

end_open_manager_door1 = InstantaneousAction(
    "end_open_manager_door1", bo=manager_door, l_from=Location, l_to=Location
)
l_from = end_open_manager_door1.parameter("l_from")
l_to = end_open_manager_door1.parameter("l_to")
bo = end_open_manager_door1.parameter("bo")
end_open_manager_door1.add_precondition(open_manager_door0(bo, l_from, l_to))
end_open_manager_door1.add_precondition(pos(l_from))
end_open_manager_door1.add_effect(open_manager_door0(bo, l_from, l_to), False)
end_open_manager_door1.add_effect(open_manager_door1(bo, l_from, l_to), True)
end_open_manager_door1.add_effect(pos(l_to), True)
end_open_manager_door1.add_effect(pos(l_from), False)

end_open_manager_door2 = InstantaneousAction(
    "end_open_manager_door2", bo=manager_door, l_from=Location, l_to=Location
)
l_from = end_open_manager_door2.parameter("l_from")
l_to = end_open_manager_door2.parameter("l_to")
bo = end_open_manager_door2.parameter("bo")
end_open_manager_door2.add_precondition(open_manager_door1(bo, l_from, l_to))
end_open_manager_door2.add_precondition(pos(l_from))
end_open_manager_door2.add_effect(open_manager_door1(bo, l_from, l_to), False)
end_open_manager_door2.add_effect(open_manager_door2(bo, l_from, l_to), True)
end_open_manager_door2.add_effect(pos(l_to), True)
end_open_manager_door2.add_effect(pos(l_from), False)

end_open_manager_door3 = InstantaneousAction(
    "end_open_manager_door3", bo=manager_door, l_from=Location, l_to=Location
)
l_from = end_open_manager_door3.parameter("l_from")
l_to = end_open_manager_door3.parameter("l_to")
bo = end_open_manager_door3.parameter("bo")
end_open_manager_door3.add_precondition(open_manager_door2(bo, l_from, l_to))
end_open_manager_door3.add_precondition(pos(l_from))
end_open_manager_door3.add_effect(pos(l_to), True)
end_open_manager_door3.add_effect(pos(l_from), False)

end_open_manager_door = InstantaneousAction(
    "end_open_manager_door", bo=manager_door, l_from=Location, l_to=Location
)
l_from = end_open_manager_door.parameter("l_from")
l_to = end_open_manager_door.parameter("l_to")
bo = end_open_manager_door.parameter("bo")
end_open_manager_door.add_precondition(open_manager_door2(bo, l_from, l_to))
end_open_manager_door.add_effect(free, True)
end_open_manager_door.add_effect(open_manager_door2(bo, l_from, l_to), False)

push_button = InstantaneousAction(
    "push_button",
    button=button,
    loc=Location,
    l_from=Location,
    l_to=Location,
)  # , l_from=Location, l_to=Location)
button = push_button.parameter("button")
loc = push_button.parameter("loc")
l_from = push_button.parameter("l_from")
l_to = push_button.parameter("l_to")
push_button.add_precondition(pos(loc))
push_button.add_precondition(free)
push_button.add_precondition(pressButton(button, loc, l_from, l_to))
push_button.add_precondition(Not(activeButton(button)))
push_button.add_effect(activeButton(button), True)
push_button.add_effect(is_connected(l_from, l_to), True)

a1.add_action(move)
a1.add_action(start_open_employer_door)
a1.add_action(open_employer_door)
a1.add_action(end_open_employer_door)
a1.add_action(start_open_manager_door)
a1.add_action(end_open_manager_door1)
a1.add_action(end_open_manager_door2)
a1.add_action(end_open_manager_door3)
a1.add_action(end_open_manager_door)
a1.add_action(push_button)

a2.add_action(move)
a2.add_action(start_open_employer_door)
a2.add_action(open_employer_door)
a2.add_action(end_open_employer_door)
a2.add_action(start_open_manager_door)
a2.add_action(end_open_manager_door1)
a2.add_action(end_open_manager_door2)
a2.add_action(end_open_manager_door3)
a2.add_action(end_open_manager_door)
a2.add_action(push_button)

a3.add_action(move)
a3.add_action(start_open_employer_door)
a3.add_action(open_employer_door)
a3.add_action(end_open_employer_door)
a3.add_action(start_open_manager_door)
a3.add_action(end_open_manager_door1)
a3.add_action(end_open_manager_door2)
a3.add_action(end_open_manager_door3)
a3.add_action(end_open_manager_door)
a3.add_action(push_button)

a4.add_action(move)
a4.add_action(start_open_employer_door)
a4.add_action(open_employer_door)
a4.add_action(end_open_employer_door)
a4.add_action(start_open_manager_door)
a4.add_action(end_open_manager_door1)
a4.add_action(end_open_manager_door2)
a4.add_action(end_open_manager_door3)
a4.add_action(end_open_manager_door)
a4.add_action(push_button)

a5.add_action(move)
a5.add_action(start_open_employer_door)
a5.add_action(open_employer_door)
a5.add_action(end_open_employer_door)
a5.add_action(start_open_manager_door)
a5.add_action(end_open_manager_door1)
a5.add_action(end_open_manager_door2)
a5.add_action(end_open_manager_door3)
a5.add_action(end_open_manager_door)
a5.add_action(push_button)

problem.add_agent(a1)
problem.add_agent(a2)
problem.add_agent(a3)
problem.add_agent(a4)
problem.add_agent(a5)

# INITIAL VALUEs
# Griglia
connections = [
    (l11, l12),
    (l12, l13),  # (l13, l14), -> employer_door
    (l21, l22),
    (l22, l23),  # (l23, l24), -> employer_door
    (l31, l32),
    (l33, l34),  # (l32, l33), -> employer_door
    (l42, l43),  # (l43, l44), ->door (l41, l42),->manager_door
    (l11, l21),
    (l21, l31),
    (l31, l41),
    (l12, l22),
    (l22, l32),
    (l32, l42),
    (l23, l33),
    (l33, l43),
    # (l14, l24), (l34, l44) -> manager_doors
]

for connection in connections:
    problem.set_initial_value(is_connected(connection[0], connection[1]), True)
    problem.set_initial_value(is_connected(connection[1], connection[0]), True)


# -------------------------------------------------------------------------
problem.set_initial_value(activeButton(s1), False)
problem.set_initial_value(activeButton(s2), False)
problem.set_initial_value(activeButton(s3), False)

problem.set_initial_value(Dot(a1, pos(l41)), True)
problem.set_initial_value(Dot(a2, pos(l33)), True)
problem.set_initial_value(Dot(a3, pos(l21)), True)
problem.set_initial_value(Dot(a4, pos(l11)), True)
problem.set_initial_value(Dot(a5, pos(l24)), True)

problem.set_initial_value(pressButton(s1, l13, l43, l44), True)
problem.set_initial_value(pressButton(s1, l13, l44, l43), True)

problem.set_initial_value(pressButton(s2, l22, l13, l23), True)
problem.set_initial_value(pressButton(s2, l22, l23, l13), True)

problem.set_initial_value(pressButton(s3, l43, l24, l34), True)
problem.set_initial_value(pressButton(s3, l43, l34, l24), True)
# Door ------------------------------------------------------
# problem.set_initial_value(has_door(d1, l43, l44), True)
# problem.set_initial_value(has_door(d1, l44, l43), True)

# problem.set_initial_value(has_door(d2, l13, l23), True)
# problem.set_initial_value(has_door(d2, l23, l13), True)

# problem.set_initial_value(has_door(d3, l24, l34), True)
# problem.set_initial_value(has_door(d3, l34, l24), True)
# employer_door ------------------------------------------------------
problem.set_initial_value(has_door_manager(br1, l13, l14), True)
problem.set_initial_value(has_door_manager(br1, l14, l13), True)

problem.set_initial_value(has_door_manager(br2, l23, l24), True)
problem.set_initial_value(has_door_manager(br2, l24, l23), True)

problem.set_initial_value(has_door_manager(br3, l32, l33), True)
problem.set_initial_value(has_door_manager(br3, l33, l32), True)
# manager_door ------------------------------------------------------
problem.set_initial_value(has_door_manager_manager(bo1, l41, l42), True)
problem.set_initial_value(has_door_manager_manager(bo1, l42, l41), True)

problem.set_initial_value(has_door_manager_manager(bo2, l44, l34), True)
problem.set_initial_value(has_door_manager_manager(bo2, l34, l44), True)

problem.set_initial_value(has_door_manager_manager(bo3, l14, l24), True)
problem.set_initial_value(has_door_manager_manager(bo3, l24, l14), True)

problem.set_initial_value(free, True)


# GOALs
problem.add_goal(Dot(a1, pos(l14)))
problem.add_goal(Dot(a2, pos(l14)))
problem.add_goal(Dot(a3, pos(l14)))
problem.add_goal(Dot(a4, pos(l14)))
problem.add_goal(Dot(a5, pos(l14)))
problem.add_goal(free)


# Generation of MA-PDDLs
w = MAPDDLWriter(problem, explicit_false_initial_states=False)
w.write_ma_domain("RM_example_office_cn")
w.write_ma_problem("RM_example_office_cn")

w = MAPDDLWriter(problem, explicit_false_initial_states=False, unfactored=True)
w.write_ma_domain("RM_example_office_cn")
w.write_ma_problem("RM_example_office_cn")


with OneshotPlanner(name="fmap") as planner:
    result = planner.solve(problem, None)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("FMAP returned: %s" % result.plan, result.plan.all_sequential_plans())
        # [print(f"{idx} Sequential Plans: {seq_plan}") for idx, seq_plan in enumerate(result.plan.all_sequential_plans())] #If you want all Sequential Plans
        print("\n")
        print("\n Adjacency list:", result.plan.get_adjacency_list)
        print("\n result:", result)
        # print("Returned office_graph: \n", result.plan.get_graph_file("office_graph"))
    else:
        print("Log Error:", result)


from unified_planning.plot import (
    plot_plan,  # plot_plan plots all the types of plans, but is not customizable, while specific methods
    show_partial_order_plan,
)

# show_partial_order_plan(result.plan, "MA_office") #If you want show a POP
