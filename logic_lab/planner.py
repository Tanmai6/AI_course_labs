"""
Logical Planning Agent
=======================

Laboratory: "Logical Reasoning for Planning"

A state is represented as a frozenset of logical proposition strings, e.g.
    frozenset({"At(Robot,A)", "At(Package,A)"})

Each Action has:
    name             - human-readable label
    pos_pre, neg_pre - propositions that must be present / absent for the
                        action to be applicable
    pos_eff, neg_eff - propositions added / removed when the action is applied

An action a is applicable in state S  iff  pos_pre(a) subset S  and
neg_pre(a) disjoint from S   (this is exactly  S |= Preconditions(a) ).

Planning uses Breadth-First Search over the state graph induced by the
actions, which explores states in order of number of actions taken and so
returns a shortest plan.
"""

from collections import deque


class Action:
    def __init__(self, name, pos_pre=(), neg_pre=(), pos_eff=(), neg_eff=()):
        self.name = name
        self.pos_pre = frozenset(pos_pre)
        self.neg_pre = frozenset(neg_pre)
        self.pos_eff = frozenset(pos_eff)
        self.neg_eff = frozenset(neg_eff)

    def applicable(self, state):
        """Logical component: does state |= Preconditions(action)?"""
        return self.pos_pre <= state and self.neg_pre.isdisjoint(state)

    def apply(self, state):
        """Apply effects: remove negative effects, then add positive effects."""
        return (state - self.neg_eff) | self.pos_eff

    def __repr__(self):
        return self.name


def make_standard_actions(include_pickup=True):
    """Actions for the A/B/C warehouse-robot delivery problem."""
    actions = [
        Action("Move(A,B)", pos_pre=["At(Robot,A)"], neg_pre=[],
               pos_eff=["At(Robot,B)"], neg_eff=["At(Robot,A)"]),
        Action("Move(B,A)", pos_pre=["At(Robot,B)"],
               pos_eff=["At(Robot,A)"], neg_eff=["At(Robot,B)"]),
        Action("Move(B,C)", pos_pre=["At(Robot,B)"],
               pos_eff=["At(Robot,C)"], neg_eff=["At(Robot,B)"]),
        Action("Move(C,B)", pos_pre=["At(Robot,C)"],
               pos_eff=["At(Robot,B)"], neg_eff=["At(Robot,C)"]),
    ]
    if include_pickup:
        for loc in ["A", "B", "C"]:
            actions.append(
                Action(f"PickUp(Package,{loc})",
                       pos_pre=[f"At(Robot,{loc})", f"At(Package,{loc})"],
                       pos_eff=["Holding(Package)"],
                       neg_eff=[f"At(Package,{loc})"])
            )
    for loc in ["A", "B", "C"]:
        actions.append(
            Action(f"Drop(Package,{loc})",
                   pos_pre=[f"At(Robot,{loc})", "Holding(Package)"],
                   pos_eff=[f"At(Package,{loc})"],
                   neg_eff=["Holding(Package)"])
        )
    return actions


def plan(initial_state, goal, actions, verbose=False):
    """BFS planner. Returns (plan_actions, states_visited_in_order) or (None, None)."""
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal <= initial_state:
        return [], [initial_state]

    frontier = deque([initial_state])
    came_from = {initial_state: None}       # state -> (prev_state, action)

    while frontier:
        state = frontier.popleft()
        for action in actions:
            if not action.applicable(state):
                continue
            new_state = action.apply(state)
            if new_state in came_from:
                continue
            came_from[new_state] = (state, action)
            if goal <= new_state:
                return _reconstruct(came_from, new_state)
            frontier.append(new_state)

    return None, None  # no plan found


def _reconstruct(came_from, final_state):
    actions = []
    states = [final_state]
    current = final_state
    while came_from[current] is not None:
        prev_state, action = came_from[current]
        actions.append(action)
        states.append(prev_state)
        current = prev_state
    actions.reverse()
    states.reverse()
    return actions, states


def print_plan_trace(actions, states):
    print(f"S0: {sorted(states[0])}")
    for i, (a, s) in enumerate(zip(actions, states[1:]), start=1):
        print(f"  --{a}-->")
        print(f"S{i}: {sorted(s)}")


def run_test(title, initial_state, goal, actions):
    print(f"\n=== {title} ===")
    print(f"Initial state: {sorted(initial_state)}")
    print(f"Goal: {sorted(goal)}")
    result_actions, states = plan(initial_state, goal, actions)
    if result_actions is None:
        print("No plan found")
        return
    print(f"Plan found ({len(result_actions)} actions):")
    print_plan_trace(result_actions, states)


if __name__ == "__main__":
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}

    # Test A: Solvable problem (standard action set)
    run_test("Test A: Solvable problem", initial_state, goal,
              make_standard_actions(include_pickup=True))

    # Test B: Impossible problem (PickUp removed)
    run_test("Test B: Impossible problem (no PickUp action)", initial_state, goal,
              make_standard_actions(include_pickup=False))

    # Test C: Irrelevant actions - robot can move without carrying the package;
    # verify planner requires Package (not just Robot) to reach C.
    irrelevant_goal = {"At(Robot,C)"}  # NOT the same as the real delivery goal
    run_test("Test C: Irrelevant-action check (goal = robot at C, not package)",
              initial_state, irrelevant_goal, make_standard_actions(include_pickup=True))
    print("\nNote: reaching 'At(Robot,C)' is trivially possible via Move actions alone")
    print("and does NOT imply 'At(Package,C)' is satisfied - the planner correctly")
    print("treats these as different goals because they are different propositions.")
