"""
Goal-Based Agent for the Warehouse Navigation Problem
======================================================

Laboratory: "Agents - Constructing a Goal-Based Agent using an LLM"

Design (Task 2)
---------------
Environment      : the warehouse grid (static, fully observable, deterministic,
                    discrete). Cells are free ('.'), obstacles ('#'), start ('S')
                    or goal ('G').
Current state     : the agent's (row, col) position in the grid.
Goal              : reach the cell marked 'G'.
Actions           : Up, Down, Left, Right (each moves the agent one cell).
Decision-making   : a search component (Breadth-First Search) that looks ahead
                    over the whole state space to select the sequence of actions
                    that reaches the goal. This makes the agent GOAL-BASED
                    rather than a simple reflex agent: it does not just react to
                    the immediate percept, it maintains a goal and reasons about
                    future consequences of actions before acting.

Block diagram:

    Environment (grid) --percept (position)--> Agent
        Agent: [state] -> [goal test] -> [search over successor states] -> [plan/actions]
    Agent --action (U/D/L/R)--> Environment (updates position)

Algorithm chosen: Breadth-First Search (BFS).
Why: every move has the same cost (1), so BFS is guaranteed to find a shortest
path (in terms of number of moves) and is simple to implement and verify.
(An LLM asked for this task typically proposes BFS or A*; BFS is sufficient
and optimal here because the graph is unweighted.)
"""

from collections import deque

WAREHOUSE_MAP = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class Warehouse:
    """The environment."""

    def __init__(self, grid_lines):
        self.grid = [list(row) for row in grid_lines]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, symbol):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == symbol:
                    return (r, c)
        raise ValueError(f"Symbol {symbol} not found in map")

    def is_free(self, pos):
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.grid[r][c] != "#"


class GoalBasedAgent:
    """The agent: knows the goal, maintains state, and searches for a plan."""

    def __init__(self, environment: Warehouse):
        self.env = environment
        self.state = environment.start
        self.goal = environment.goal

    def search_plan(self):
        """Breadth-first search for a collision-free path from state to goal.

        Returns (path_of_positions, path_of_actions) or (None, None) if no
        path exists.
        """
        start, goal = self.state, self.goal
        frontier = deque([start])
        came_from = {start: None}
        action_taken = {start: None}

        while frontier:
            current = frontier.popleft()
            if current == goal:
                return self._reconstruct(came_from, action_taken, current)

            for action, (dr, dc) in MOVES.items():
                nxt = (current[0] + dr, current[1] + dc)
                if self.env.is_free(nxt) and nxt not in came_from:
                    came_from[nxt] = current
                    action_taken[nxt] = action
                    frontier.append(nxt)

        return None, None

    @staticmethod
    def _reconstruct(came_from, action_taken, current):
        positions = []
        actions = []
        while current is not None:
            positions.append(current)
            if action_taken[current] is not None:
                actions.append(action_taken[current])
            current = came_from[current]
        positions.reverse()
        actions.reverse()
        return positions, actions


def main():
    env = Warehouse(WAREHOUSE_MAP)
    agent = GoalBasedAgent(env)

    print(f"Start: {agent.state}   Goal: {agent.goal}")
    positions, actions = agent.search_plan()

    if positions is None:
        print("No path exists from S to G.")
        return

    print(f"Search algorithm used: Breadth-First Search (BFS)")
    print(f"Path length: {len(actions)} moves")
    print(f"Action sequence: {actions}")
    print(f"Visited positions: {positions}")


if __name__ == "__main__":
    main()
