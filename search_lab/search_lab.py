"""
Search Laboratory: A* and BFS for Warehouse Robot Navigation
===============================================================

Search problem formulation (Task 0):
    S  - the set of free grid cells (row, col)
    A  - {Up, Down, Left, Right}
    T  - moving one cell in the chosen direction, if that cell is free & in-bounds
    s0 - the cell marked 'S'
    G  - the cell marked 'G'
    c  - every move costs 1

This is deterministic: the same action in the same state always yields the same
next state (or is simply inapplicable). A solution is a sequence of actions
(equivalently, a sequence of cells) connecting s0 to a goal cell.
"""

import heapq
import math
from collections import deque

MOVES = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

MAIN_MAP = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

TRIVIAL_MAP = [
    "#####",
    "#SG##",
    "#####",
]

NO_SOLUTION_MAP = [
    "#######",
    "#S....#",
    "###.###",
    "#...#G#",
    "#######",
]

ALTERNATE_PATHS_MAP = [
    "#########",
    "#S..#...#",
    "#.#.#.#.#",
    "#.#...#.#",
    "#.#####.#",
    "#.......#",
    "#######G#",
    "#########",
]


class Grid:
    def __init__(self, lines):
        self.grid = [list(row) for row in lines]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._find("S")
        self.goal = self._find("G")

    def _find(self, sym):
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == sym:
                    return (r, c)
        raise ValueError(f"{sym} not in map")

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#"

    def neighbours(self, pos):
        for dr, dc in MOVES.values():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_free(nxt):
                yield nxt


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def zero_heuristic(a, b):
    return 0


def reconstruct(came_from, goal):
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def bfs(grid: Grid):
    """Blind search: expands states in order of number of moves (unweighted BFS)."""
    frontier = deque([grid.start])
    came_from = {grid.start: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == grid.goal:
            path = reconstruct(came_from, current)
            return path, len(path) - 1, expanded
        for nxt in grid.neighbours(current):
            if nxt not in came_from:
                came_from[nxt] = current
                frontier.append(nxt)

    return None, None, expanded


def astar(grid: Grid, heuristic=manhattan, weight=1.0):
    """Informed search: f(n) = g(n) + weight * h(n)."""
    start, goal = grid.start, grid.goal
    frontier = [(weight * heuristic(start, goal), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    expanded = 0
    closed = set()

    while frontier:
        _, g, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            path = reconstruct(came_from, current)
            return path, len(path) - 1, expanded

        for nxt in grid.neighbours(current):
            tentative_g = g + 1
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                f = tentative_g + weight * heuristic(nxt, goal)
                heapq.heappush(frontier, (f, tentative_g, nxt))

    return None, None, expanded


def report(name, path, length, expanded):
    print(f"[{name}] solution found: {path is not None}")
    if path is not None:
        print(f"[{name}] path length: {length}")
    print(f"[{name}] states expanded: {expanded}")


def run_map_tests():
    for title, lines in [
        ("Test 1: Original warehouse", MAIN_MAP),
        ("Test 2: Trivial case", TRIVIAL_MAP),
        ("Test 3: No solution", NO_SOLUTION_MAP),
        ("Test 4: Alternative paths", ALTERNATE_PATHS_MAP),
    ]:
        print(f"\n=== {title} ===")
        grid = Grid(lines)
        b_path, b_len, b_exp = bfs(grid)
        a_path, a_len, a_exp = astar(grid)
        report("BFS", b_path, b_len, b_exp)
        report("A*", a_path, a_len, a_exp)
        if b_path is not None and a_path is not None:
            print(f"Both found path of the same length? {b_len == a_len}")


def run_heuristic_investigation():
    print("\n=== Task 6: Heuristic investigation (original warehouse map) ===")
    grid = Grid(MAIN_MAP)
    variants = [
        ("Manhattan h(n)", manhattan, 1.0),
        ("Zero h(n)=0 (=> uniform-cost search)", zero_heuristic, 1.0),
        ("Euclidean distance", euclidean, 1.0),
        ("Manhattan x2 (inadmissible)", manhattan, 2.0),
    ]
    for label, h, w in variants:
        path, length, expanded = astar(grid, heuristic=h, weight=w)
        print(f"{label:38s} | found={path is not None} | "
              f"length={length} | expanded={expanded}")


if __name__ == "__main__":
    run_map_tests()
    run_heuristic_investigation()
