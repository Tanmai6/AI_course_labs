import heapq

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


def find_symbol(grid, symbol):
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == symbol:
                return (r, c)
    raise ValueError(f"{symbol} not found in map")


def is_free(grid, pos):
    r, c = pos
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
        return False
    return grid[r][c] != "#"


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    frontier = [(manhattan(start, goal), 0, start)]  # (f, g, position)
    came_from = {start: None}   # position -> (prev_position, action)
    g_score = {start: 0}
    closed = set()

    while frontier:
        f, g, current = heapq.heappop(frontier)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        for action, (dr, dc) in MOVES.items():
            nxt = (current[0] + dr, current[1] + dc)
            if not is_free(grid, nxt) or nxt in closed:
                continue

            tentative_g = g + 1
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = (current, action)
                f_score = tentative_g + manhattan(nxt, goal)
                heapq.heappush(frontier, (f_score, tentative_g, nxt))

    return None  # no path found


def reconstruct_path(came_from, current):
    actions = []
    while came_from[current] is not None:
        prev, action = came_from[current]
        actions.append(action)
        current = prev
    actions.reverse()
    return actions


def main():
    start = find_symbol(WAREHOUSE_MAP, "S")
    goal = find_symbol(WAREHOUSE_MAP, "G")

    path = astar(WAREHOUSE_MAP, start, goal)

    if path is None:
        print("No path found.")
    else:
        print(f"Shortest path ({len(path)} moves):")
        print(path)


if __name__ == "__main__":
    main()