# Agents Laboratory Report — Goal-Based Warehouse Agent

## Task 1 — Understanding the Problem

1. **Environment:** The warehouse grid itself — a static, fully observable, deterministic,
   discrete 2D layout of free cells and shelving obstacles.
2. **Goal of the agent:** Reach the dispatch cell marked `G` from the loading-bay cell `S`
   without passing through any obstacle (`#`).
3. **Available actions:** `Up`, `Down`, `Left`, `Right` — each shifts the vehicle by one
   grid square.
4. **Information the agent must maintain:** its current position (row, column); the map of
   obstacles; and the goal location, so it can test whether a candidate move is legal
   (stays in bounds, not an obstacle) and whether it has reached the goal.
5. **Why goal-based, not simple reflex:** A simple reflex agent chooses an action from the
   current percept alone (e.g. "if wall ahead, turn"), which can loop or fail in a maze
   with dead ends. This agent instead maintains an explicit goal and *searches* over
   possible future states before acting, choosing the whole action sequence that is
   predicted to reach the goal. That look-ahead/goal-maintenance is exactly what defines a
   goal-based agent.

### Think About It — larger warehouse
Breadth-first search still finds a solution and is still optimal for unit-cost moves, but
its memory and time cost both grow with the number of reachable cells (O(V+E)), so on a
much larger map it becomes expensive. Additional difficulties: longer solution paths,
larger frontier/queue memory, and — if the map were not fully known in advance — the need
for a more informed strategy (e.g. A* with a distance heuristic) to search efficiently
rather than exploring uniformly outward from the start.

## Task 2 — Design

See the docstring in `warehouse_agent.py` for the block diagram. Summary:

```
Environment (grid) --percept: position--> Agent
   Agent: state -> goal test -> search over successor states -> plan
Agent --action (U/D/L/R)--> Environment (updates position)
```

Components: **environment** (grid), **current state** (row, col), **goal** (`G` cell),
**actions** (U/D/L/R), **decision-making component** (BFS planner producing a full action
sequence before execution).

## Task 3 — Prompt Engineering / Results

**Prompt used:** the specification prompt suggested in the lab sheet (grid representation,
collision-free path from S to G, avoid obstacles, print path or failure message, explain
the chosen algorithm).

**Program output (run of `warehouse_agent.py`):**

```
Start: (1, 1)   Goal: (1, 19)
Search algorithm used: Breadth-First Search (BFS)
Path length: 20 moves
Action sequence: ['Right','Right','Right','Down','Right','Right','Right','Up',
                  'Right','Right','Right','Right','Right','Right','Right','Right',
                  'Right','Right','Right','Right']
```

A 20-move path is found, dipping down one row and back up to avoid the shelving block,
then running straight along row 1 to the goal.

1. **Did it run on the first attempt?** Yes — the implementation is a standard BFS over a
   grid, which is a well-tested pattern.
2. **How to improve the prompt further?** Ask explicitly for the number of *states
   expanded* and for a visual overlay of the path on the map, and specify whether ties in
   path length should be broken in a particular direction.
3. **Search algorithm chosen:** Breadth-First Search.
4. **Why BFS is appropriate:** Every move has identical cost (1 grid square), so BFS's
   guarantee of finding the path with the fewest edges is equivalent to finding the
   shortest (least-cost) path — no heuristic or priority queue is needed, keeping the
   implementation simple while still optimal.
