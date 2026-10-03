# Search Laboratory Report — A* and BFS for Warehouse Navigation

## Task 0 — Formulate the search problem

| Component | Specification |
|---|---|
| State S | a free grid cell `(row, col)` |
| Actions A | `{Up, Down, Left, Right}` |
| Transition T | move one cell in the chosen direction if the destination is in-bounds and not `#` |
| Initial state s0 | the cell marked `S` |
| Goal G | the cell marked `G` |
| Cost c | 1 per move |

(a) A state needs only its `(row, col)` coordinates — the grid itself is static and fully
known, so no other information is required to determine future behaviour.
(b) An action is invalid if it would move outside the grid bounds or onto an obstacle cell.
(c) Yes — deterministic: a given action in a given state always produces the same
successor (or is simply blocked).
(d) A solution is any sequence of actions taking the robot from S to G without ever
crossing an obstacle; an *optimal* solution is the shortest such sequence.

## Task 1 — Agent design

State representation: `(row, col)` tuples. Warehouse representation: list of strings
indexed `grid[row][col]`. Valid actions: check `is_free(next_cell)`. Goal recognition:
`current == goal`. Frontier: FIFO queue for BFS, min-heap keyed on `f(n)=g(n)+h(n)` for A*.
Path reconstruction: a `came_from` dictionary walked backwards from the goal to the start.
Reported at termination: whether a solution was found, the path, its length, and the number
of states expanded.

## Task 2/3 — A* implementation and systematic testing

`search_lab.py` implements both BFS (blind) and A* (informed, Manhattan heuristic) and runs
four required tests plus a heuristic investigation.

```
=== Test 1: Original warehouse ===
BFS: found=True, length=40, expanded=64
A*:  found=True, length=40, expanded=64

=== Test 2: Trivial case ===
BFS: found=True, length=1, expanded=2
A*:  found=True, length=1, expanded=2

=== Test 3: No solution ===
BFS: found=False, expanded=9
A*:  found=False, expanded=9

=== Test 4: Alternative paths ===
BFS: found=True, length=11, expanded=23
A*:  found=True, length=11, expanded=18
```

- **Test 1 (original map):** both algorithms agree on a 40-move shortest path; because the
  original warehouse is essentially one narrow corridor with few branch points, A*'s
  heuristic provides little extra pruning over BFS here (both expand the same 64 states).
- **Test 2 (trivial):** both correctly return the one-step solution — no algorithmic
  surprises expected, and none occurred.
- **Test 3 (no solution):** both correctly report failure (`found=False`) after exhausting
  the reachable region (9 cells), rather than looping forever — confirming termination on
  unsolvable instances.
- **Test 4 (map with genuine alternative routes):** both find the same optimal length (11),
  but here A* expands fewer states (18 vs 23) because its heuristic actively steers search
  toward the goal once multiple branches are available, whereas BFS explores all branches
  equally regardless of direction.

## Task 4 — Inspecting the algorithm

| Concept | Where it appears |
|---|---|
| State | `(row, col)` tuples throughout |
| Action | keys of `MOVES` dict, applied via `Grid.neighbours` |
| Transition | `Grid.neighbours` / `Grid.is_free` |
| Goal test | `current == goal` inside `bfs`/`astar` |
| g(n) | `g_score` dict / `tentative_g` in `astar` |
| h(n) | `heuristic(nxt, goal)` call |
| f(n) | `tentative_g + weight * heuristic(...)` pushed onto the heap |
| Frontier | `deque` (BFS) / `heapq` binary heap (A*) |
| Visited states | `came_from` keys (BFS) / `closed` set (A*) |
| Path reconstruction | `reconstruct()` |

(a) A* uses a **binary min-heap** (`heapq`) ordered by `f(n)`.
(b) It selects the frontier entry with smallest `f(n) = g(n) + h(n)`.
(c) The heuristic is calculated once per generated neighbour, when computing `f` for the
heap push.
(d) Yes — `f = tentative_g + weight * heuristic(nxt, goal)` computes it explicitly.
(e) Repeated exploration is prevented by tracking `g_score`/`closed`: a state is only
re-expanded if a strictly cheaper path to it is found (`tentative_g < g_score[nxt]`), and
already-closed states are skipped outright.

## Task 5 — A* vs. BFS comparison

See the Task 2/3 results table above (Tests 1 and 4 are the head-to-head comparisons).

(a) Yes, both algorithms found a solution in every solvable test.
(b) Yes, they always found paths of the same length — because all move costs are equal (1),
BFS is itself optimal, so there is no length advantage available for A* to find.
(c) On Test 4, A* expanded fewer states (18 vs 23); on the tighter corridor of Test 1 they
expanded the same number.
(d) A* can expand fewer states because the Manhattan heuristic gives it a directional
preference — it prioritises frontier nodes that are closer (by straight-line/grid distance)
to the goal, so it avoids wasting time growing the frontier away from the goal, an
advantage that only materialises when the map actually offers alternative directions to
explore (as in Test 4) rather than a single forced corridor (as in Test 1).

## Task 6 — Heuristic investigation

```
Manhattan h(n)                         | found=True | length=40 | expanded=64
Zero h(n)=0 (=> uniform-cost search)   | found=True | length=40 | expanded=64
Euclidean distance                     | found=True | length=40 | expanded=64
Manhattan x2 (inadmissible)            | found=True | length=40 | expanded=64
```

Why Manhattan distance is appropriate: the robot can only move horizontally/vertically
(no diagonals), so the Manhattan distance is exactly the cost of an *unobstructed* shortest
path — it never overestimates the true remaining cost, i.e. it is admissible.

On this particular (corridor-like) map all four heuristic variants happen to expand the
same 64 states and find the same optimal 40-move path, because the map offers almost no
branching for a heuristic to exploit — the same effect seen when comparing A* to BFS in
Task 5. On a map with more open branching (e.g. the alternate-paths map), `h=0` degrades
A* to plain uniform-cost search (behaving like BFS on unit costs) and would typically
expand more states than Manhattan, while Euclidean distance (which never overestimates
grid distance either, since it is always ≤ Manhattan distance for axis-aligned moves)
remains admissible but slightly less informative than Manhattan. Doubling the heuristic
(`2×Manhattan`) makes it **inadmissible** — it can overestimate the true remaining cost —
which can cause A* to still find *a* path but no longer guarantee the *shortest* one; on
this map it happened not to distort the result, but that is not guaranteed in general.

### Think About It — admissibility
When `h(n)` exceeds the true remaining cost (as with the doubled heuristic), A* becomes
overly "confident" that certain nodes are close to the goal and can commit to a
suboptimal path early, potentially expanding fewer states but sacrificing the optimality
guarantee. When `h(n)` is 0 (too conservative), A* degenerates into uniform-cost search /
BFS-like exploration, remaining optimal but losing the efficiency benefit of a good
heuristic.

## Task 7 — Evaluating the LLM-generated agent

1. **Correct immediately:** the grid representation, neighbour generation, and BFS
   implementation are standard patterns and worked first try.
2. **Bugs/design issues found:** none in the final version, but an early draft's A* did not
   guard against re-expanding closed nodes from a stale heap entry — fixed by checking
   `if current in closed: continue` after popping.
3. **How discovered:** by running Test 3 (no-solution map) repeatedly and confirming
   `states expanded` stayed finite and bounded rather than growing unexpectedly.
4. **Unfamiliar terminology/data structures:** none beyond the standard `heapq` module.
5. **Modifications made:** added the `closed` set guard described above, and separated the
   heuristic function as a parameter so Task 6's experiments could reuse the same `astar`.
6. **Most useful tests:** Test 3 (no-solution) for verifying termination, and Test 4
   (alternative paths) for actually seeing A* outperform BFS — Test 1's narrow-corridor map
   masked any difference between the two algorithms.
7. **Could the program have been trusted without testing?** No — the algorithms *looked*
   correct on inspection, but only executing them against a deliberately unsolvable map and
   a deliberately branchy map actually confirmed correct termination and revealed A*'s
   real advantage.
8. **New understanding of A*:** that A*'s benefit over BFS is entirely conditional on the
   map having real branching for the heuristic to prune — on a single forced corridor the
   two algorithms are indistinguishable in states expanded.

## Final Reflection

1. **Why formulate before coding:** a precise `(S, A, T, s0, G, c)` specification removes
   ambiguity about what counts as a valid state/action/goal, so the resulting search
   algorithm is built on an unambiguous model rather than an implicit, possibly incorrect,
   assumption baked into generated code.
2. **A* as "informed" search:** it incorporates problem-specific knowledge (the heuristic
   `h(n)`) about the estimated distance to the goal to guide which frontier node to expand
   next, rather than expanding purely in discovery order as blind search does.
3. **Why the heuristic matters:** an admissible heuristic preserves A*'s optimality
   guarantee while potentially reducing the number of states expanded; a poorly chosen or
   inadmissible heuristic can either waste the informedness advantage (h too weak) or break
   the optimality guarantee (h too strong), as seen with the doubled heuristic.
4. **LLM contribution:** rapid, correct boilerplate for standard search-algorithm patterns
   (frontier management, heap-based priority ordering, path reconstruction), letting effort
   focus on problem formulation and testing rather than syntax.
5. **Risk of accepting ungenerated code without testing:** subtle correctness bugs (e.g. the
   missing closed-set guard) or a program that merely "looks plausible" without ever being
   checked against edge cases like unsolvable maps could silently propagate into a deployed
   system that occasionally loops forever or returns a non-optimal path while claiming
   otherwise.
