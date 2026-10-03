# Logic Laboratory Report — Planning with Logical Preconditions/Effects

## Task 0 — Understand the planning problem

- **Initial state I:** `{At(Robot,A), At(Package,A)}`
- **Goal G:** `{At(Package,C)}`
- **Actions:** `Move(A,B)`, `Move(B,A)`, `Move(B,C)`, `Move(C,B)`, `PickUp(Package,·)`, `Drop(Package,·)`
  (preconditions/effects as given in the lab sheet).

**Is `PickUp(Package,A)` applicable in I?** Yes — its preconditions `At(Robot,A)` and
`At(Package,A)` are both true in I.

**Is `Drop(Package,C)` applicable in I?** No — its preconditions require `At(Robot,C)` and
`Holding(Package)`, neither of which holds in I.

## Task 1 — Manual plan

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 (after PickUp(Package,A)) | At(Robot,A), Holding(Package) |
| S2 (after Move(A,B)) | At(Robot,B), Holding(Package) |
| S3 (after Move(B,C)) | At(Robot,C), Holding(Package) |
| S4 (after Drop(Package,C)) | At(Robot,C), At(Package,C) |

S4 satisfies the goal `At(Package,C)`.

## Task 2/3 — LLM-generated planner, run on the warehouse problem

`planner.py` implements exactly this specification (states as sets of propositions, actions
with positive/negative preconditions and effects, BFS over the state graph). Output:

```
=== Test A: Solvable problem ===
Plan found (4 actions):
S0: At(Package,A), At(Robot,A)
  --PickUp(Package,A)-->
S1: At(Robot,A), Holding(Package)
  --Move(A,B)-->
S2: At(Robot,B), Holding(Package)
  --Move(B,C)-->
S3: At(Robot,C), Holding(Package)
  --Drop(Package,C)-->
S4: At(Package,C), At(Robot,C)
```

This exactly matches the manually derived plan.

```
=== Test B: Impossible problem (no PickUp action) ===
No plan found
```
Correct — with `PickUp` removed the robot can never hold the package, so BFS exhausts the
reachable state space and reports failure rather than inventing an action.

```
=== Test C: Irrelevant-action check (goal = robot at C, not package) ===
Plan found (2 actions): Move(A,B), Move(B,C)
```
This deliberately-different goal (`At(Robot,C)`) is trivially reachable by moving alone. It
confirms the planner distinguishes `At(Robot,C)` from `At(Package,C)` — the robot reaching
C is *not* treated as equivalent to delivering the package, because they are different
propositions and the goal test only checks for the propositions actually listed in G.

## Task 4 — Logic and search

```
Current state
   ↓
Check action preconditions   <- LOGICAL REASONING: S |= Preconditions(a)?
   ↓
[keep action if applicable]
   ↓
Generate successor state     <- apply effects: S' = Apply(S,a)
   ↓
Search over alternatives     <- SEARCH: BFS frontier decides which successor to expand next
   ↓
Goal?
```

**In words:** logic determines, for a given state, *which actions are even possible*
(precondition satisfaction) and *what the world looks like after* applying one (effects).
Search decides, among all the states that logic says are reachable, *in what order to
explore them* in pursuit of the goal. Logic answers "what is possible"; search answers
"what to try, and in what order."

## Task 5 — LLM self-verification vs. independent verification

Trust **(b) the independently executed state transitions** more. An LLM's natural-language
explanation of why a plan is valid is itself a generated artefact — it can sound convincing
while being subtly wrong (e.g. citing the wrong precondition or skipping an effect). The
Python program's step-by-step state transitions are computed mechanically by re-applying
the same precondition/effect logic used to generate the plan, so agreement between the plan
and the transition trace is a genuine, checkable proof, not just a plausible-sounding
narrative. This is the same idea explored with Prolog in the optional extension: a
*second, independent* mechanism checking the first is stronger evidence than the first
mechanism explaining itself.

## Optional Extension — Prolog as a verifier (`planner.pl`)

- `can_move(a,b)` succeeds because `connected(a,b)` is an explicit fact and the rule
  `can_move(X,Y) :- connected(X,Y)` unifies directly with it.
- `can_move(a,c)` fails because there is no fact `connected(a,c)`, and the single-step rule
  does not chain through `b`; Prolog only proves what follows from the given facts/rules.
- The rule `can_move(X,Y) :- connected(X,Y)` is precisely the logical implication
  `Connected(X,Y) → CanMove(X,Y)`.
- For the proposed plan `Move(a,b), Move(b,c)`: `valid_move(a,b)` and `valid_move(b,c)`
  succeed, `valid_move(a,c)` fails — matching the expected result.
- Challenge: `Move(a,c)` is **not** supported — `valid_move(a,c)` fails, so Prolog
  independently rejects an action the Python planner might otherwise have proposed.
- `reduce_speed` succeeds via the chain `wet_road ⇒ slippery ⇒ reduce_speed`
  (fact ⇒ rule ⇒ rule ⇒ conclusion).

## Reflection Questions

1. **Why specify preconditions/effects before prompting the LLM?** It turns "write a
   planner" into a precise, checkable contract; without it the LLM must guess the intended
   semantics of each action and is more likely to produce a planner that "looks" right but
   applies actions in states where they are not logically valid.
2. **Example error from missing a precondition check:** the planner could allow
   `Drop(Package,C)` even when the robot is not holding the package, producing an
   impossible plan such as delivering a package the robot never picked up.
3. **Why "looks reasonable" ≠ valid:** a plan can name real actions in a plausible order
   while silently violating a precondition at some step (e.g. moving before picking up);
   only checking `S ⊨ Preconditions(a)` at every step establishes actual validity.
4. **LLM contribution:** translating the specification (state representation, precondition
   checking, effect application, BFS frontier, path reconstruction) into working Python
   boilerplate quickly.
5. **What had to be verified independently:** that Test B genuinely reports failure instead
   of looping, that Test A's plan matches the hand-derived plan exactly, and that Test C's
   goal is not conflated with the real delivery goal — none of this can be assumed just
   because the code runs.
6. **Where logic is used:** in the `applicable()` check (precondition satisfaction) and in
   `apply()` (effect-based state update) — this is the propositional-logic layer underneath
   every transition.
7. **Relation to search:** planning here is literally graph search (BFS) over a state space
   whose edges are logically defined by action preconditions/effects, directly extending the
   blind-search algorithms from the previous module to a symbolic, logic-based state space.
