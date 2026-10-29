% ============================================================
% planner.pl -- Prolog verifier for the warehouse planning lab
% ============================================================

% --- Task 6: warehouse connectivity facts ---
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

% Example queries:
% ?- can_move(a,b).   -> true.
% ?- can_move(a,c).   -> false. (a and c are not directly connected;
%                                 this rule does not chain through b)

% --- Task 7: checking a proposed Python plan ---
valid_move(X,Y) :-
    connected(X,Y).

% Proposed plan: Move(a,b), Move(b,c)
% ?- valid_move(a,b).   -> true.
% ?- valid_move(b,c).   -> true.
% ?- valid_move(a,c).   -> false.  (no direct connection a-c)

% Challenge: is Move(a,c) supported by the warehouse knowledge?
% ?- valid_move(a,c).   -> false. Prolog independently rejects the action,
%                          confirming (or refuting) whatever a Python planner
%                          proposed, without trusting the planner's own claim.

% --- Task 8: chained rule-based inference ---
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% ?- reduce_speed.   -> true.
% Reasoning chain:  wet_road (fact)
%                => slippery      (rule: wet_road -> slippery)
%                => reduce_speed  (rule: slippery -> reduce_speed)
% i.e.  wet_road ⇒ slippery ⇒ reduce_speed.
