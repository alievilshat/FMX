:-[generator].
:-[polinom].

run(T) :-
	generate_composition(3200, T),
	output(T),
	map_p(T, C),
	%           [ae, af, ag, ah, be, bf, bg, bh, ce, cf, cg, ch, de, df, dg, dh]
	validate(C, [1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
	validate(C, [0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0]),
	validate(C, [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0]),
	validate(C, [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1]).

validate(C, R) :-
	generate_sequence(7, [0,1], M),
	sum(M, C, R).

sum(_, [[],[],[],[],[],[],[]], []) :- !.
sum(M, [[H1|T1], [H2|T2], [H3|T3], [H4|T4], [H5|T5], [H6|T6], [H7|T7]], [R|TR]) :-
	subsum(M, [H1, H2, H3, H4, H5, H6, H7], R),
	sum(M, [T1, T2, T3, T4, T5, T6, T7], TR).

subsum([], [], 0).
subsum([M|MT], [H|T], R) :-
	subsum(MT, T, K),
	R is ((M*H) + K).

map_p([], []).
map_p([H|T], [R|TR]) :- p(H, R), !, map_p(T, TR).

output([I1, I2, I3, I4, I5, 3200, 3200]) :-
	%get_time(T),
	%current_output(S),
	%format_time(S, '%H:%M:%S    ', T),
	println([I1, I2, I3, I4, I5, 3200, 3200]), !.
output(_).

:- run(T).
