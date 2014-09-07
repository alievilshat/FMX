generate_composition(MAX, [I1, I2, I3, I4, I5, I6, I7]) :-
	C = MAX,
	between(0, C, I1),
	between(I1, C, I2),
	between(I2, C, I3),
	between(I3, C, I4),
	between(I4, C, I5),
	between(I5, C, I6),
	between(I6, C, I7).

generate_sequence(0, _, []).
generate_sequence(LEN, A, [E|T]) :- LEN > 0,
	L is LEN - 1,
	member(E, A),
	generate_sequence(L, A, T).