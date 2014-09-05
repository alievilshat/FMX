:- module(singleproduct, [singleproduct/1]).

singleproduct([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) :- !.

singleproduct([AE, AF, AG, AH, BE, BF, BG, BH, CE, CF, CG, CH, DE, DF, DG, DH]) :-
	M = [[AE, AF, AG, AH], [BE, BF, BG, BH], [CE, CF, CG, CH], [DE, DF, DG, DH]],
	member(E, M), E \= [0,0,0,0],
	negate(E, N), !,
	check(M, E, N).

check([], _, _) :- !.
check([[0,0,0,0]|T], E, N) :- !, check(T, E, N).
check([E|T], E, N) :- !, check(T, E, N).
check([N|T], E, N) :- !, check(T, E, N).

negate([A,B,C,D], [NA,NB,NC,ND]) :-
	NA is -A,
	NB is -B,
	NC is -C,
	ND is -D.