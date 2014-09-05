:- module(main, [run/1]).

:- set_prolog_flag(toplevel_print_options, [quoted(true), portray(true)]).

:- use_module(generator).
:- use_module(polinom).
:- use_module(singleproduct).

run(R) :-
	generate_sequence(16, [0, 1, -1], R),
	singleproduct(R).