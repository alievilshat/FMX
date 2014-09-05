:-module(test, [generate_polinoms/1]).

:- use_module(main).

:- dynamic t/1.
set_t(X) :- retractall(t(_)), assert(t(X)).
inc_t :- t(X), Y is X + 1, set_t(Y).

generate_polinoms(F) :-
	set_t(0),
    open(F, write, OS),
    (
    	run(X),
    	t(C),
    	write(OS, 'p('), write(OS, C), write(OS, ', '),
        write(OS, X), write(OS, ').'), nl(OS),
        inc_t,
        false
        ;
        close(OS)
    ).