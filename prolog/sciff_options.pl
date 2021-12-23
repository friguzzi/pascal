:-module(sciff_options,
     [get_option/2,
      set_option/2,
      sciff_option/2,
      set_options/1,
      show_options/0]).


:- dynamic(sciff_option/2).




%----------------------------------------------------------
% ALL OPTIONS
%----------------------------------------------------------
sciff_option(fulfiller,off).

sciff_option(fdet,off).
%sciff_option(fdet,on).

sciff_option(seq_act,off).

sciff_option(factoring,off).

sciff_option(sciff_debug, on).

sciff_option(violation_causes_failure, yes).

sciff_option(graphviz, off).

sciff_option(allow_events_not_expected, yes).



get_option(O,V):-
    sciff_option(O,V).

set_option(Option,Value):-
    retract(sciff_option(Option,_)),
    assert(sciff_option(Option,Value)).

show_options :-
	findall(sciff_option(Option, Value), sciff_option(Option, Value), ListOption),
	print_options(ListOption).
print_options([]) :- nl, nl.
print_options([sciff_option(Option, Value)| T]) :-
	write(Option),
	write(' is '),
	write(Value),
	write('.'), nl,
	print_options(T).

set_options([]).
set_options([[O,V]|T]):-
    set_option(O,V),
    set_options(T).
	

