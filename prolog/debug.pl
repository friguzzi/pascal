:-module(debug,
     [write_debug/1,writeln_debug/1]).

:- use_module(sciff_options).



writeln_debug(Message) :-
	get_option(sciff_debug, on),
	!,
	write(Message), nl.
writeln_debug(_).

write_debug(Message) :-
	get_option(sciff_debug, on),
	!,
	write(Message).
write_debug(_).
