:- module(ic_parser,
	 [translate_ics/2,
	  translate_ics_files/2,
      download_ics/1,
	  parse_ics/2,
	  parse_ics_string/2,
	  add_var/2]).

:- use_module(library(lists),
	     [append/3,
	      member/2]).
/*:- use_module(library(system),
	     [system/1]).
*/
:- use_module(parser_utils).
:- use_module(debug).

%:- use_module(ruleml_parser).

/*
translate_ics_files(FileList,OutFile):-
	write_debug('translate_ics_files: init...'),
	write_debug('Parsing ICS: init...'),
	merge_files(FileList,'_temp_ics_file_.txt'),
	translate_ics('_temp_ics_file_.txt',OutFile),!,
	% This cut is important, as in order to signal translation errors, 
    % choice points are left open 
	write_debug('ICS successfully translated and written to:'),
	write_debug(OutFile),
	write_debug('Parsing ICS: end.'). 
*/

download_ics(URLstring):-
    atom_codes(URL,URLstring),
    translate_ics_files([URL],'./temp.pl'),
    open('./temp.pl',read,Stream),
    call_terms(Stream),
    close(Stream).

call_terms(Stream):-
    read(Stream,Term),
    (Term=end_of_file -> true
        ; (Term = ics(Body,Head)-> call(user:ic(Body,Head)) % invokes all atoms ic/2 in the file
            ; true),
          call_terms(Stream)
    ).

translate_ics_files(FileList,OutFile):-
	open(OutFile,write,Stream),
	write(Stream,':-module(ics,[ics/2]).'),nl(Stream),nl(Stream),
	translate_ics_list(FileList,Stream),
	close(Stream).

translate_ics_list([],_).
translate_ics_list([InFile|FileList],Stream):-
    write_debug('Parsing file '), write_debug(InFile),
    translate_ics_opened(InFile,Stream),!,
    write_debug(' --> OK'), nl,
    translate_ics_list(FileList,Stream).

merge_files(FileList,OutFile):-
%	write('About to open outfile'),nl,
	open(OutFile,write,Stream),
%	write('outfile opened'),nl, write(FileList), nl,
	FileList=[H|T],write(H),nl,write(T),nl,
	merge_files_to_stream(FileList,Stream),
	close(Stream).

merge_files_to_stream([],_):-write('empty list'),nl.
merge_files_to_stream([File|MoreFiles],OutStream):-	
	read_file_to_string(File,String),
	write_string_to_stream(String,OutStream),
	merge_files_to_stream(MoreFiles,OutStream).

write_string_to_stream([],_).
write_string_to_stream([Code|MoreCodes],Stream):-
	put_code(Stream,Code),
	write_string_to_stream(MoreCodes,Stream).

translate_ics(InFile,OutFile):-
	open(OutFile,write,Stream),
	write(Stream,':-module(ics,[ics/2]).'),nl(Stream),nl(Stream),
    translate_ics_opened(InFile,Stream),
	close(Stream).

% Assumes the outfile is already open
translate_ics_opened(InFile,Stream):-
	% If the XML succeeds, OK, otherwise try to parse as normal
	ruleml_parse_file(InFile,ICSR,Error),
	(Error = no_ruleml
	   ->  parse_ics(InFile,ICS),
	       write_ics_to_stream(ICS,Stream)
	   ;   write_ics_to_stream(ICSR,Stream)
    ).

parse_ics(FileName,ICList):-
	read_file_to_string(FileName,FileString),
	phrase(elementList(FileString2),FileString),
	drop_whites(FileString2, NoWhitesString),
	phrase(ic_list(ICList,1), NoWhitesString).

parse_ics_string(S,ICList):-
	string_codes(S,String),
	phrase(elementList(FileString2),String),
	drop_whites(FileString2, NoWhitesString),
	phrase(ic_list(ICList,1), NoWhitesString).



%----------------------------------------------------------
% ICS DCG
%----------------------------------------------------------

ic_list([],_) -->
	[].
ic_list([IC|MoreICs],N) -->
	ic(IC),
	!,
	{N1 is N+1},
	ic_list(MoreICs,N1).
ic_list([_|_],N) -->
    {write('Error in IC number '), 
    write(N), write(' ***'), nl, fail}.

ic(rule(((Head,_):-(Body,_)),0,P)) -->
        number(P),
	"::",
	body(Body),
	impl_symbol,!,
	head2(Head).
ic(_) -->
    {nl, write('*** Error in Body or could not find implication symbol: '), nl, fail}.

body([BodyAtom|MoreAtoms]) -->
	abducible(BodyAtom),!,
	body_tail(MoreAtoms).
body([BodyAtom|MoreAtoms]) -->
	event(BodyAtom),!,
	body_tail(MoreAtoms).
body([BodyAtom|MoreAtoms]) -->
	body_atom(BodyAtom),!,
	body_tail(MoreAtoms).
body([true]) -->"true".
body(_) -->
    {nl, write('*** Body must begin with event or abducible.'), nl, fail}.

body_tail([BodyAtom|MoreBodyAtoms]) -->
	and_symbol,
	body_atom(BodyAtom),
	!,
	body_tail(MoreBodyAtoms).
body_tail([]) -->
	[].
body_tail(_) -->
    comma,
    {nl, write('*** Error in body conjunct: comma instead of /\\ symbol?'), fail}.

body_atom(BodyAtom) -->
	abducible(BodyAtom).
body_atom(BodyAtom) -->
	event(BodyAtom).
body_atom(BodyAtom) -->
	atom(BodyAtom).
body_atom(BodyAtom) -->
	relat(BodyAtom).

relat(Relation) -->
	clp_relation(Relation),
	!.
relat(Relation) -->
	unify_relation(Relation).

unify_relation(Relation) -->
	term(Term1),
	unify_operator(Operator),
	term(Term2),
	{Relation=..[Operator,Term1,Term2]}.

clp_relation(Relation) -->
	expression(Expression1),
	clp_relop(Relop),
	expression(Expression2),
	{Relation=..[Relop,Expression1,Expression2]}.



expression(Expression) -->
	operand(Operand1),
	clp_operator(CLPOperator),
	operand(Operand2),
	{Expression=..[CLPOperator,Operand1,Operand2]}.
expression(Expression) -->
	operand(Expression).
expression(Expression) -->
	term(Expression).

is_constraint(C):-
	C=..[R|_],
	member(R,[=,<>,>=,>,=<,<,::]).

clp_relop(=) -->
	"==",
	!.
clp_relop(<>) -->
	"<>",
	!.
clp_relop(>=) -->
	">=",
	!.
clp_relop(>) -->
	">",
	!.
clp_relop(=<) -->
	"<=",
	!.
clp_relop(<) -->
	"<".
clp_relop(::) -->
	"::".

clp_operator(+) -->
	"+".
clp_operator(-) -->
	"-".
clp_operator(*) -->
	"*".
clp_operator(/) -->
	"/".


unify_operator(unif) -->
	"=".
unify_operator(not_unif) -->
	"!=".

operand(Number) -->
	number(Number).
operand(Variable) -->
	variable(Variable).

head2(Head) -->
    head1(Head),
	full_stop,!.
head2(_) -->
    {nl, write('*** Error in Head or could not find full stop: '), fail}.

head1([])-->"false",!.
head1(Head)-->head(Head).

head([Disjunct|MoreDisjuncts]) -->
	disjunct(Disjunct),
	head_tail(MoreDisjuncts).


disjunct_1(Disjunct1) -->
	disjunct(Disjunct),
	{constraints_before(Disjunct,Disjunct1)}.

constraints_before(L1,L2):-
	divide_constraints_from_abducibles(L1,Constraints,Abducibles),
	append(Constraints,Abducibles,L2).

divide_constraints_from_abducibles([],[],[]).
divide_constraints_from_abducibles([H|T],[H|T1],L2):-
	is_constraint(H),
	!,
	divide_constraints_from_abducibles(T,T1,L2).
divide_constraints_from_abducibles([H|T],L1,[H|T2]):-
	divide_constraints_from_abducibles(T,L1,T2).



head_tail([Disjunct|MoreDisjuncts]) -->
	or_symbol,
	disjunct(Disjunct),
	!,
	head_tail(MoreDisjuncts).
head_tail([]) -->
	[].

disjunct((-,[Conjunct|MoreConjuncts],_)) -->
	en(Conjunct),
	disjunct_tail(MoreConjuncts),!,
	closing_parenthesis.

disjunct((+,[Conjunct|MoreConjuncts],_)) -->
	content(Conjunct),
	disjunct_tail(MoreConjuncts).


en(Content) -->
	"not",
	opening_parenthesis,
	content(Content).


/*disjunct([Conjunct|MoreConjuncts]) --> %% Added MarcoG: let's extend the syntax!!!!
	atom(Conjunct),
	{writeln_debug(''), writeln_debug('*** Warning: atom in head ***')},
	disjunct_tail(MoreConjuncts).
disjunct([Conjunct|MoreConjuncts]) --> %% Added MarcoG: let's extend the syntax!!!!
	event(Conjunct),
	{writeln_debug(''), writeln_debug('*** Warning: H in head ***'), nl},
	disjunct_tail(MoreConjuncts).
disjunct([Conjunct|MoreConjuncts]) --> %% Added MarcoG: let's extend the syntax!!!!
	relat(Conjunct),
	disjunct_tail(MoreConjuncts).
*/
disjunct_tail([Conjunct|MoreConjuncts]) -->
	and_symbol,
	head_conjunct(Conjunct),
	!,
	disjunct_tail(MoreConjuncts).
disjunct_tail([]) -->
	[].
disjunct_tail(_) -->
    comma,
    {nl, write('*** Error in conjunct: comma instead of /\\ symbol?'), fail}.

head_conjunct(Conjunct) -->
	abducible(Conjunct).
head_conjunct(Conjunct) -->
	atom(Conjunct).
head_conjunct(Conjunct) -->
	relat(Conjunct).

atom(Atom) -->
	funct(Functor),
	opening_parenthesis,
	!,
	term_list(Arguments),
	closing_parenthesis,
	{Atom=..[Functor|Arguments]}.



abducible(Abducible) -->
	abducible_functor(Functor),
	opening_parenthesis,
	content(Content),
	closing_parenthesis,
	{Abducible=..[Functor,Content]}.

event(Event) -->
	event_functor(hap),
	opening_parenthesis,
	content(Content),
	comma,
	time(Time),
	closing_parenthesis,
	{Event=..[hap,Content,Time]}.

event((\+ Event)) -->
	event_functor(noth),
	opening_parenthesis,
	content(Content),
	comma,
	time(Time),
	closing_parenthesis,
	{Event=..[hap,Content,Time]}.


abducible_functor(e) -->
	"E".
abducible_functor(en) -->
	"EN".
abducible_functor(note) -->
	"!E".
abducible_functor(noten) -->
	"!EN".
abducible_functor(abd) -->
	"ABD".
		  
event_functor(hap) -->
	"hap".
event_functor(noth) -->
	"!hap".


content(Content) -->
	term(Content).





			 

	




or_symbol -->
	"\\/".
and_symbol -->
	"/\\".	

impl_symbol -->
	"--->".






write_ics_to_file(FileName,ICList):-
	open(FileName,write,Stream),
	write_ics_to_stream(ICList,Stream),
	close(Stream).

write_ics_to_stream([],_).
write_ics_to_stream([IC|MoreICs],Stream):-
	write_ic_to_stream(IC,Stream),
	write_ics_to_stream(MoreICs,Stream).

write_ic_to_stream(ic(Body,Head),Stream):-
	write(Stream,'ics('),
	write(Stream,Body),write(Stream,','),
	nl(Stream),
	spaces(Stream),
	write(Stream,'['),
	write_head_to_stream(Head,Stream),
	write(Stream,']).'),
	nl(Stream),
	nl(Stream).

write_head_to_stream([Disjunct],Stream):-
	write(Stream,Disjunct).
write_head_to_stream([Disjunct1,Disjunct2|MoreDisjuncts],Stream):-
	write(Stream,Disjunct1),
	write(Stream,','),
	nl(Stream),
	spaces(Stream),
	write_head_to_stream([Disjunct2|MoreDisjuncts],Stream).
	
		     
spaces(Stream):-
	write(Stream,'        ').


	

add_var([],[]).

add_var([rule(C,S,P)|T],[rule(CV,S,P)|TV]):-
  add_var_ic(C,CV),
  add_var(T,TV).

add_var_ic(((H,HL):-(B,BL)),((HV,HL):-(BV,BL))):-
  collect_vars([H,B],[],VA),
  length(VA,N),
  length(V,N),
  replace_vars([H,B],[HV,BV],VA,V).

replace_vars([],[],_VA,_V).
  
replace_vars([A|T],[Var|T1],VA,V):-
  atomic(A),
  nth1(N, VA, A),!,
  nth1(N,V,Var),
  replace_vars(T,T1,VA,V).

replace_vars([A|T],[A|T1],VA,V):-
  (atomic(A);var(A)),!,
  replace_vars(T,T1,VA,V).

replace_vars([A|T],[AV|T1],VA,V):-
  A=..[F|Args],
  replace_vars(Args,ArgsV,VA,V),
  AV=..[F|ArgsV],
  replace_vars(T,T1,VA,V).



collect_vars([],V,V).
  
collect_vars([A|T],VIn,VOut):-
  atomic(A),
  \+number(A),
  A\=[],
  atom_codes(A,AC),
  is_var(AC),!,
  (member(A,VIn)->
    V1=VIn
  ;
    V1=[A|VIn]
  ),
  collect_vars(T,V1,VOut).

collect_vars([A|T],VIn,VOut):-
  (atomic(A);var(A);number(A)),!,
  collect_vars(T,VIn,VOut).

collect_vars([A|T],VIn,VOut):-
  A=..[_F|Args],
  collect_vars(Args,VIn,V1),
  collect_vars(T,V1,VOut).

is_var([H|_T]):-
  (H>=65,
  H=<90);
  H=95.