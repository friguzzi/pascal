:- module(test_pascal,
  [test_pascal/0]).
:- use_module(library(plunit)).

test_pascal:-
    run_tests([triazine,bupa,triazinegd]).

:-use_module(library(cplint_test/cplint_test)).



:- begin_tests(triazine, []).

:-ensure_loaded('examples/triazine-d').
:- set_pascal(verbosity,0).

test(ip):-
  induce_pascal([train],_).

test(ip_tp):-
  induce_pascal([train],P),
  test_prob_pascal(P,[test],_LL,_Po,_N,_L).    

test(ip_t):-
  induce_pascal([train],P),
  test_pascal(P,[test],_LL,_AUCROC,_ROC,_AUCPR,_PR).

test(ipar):-
  induce_par_pascal([train],_R).

%test(ipar_f):-
%  pascal:induce_par_pascal_func([train],1,2,-2.0,2.0,-2.0,2.0,10,_).

%test(ipar_f_l):-
%  induce_par_pascal_func([train],1,2,10,_).

:- end_tests(triazine).

:- begin_tests(bupa, []).

:-ensure_loaded('examples/bupa-d').
:- set_pascal(verbosity,0).
test(i_bupa):-
  induce_pascal([train],P),test_pascal(P,[test],_LL,_AUCROC,_ROC,_AUCPR,_PR).


:- end_tests(bupa).
:- begin_tests(canc, []).


:-ensure_loaded('examples/canc-d').
:- set_pascal(verbosity,0).
test(i_canc):-
  induce_pascal([train],P),test_prob_pascal(P,[test],_Pos,_Neg,_LL,_L).


:- end_tests(canc).

:- begin_tests(triazinegd, []).

:-ensure_loaded('examples/triazine-dgd').
:- set_pascal(verbosity,0).
test(ip_gd):-
  induce_pascal([train],_).

test(ip_tp_gd):-
  induce_pascal([train],P),
  test_prob_pascal(P,[test],_LL,_Po,_N,_L).    

test(ip_t_gd):-
  induce_pascal([train],P),
  test_pascal(P,[test],_LL,_AUCROC,_ROC,_AUCPR,_PR).

test(ipar_gd):-
  induce_par_pascal([train],_R).

%test(ipar_f_gd):-
%  pascal:induce_par_pascal_func([train],1,2,-2.0,2.0,-2.0,2.0,10,_).

%test(ipar_f_l_gd):-
 % pascal:induce_par_pascal_func([train],1,2,10,_).

:- end_tests(triazinegd).