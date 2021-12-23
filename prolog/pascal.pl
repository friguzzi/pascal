
:-module(pascal,[set_pascal/2,setting_pascal/2,
  induce_pascal/2,op(500,fx,#),op(500,fx,'-#'),
  induce_par_pascal/2,
  %evaluate/3, progress/8,
  test_pascal/7,
  test_prob_pascal/6
%  objective_func/9,
%  induce_pascal_func/9,
%  induce_pascal_func/5
  %induce_par_pascal_func/9,
 % induce_par_pascal_func/5
/*  ,induce_par/2,test/7,
  induce_par_func/9,
  induce_par_func/5,
  objective_func/8,
  list2or/2,list2and/2,
  sample/4,learn_params/6,
  op(500,fx,#),op(500,fx,'-#'),
  test_prob/6,rules2terms/2,
  process_clauses/6,
  generate_clauses/6,
  generate_clauses_bg/2,
  generate_body/3,
  make_dynamic/1,
  extract_fancy_vars/2,
  linked_clause/3,
  banned_clause/3,
  take_var_args/3,
  remove_duplicates/2,
  exctract_type_vars/3,
  delete_one/3,
  get_next_rule_number/2,
  member_eq/2,
  retract_all/1,assert_all/3,
  write2/2,write3/2,format2/3,format3/3,
  write_rules2/3,write_rules3/3,
  nl2/1,nl3/1,
  forward/3,backward/4,write_net/3,write_eval_net/3,update_weights/3,
  onec/2,zeroc/2,andc/4,bdd_notc/3,
  orc/3,
  ret_probc/3,equalityc/4,
  or_list/3
  */
  ]).
:-use_module(library(system)).
:-use_module(library(lists)).
:-use_module(library(lbfgs)).
:-use_module(library(random)).
:-use_module(library(auc)).
:-use_module(ic_parser).

:- thread_local  pascal_input_mod/1,p/2.

:- meta_predicate induce_pascal(:,-).
:- meta_predicate induce_par_pascal(:,-).
:- meta_predicate set_pascal(:,+).
:- meta_predicate setting_pascal(:,-).
:- meta_predicate test_pascal(:,+,-,-,-,-,-).
:- meta_predicate test_prob_pascal(:,+,-,-,-,-).
:- meta_predicate objective_func(:,-,-,-,-,-,-,-,-).
:- meta_predicate induce_pascal_func(:,-,-,-,-,-,-,-,-).
:- meta_predicate induce_pascal_func(:,-,-,-,-).
:- meta_predicate induce_par_pascal_func(:,-,-,-,-,-,-,-,-).
:- meta_predicate induce_par_pascal_func(:,-,-,-,-).


:- multifile sandbox:safe_meta/2.

sandbox:safe_meta(pascal:induce_par_pascal(_,_) ,[]).
sandbox:safe_meta(pascal:induce_pascal(_,_), []).
sandbox:safe_meta(pascal:test_prob_pascal(_,_,_,_,_,_), []).
sandbox:safe_meta(pascal:test_pascal(_,_,_,_,_,_,_), []).
sandbox:safe_meta(pascal:set_pascal(_,_), []).
sandbox:safe_meta(pascal:setting_pascal(_,_), []).

% NOTE: resi dinamici per poter fare retract nel caso non si usi la bottom_clause



/* allowed values: auto, keys(pred) where pred is the predicate indicating the class (e.g. bongard) */
default_setting_pascal(examples,auto).

default_setting_pascal(der_depth,20).
default_setting_pascal(beamsize,10).
default_setting_pascal(verbosity,3).
default_setting_pascal(significance_level,0).
default_setting_pascal(pruning,true).
/*allowed values 0.995 / 0.99 / 0.975 / 0.95 / 0.90 / 0.75 / 0.0 */
default_setting_pascal(min_coverage,3).
default_setting_pascal(min_accuracy,0.75).
default_setting_pascal(max_nodes,10). %max num iterazioni findBestIC
default_setting_pascal(heur,acc). /* allowed values: acc, laplace */
default_setting_pascal(optimal,no). /* allowed values: yes, no */
default_setting_pascal(max_length,4).

/* PASCAL*/
default_setting_pascal(max_refinements, none).
default_setting_pascal(num_samples,50).
default_setting_pascal(rand_seed, 1234).
default_setting_pascal(max_initial_weight,0.1).
default_setting_pascal(lookahead, no).
% default_setting_pascal for approximate parameter learning
default_setting_pascal(approx_pl,none).
default_setting_pascal(max_rules,10).
/* default_setting_pascal(max_lengths[Body,Disjucts,LitIn+,LitIn-]). */
default_setting_pascal(max_lengths,[1,1,1,0]).
default_setting_pascal(epsilon_em,0.0001).
default_setting_pascal(epsilon_em_fraction,0.00001).
default_setting_pascal(logzero,log(0.01)).
default_setting_pascal(zero,0.0001).
default_setting_pascal(minus_infinity,-1.0e20).
default_setting_pascal(iter,-1).
default_setting_pascal(random_restarts_number,1).
default_setting_pascal(d,1).
default_setting_pascal(depth_bound,false).  %if true, it limits the derivation of the example to the value of 'depth'
default_setting_pascal(depth,2).
% selezionare se si vuole bottom clause o no
default_setting_pascal(bottom_clause,no).
default_setting_pascal(ex_bottom,3). %n. bottom c. da generare
default_setting_pascal(eps,0.0001).
default_setting_pascal(eps_f,0.00001).

default_setting_pascal(fixed_parameters,no).

default_setting_pascal(default_parameters,0).
% allowed values: gradient_descent, lbfgs
default_setting_pascal(learning_algorithm,lbfgs).
% allowed values: fixed(value), decay(eta_0,eta_tau,tau)
default_setting_pascal(learning_rate,fixed(0.01)).
default_setting_pascal(gd_iter,1000).
default_setting_pascal(regularizing_constant,5).
default_setting_pascal(regularization,2).
% allowed values: 1, 2

/**
 * test_pascal(:T:probabilistic_program,+TestFolds:list_of_atoms,-LL:float,-AUCROC:float,-ROC:dict,-AUCPR:float,-PR:dict) is det
 *
 * The predicate takes as input in T a probabilistic constraint logic theory,
 * tests T on the folds indicated in TestFolds and returns the
 * log likelihood of the test examples in LL, the area under the Receiver
 * Operating Characteristic curve in AUCROC, a dict containing the points
 * of the ROC curve in ROC, the area under the Precision Recall curve in AUCPR
 * and a dict containing the points of the PR curve in PR
 */
test_pascal(P,TestFolds,LL,AUCROC,ROC,AUCPR,PR):-
  test_prob_pascal(P,TestFolds,_NPos,_NNeg,LL,LG),
  compute_areas_diagrams(LG,AUCROC,ROC,AUCPR,PR).

/**
 * test_prob_pascal(:T:probabilistic_program,+TestFolds:list_of_atoms,-NPos:int,-NNeg:int,-LL:float,-Results:list) is det
 *
 * The predicate takes as input in T a probabilistic constraint logic theory,
 * tests T on the folds indicated in TestFolds and returns
 * the number of positive examples in NPos, the number of negative examples
 * in NNeg, the log likelihood in LL
 * and in Results a list containing the probabilistic result for each query contained in TestFolds.
 */
test_prob_pascal(M:P,TestFolds,NPos,NNeg,CLL,Results) :-
  write2(M,'Testing\n'),
  findall(Exs,(member(F,TestFolds),M:fold(F,Exs)),L),
  append(L,TE),
  test_no_area(TE,P,M,NPos,NNeg,CLL,Results).

test_no_area(TestSet,P0,M,NPos,NNeg,CLL,Results):-
  rule_to_int(P0,P),
  test_ex(TestSet,P,M,Results,0,NPos,0,NNeg,0,CLL).


test_ex([],_P,_M,[],Pos,Pos,Neg,Neg,CLL,CLL).

test_ex([HT|TT],P,M,[Prob-Ex|TE],Pos0,Pos,Neg0,Neg,CLL0,CLL):-
  convert_prob(P,Pr1),
  %  gen_par(0,NC,Par0),
  length(P,N),
  gen_initial_counts(N,MIP0), %MIP0=vettore di N zeri
  test_theory_pos_prob([HT],M,Pr1,MIP0,MIP), %MIP=vettore di N zeri
  foldl(compute_prob,P,MIP,0,LL),
  (is_pos(HT,M)->
    Pos2 is Pos0+1,
    Neg2 = Neg0,
    Ex = HT,
    Prob is exp(LL),
    CLL2 is CLL0+LL
  ;
    Pos2 = Pos0,
    Neg2 is Neg0+1,
    Ex = (\+ HT),
    Prob is exp(LL),
    (Prob=:=1.0->
      M:local_setting(logzero,LZ),
      CLL2 is CLL0+LZ
    ;
      CLL2 is CLL0+log(1-Prob)
    )
  ),
  test_ex(TT,P,M,TE,Pos2,Pos,Neg2,Neg,CLL2,CLL).

is_pos(M,Mod):-
  (Mod:local_setting(examples,keys(P))->
    AtomP=..[P,M,pos],
    Atom=..[P,M],
    (current_predicate(Mod:P/1)->
      (current_predicate(Mod:P/2)->
        (Mod:AtomP;Mod:Atom)
      ;
        Mod:Atom
      )
    ;
      Mod:AtomP
    )
 ;
    AtomP=..[pos,M],
    Mod:AtomP
  ).


compute_prob(rule(_,_,P),N,LL0,LL):-
  LL is LL0+N*log(1-P).

/**
 * induce_pascal(:TrainFolds:list_of_atoms,-T:probabilistic_theory) is det
 *
 * The predicate performs structure learning using the folds indicated in
 * TrainFolds for training.
 * It returns in T the learned probabilistic constraint logic theory.
 */
induce_pascal(M:Folds,P):-
  induce_int(Folds,M,_DB,Program),
  rule_to_ext(Program,P).


/**
 * induce_par_pascal(:TrainFolds:list_of_atoms,-T:probabilistic_program) is det
 *
 * The predicate learns the parameters of the theory stored in begin_in/end_in
 * section of the input file using the folds indicated in TrainFolds for training.
 * It returns in T the input theory with the updated parameters.
 */
induce_par_pascal(M:Folds,P):-
  induce_par_int(Folds,M,_DB,Program),
  rule_to_ext(Program,P).

    
  
induce_par_int(Folds,M,DB,Program):-
  M:in(Program00),
  rule_to_int(Program00,Program0),
  statistics(runtime,[_,_]),
  (M:bg(BG)->
    maplist(process,BG,BGP),
    assert_all(BGP,M,BGRefs)
  ;
    BGRefs=[]
  ),
  findall(Exs,(member(F,Folds),M:fold(F,Exs)),Le),
  append(Le,DB),
  get_pos_neg(DB,M,Pos,Neg),
  length(Pos,NP),
  length(Neg,NN),
  format2(M,"/* Inizio l'apprendimento dei pesi, N pos ~d N neg ~d */~n",[NP,NN]),
  learn_param(Program0,M,Pos,Neg,Program,LL),
  format2(M,"/* Log likelihood ~f~n*/~n",[LL]),
  write_rules2(M,Program),
  retract_all(BGRefs).

rule_to_ext(P0,P):-
  maplist(to_ext,P0,P).

rule_to_int(P0,P):-
  maplist(to_int,P0,P).

to_ext(rule(_,((H,_):-(B,_BL)),P),rule((H:-B),P)).

to_int(rule((H:-B),P),rule(r,((H,[]):-(B,[])),P)).

induce_int(Folds,M,DB,Program):-    
  statistics(runtime,[_,_]),
%  load_bg(FileBG),
%    load_models(FileKB,HB,ModulesList),
  findall(Exs,(member(F,Folds),M:fold(F,Exs)),Le),
  append(Le,DB),
  (M:bg(BG)->
    maplist(process,BG,BGP),
    assert_all(BGP,M,BGRefs)
  ;
    BGRefs=[]
  ),
  get_pos_neg(DB,M,Pos,Neg),
  length(Pos,NP),
  length(Neg,NN),
  format2(M,"/* Learning start, N pos ~d N neg ~d */~n",[NP,NN]),
  induce(Pos,Neg,M,Program,LL),
  %remove_red(Pos,ProgramRed,[],Program0),
  /*	seleziona max rules regole 
  setting(max_rules,MR),
  insert_max_rules(Program0,MR,CL3), % inserisce in CL3 il minimo tra MR e N0 regole in CL3
  length(CL3,LCL3),
  %format("lunghezza programma max_rules  = ~d",[LCL3]),
    insert_starting_prob(CL3, Program1),*/
%  insert_starting_prob(Program0,Program1),
 % learn_param(Program1,M,Pos,Neg,Program,LL),
    statistics(runtime,[_,T]),
    T1 is T /1000,
    findall(setting(A,B),M:local_setting(A,B),L),
  %  length(NegRem,NR),
  length(Program,N1),
    %findall(template(HeadType,BodyType,Name,Head,Body),template(HeadType,BodyType,Name,Head,Body),LT),
    %numbervars(LT,0,_),
  M:local_setting(optimal,Opt),
  format2(M,"/* Learning time ~f seconds. */~N",[T1]),
  format2(M,"/* Number of rules ~d */~n",[N1]),
  format2(M,"/* ~p */~n~n",[L]),    
 % format("/* Negative examples remaining: ~d~n~p~n*/~n",[NR,NegRem]),   
  format2(M,"/* Language bias ~n~p~n*/~n",[optimal(Opt)]),
  format2(M,"/* Log likelihood ~f~n*/~n",[LL]),
  write_rules2(M,Program),
  retract_all(BGRefs).
%clear_kb(HB).

induce_pascal_func(M:Folds,XN,YN,XMin,XMax,YMin,YMax,Steps,POut):-
  induce_int(Folds,M,DB,Prog),
  rule_to_ext(Prog,POut),
  get_hist(M,Hist),
  obj_fun_hist_plot(DB,M,Prog,XN,YN,XMin,XMax,YMin,YMax,Steps,Hist).

induce_pascal_func(M:Folds,XN,YN,Steps,Prog):-
  induce_int(Folds,M,DB,ROut),
  rule_to_ext(ROut,Prog),
  get_hist(M,Hist),
  get_min_max_hist(Hist,XN,YN,XMin,XMax,YMin,YMax),
  obj_fun_hist_plot(DB,M,ROut,XN,YN,XMin,XMax,YMin,YMax,Steps,Hist).

induce_par_pascal_func(M:Folds,XN,YN,XMin,XMax,YMin,YMax,Steps,POut):-
  induce_par_int(Folds,M,DB,Prog),
  rule_to_ext(Prog,POut),
  get_hist(M,Hist),
  obj_fun_hist_plot(DB,M,Prog,XN,YN,XMin,XMax,YMin,YMax,Steps,Hist).

induce_par_pascal_func(M:Folds,XN,YN,Steps,Prog):-
  induce_par_int(Folds,M,DB,ROut),
  rule_to_ext(ROut,Prog),
  get_hist(M,Hist),
  get_min_max_hist(Hist,XN,YN,XMin,XMax,YMin,YMax),
  obj_fun_hist_plot(DB,M,ROut,XN,YN,XMin,XMax,YMin,YMax,Steps,Hist).

/**
 * objective_func(:TrainFolds:list_of_atoms,-P:probabilistic_program) is det
 *
 * The predicate learns the parameters of the program stored in the in/1 fact
 * of the input file using the folds indicated in TrainFolds for training.
 * It returns in P the input program with the updated parameters.
 */
objective_func(M:Folds,P0,XN,YN,XMin,XMax,YMin,YMax,Steps):-
  rule_to_int(P0,P),
  findall(Exs,(member(F,Folds),M:fold(F,Exs)),L),
  append(L,DB),
  statistics(walltime,[_,_]),
  obj_fun_plot(DB,M,P,XN,YN,XMin,XMax,YMin,YMax,Steps),
  statistics(walltime,[_,CT]),
  CTS is CT/1000,
%  format2(M,'/* EMBLEM Final score ~f~n',[Score]),
  format2(M,'Wall time ~f */~n',[CTS]),
  true.

  /**
 * obj_fun(+DB:list_of_atoms,+M:atom,+R0:probabilistic_program,-P:probabilistic_program,-Score:float) is det
 *
 * The predicate learns the parameters of the program R0 and returns
 * the updated program in R and the score in Score.
 * DB contains the list of interpretations ids and M the module where
 * the data is stored.
 */

obj_fun(DB,M,R0,XN,YN,XMin,XMax,YMin,YMax,Steps,X,Y,Z):-  %Parameter Learning
  compute_stats(DB,M,R0,NR,MIP,MI),
  draw(NR,MIP,MI,M,XN,YN,XMin,XMax,YMin,YMax,Steps,X,Y,Z).

compute_stats(DB,M,Program0,N,MIP,MI):-
  get_pos_neg(DB,M,Pos,Neg),
  convert_prob(Program0,Pr1),
  %  gen_par(0,NC,Par0),
  length(Program0,N),
  gen_initial_counts(N,MIP0), %MIP0=vettore di N zeri
  test_theory_pos_prob(Pos,M,Pr1,MIP0,MIP), %MIP=vettore di N zeri
  test_theory_neg_prob(Neg,M,Pr1,N,MI). %MI = [[1, 1, 1, 1, 1, 1, 1|...], [0, 0, 0, 0, 0, 0|...]
  

obj_fun_plot(DB,M,R0,XN,YN,XMin,XMax,YMin,YMax,Steps):-
  obj_fun(DB,M,R0,XN,YN,XMin,XMax,YMin,YMax,Steps,X,Y,Z),
  atomic_list_concat(['graph_obj_',XN,'_',YN,'.m'],File),
  open(File,write,S),
  write(S,'X = '),
  write_mat(S,X),
  write(S,'Y = '),
  write_mat(S,Y),
  write(S,'Z = '),
  write_mat(S,Z),
  write(S,"XP = 1 ./(1+exp(-X));
  YP= 1./(1+exp(-Y));"),
    write(S,"figure('Name','"),
  write(S,objective_func_w(XN,YN,XMin,XMax,YMin,YMax,Steps)),
  writeln(S,"','NumberTitle','off');"),
  writeln(S,'surf(X,Y,Z)'),
  write(S,"xlabel("),write(S,XN),writeln(S,");"),
  write(S,"ylabel("),write(S,YN),writeln(S,");"),
  writeln(S,"zlabel('-LogLik');"),
  write(S,"figure('Name','"),
  write(S,objective_func_p(XN,YN,XMin,XMax,YMin,YMax,Steps)),
  writeln(S,"','NumberTitle','off');"),
  writeln(S,'surf(XP,YP,Z)'),
  write(S,"xlabel("),write(S,XN),writeln(S,");"),
  write(S,"ylabel("),write(S,YN),writeln(S,");"),
  writeln(S,"zlabel('-LogLik');"),
  close(S).

obj_fun_hist_plot(DB,M,R0,XN,YN,XMin,XMax,YMin,YMax,Steps,Hist):-
  obj_fun(DB,M,R0,XN,YN,XMin,XMax,YMin,YMax,Steps,X,Y,Z),
  get_hist(Hist,XN,YN,XH,YH,ZH),
  atomic_list_concat(['graph_obj_traj_',XN,'_',YN,'.m'],File),
  open(File,write,S),
  write(S,'X = '),
  write_mat(S,X),
  write(S,'Y = '),
  write_mat(S,Y),
  write(S,'Z = '),
  write_mat(S,Z),
  write(S,'XH = ['),
  maplist(write_col(S),XH),
  writeln(S,'];'),
  write(S,'YH = ['),
  maplist(write_col(S),YH),
  writeln(S,'];'),
  write(S,'ZH = ['),
  maplist(write_col(S),ZH),
  writeln(S,'];'),
  write(S,"XP = 1 ./(1+exp(-X));
YP = 1 ./(1+exp(-Y));
XHP = 1 ./(1+exp(-XH));
YHP = 1 ./(1+exp(-YH));"),
  write(S,"figure('Name','"),
  write(S,objective_func_w(XN,YN,XMin,XMax,YMin,YMax,Steps)),
  writeln(S,"','NumberTitle','off');"),
  writeln(S,"plot3(XH,YH,ZH,'LineWidth',2)"),
  write(S,"xlabel("),write(S,XN),writeln(S,");"),
  write(S,"ylabel("),write(S,YN),writeln(S,");"),
  writeln(S,"zlabel('-LogLik');
hold on
surf(X,Y,Z)
hold off"),
write(S,"figure('Name','"),
write(S,objective_func_p(XN,YN,XMin,XMax,YMin,YMax,Steps)),
writeln(S,"','NumberTitle','off');"),
writeln(S,"plot3(XHP,YHP,ZH,'LineWidth',2)"),
write(S,"xlabel("),write(S,XN),writeln(S,");"),
write(S,"ylabel("),write(S,YN),writeln(S,");"),
writeln(S,"zlabel('-LogLik');
hold on
surf(XP,YP,Z)
hold off"),
close(S).



get_hist(M,Hist):-
  findall(p(A,B),M:p(A,B),Hist).
  
get_hist(Hist,XN,YN,XH,YH,ZH):-
  maplist(get_w(XN),Hist,XH),
  maplist(get_w(YN),Hist,YH),
  maplist(get_z,Hist,ZH).

get_min_max_hist(Hist,XN,YN,XMin,XMax,YMin,YMax):-
  get_hist(Hist,XN,YN,XH,YH,_ZH),
  min_list(XH,XMin),
  max_list(XH,XMax),
  min_list(YH,YMin),
  max_list(YH,YMax).

get_w(N,p(Ws,_),W):-
  arg(N,Ws,W).

get_z(p(_,Z),Z).

write_mat(S,M):-
  writeln(S,'['),
  append(M0,[ML],M),!,
  maplist(write_row(S),M0),
  maplist(write_col(S),ML),
  nl(S),
  writeln(S,']'),
  nl(S).

write_row(S,R):-
  maplist(write_col(S),R),
  writeln(S,';').

write_col(S,E):-
  write(S,E),
  write(S,' ').

draw(NR,MIP,MI,M,XN,YN,XMin,XMax,YMin,YMax,Steps,X,Y,Z):-
  XStep is (XMax-XMin)/Steps,
  YStep is (YMax-YMin)/Steps,
  cycle_X(NR,MIP,MI,M,XN,YN,XMin,XMax,YMin,YMax,XStep,YStep,X,Y,Z).

initial_w(NR,M,W):-
  M:local_setting(default_parameters,L),
  is_list(L),!,
  length(WA,NR),
  maplist(init_w_par,L,WA),
  W=..[w|WA].

initial_w(NR,M,W):-
  M:local_setting(default_parameters,V),
  length(WA,NR),
  maplist(init_w_par(V),WA),
  W=..[w|WA].

init_w_par(W,W).

cycle_X(NR,MIP,MI,M,XN,YN,X,XMax,YMin,YMax,_,YStep,[XL],[YL],[ZL]):-
  X>=XMax,!,
  initial_w(NR,M,W),
  setarg(XN,W,X),
  cycle_Y(W,MIP,MI,M,YN,X,YMin,YMax,YStep,XL,YL,ZL).

cycle_X(NR,MIP,MI,M,XN,YN,X,XMax,YMin,YMax,XStep,YStep,[XL|XT],[YL|YT],[ZL|ZT]):-
  initial_w(NR,M,W),
  setarg(XN,W,X),
  cycle_Y(W,MIP,MI,M,YN,X,YMin,YMax,YStep,XL,YL,ZL),
  X1 is X+XStep,
  cycle_X(NR,MIP,MI,M,XN,YN,X1,XMax,YMin,YMax,XStep,YStep,XT,YT,ZT).

cycle_Y(W,MIP,MI,M,YN,X,Y,YMax,_,[X],[Y],[Z]):-
  Y>=YMax,!,
  setarg(YN,W,Y),
  evaluate_w(MIP,MI,W,M,_LN,Z).

cycle_Y(W,MIP,MI,M,YN,X,Y,YMax,YStep,[X|XT],[Y|YT],[Z1|ZT]):-
  setarg(YN,W,Y),
  Y1 is Y+YStep,
  evaluate_w(MIP,MI,W,M,_LN,Z),
  Z1 is Z,
  cycle_Y(W,MIP,MI,M,YN,X,Y1,YMax,YStep,XT,YT,ZT).
    

evaluate_w(MIP,MI,W,M,LN,L):-
  compute_likelihood_pos_w(MIP,W,1,0,LP),
  compute_likelihood_neg_w(MI,W,LN), %MI lista di liste
  compute_likelihood(LN,LP,M,L). %LN=[6.931471805599453, 0.0, 6.931471805599453, 0.0, 0.0, 0.0, 0.0, 0.0|...]

compute_likelihood_neg_w([],_W,[]).

compute_likelihood_neg_w([HMI|TMI],W,[HLN|TLN]):- %HMI=lista
  compute_likelihood_pos_w(HMI,W,1,0,HLN),
  compute_likelihood_neg_w(TMI,W,TLN).

compute_likelihood_pos_w([],_,_,LP,LP).%LP=0 alla fine

compute_likelihood_pos_w([HMIP|TMIP],W,I,LP0,LP):- %primo arg=vettore di 0 MI
  arg(I,W,W0), 
  P is 1/(1+exp(-W0)), %P=sigma(w)=1/(1+exp(-W0))
  LP1 is LP0-log(1-P)*HMIP,
  I1 is I+1,
  compute_likelihood_pos_w(TMIP,W,I1,LP1,LP).

get_cl(([R],_),R).

insert_max_rules([],_,[]):-!.

insert_max_rules(_,0,[]):-!.

insert_max_rules([H|T],N,[H|T1]):-
	N1 is N - 1,
	insert_max_rules(T,N1,T1).

%input desiderato:learn_param([rule(bottom,  ([], []:-[], []), 0.5)], [71, 72, 73, 74, 75, 76, 89, 90|...], [70, 77, 78, 79, 80, 81, 82, 83|...], _G9197, _G9198)
%
%input in arrivo [rule(r,  ([], []:-[alkphos(_G860, 64)], []),  (heur(1), negcov(3), poscov(113), emc([275|...]), epnc([]))), rule(r....), ....]
insert_starting_prob([], []):-!.

insert_starting_prob([Rule|Pr0], [RuleProb|Pr1]):-
		%		Rule = rule(r, Clause, _Stat),
		Rule = (r, Clause, _Stat),
		RuleProb = rule(r, Clause, 1.0),
		insert_starting_prob(Pr0,Pr1).

generate_file_names(File,FileKB,FileBG,FileOut,FileL):-
        atom_concat(File,'.kb',FileKB),
        atom_concat(File,'.bg',FileBG),
        atom_concat(File,'.l',FileL),
        atom_concat(File,'.icl.out',FileOut).

divide_pos_neg([],Pos,Pos,Neg,Neg):-!.
    
divide_pos_neg([MH|MT],PosIn,PosOut,NegIn,NegOut):-
    (pos(MH)->
        PosOut=[MH|Pos],
        NegOut=Neg
    ;
        PosOut=Pos,
        NegOut=[MH|Neg]
    ),
    divide_pos_neg(MT,PosIn,Pos,NegIn,Neg).
        
%inizio doppio ciclo dpml
induce(Pos,Neg,M,Program,LL):-
    prior_prob(Pos,Neg,M,NP,NN),
	manage_modex(M), %asserisce i modeh/b
	%write('manage_modex'),
  M:local_setting(max_rules,MR),
  M:local_setting(minus_infinity,MInf),
	covering_loop1(Pos,Neg,M,NP,NN,MR,[],Program,MInf,LL).
	%Rin = [rule(null,null,(0,0,_,_,_))],  %formato regola
	%covering_loop(Pos,Neg,NegRem,NP,NN,0,NR,Rin,Program,S).
	

prior_prob(Pos,Neg,M,NP,NN):-
    total_number(Pos,M,0,NP),
    total_number(Neg,M,0,NN),
    assert(M:npt(NP)),
    assert(M:nnt(NN)).
    
total_number([],_,N,N):-!.

total_number([H|T],Mod,NIn,NOut):-
  (Mod:mult(H,M)->
    N1 is NIn+M
  ;
    N1 is NIn+1
  ),
  total_number(T,Mod,N1,NOut).

manage_modex(M):-
		get_modeb(M,BL0), %(BL=[(A,B)...] modeb(A,B)
		%flatten_multiple_var_modex(BL0,BL),
    get_const_types(M,Const),
		cycle_modex(BL0,M,'modeb',Const),
		get_modeh(M,HL0),
		%flatten_multiple_var_modex(HL0,HL),
	  cycle_modex(HL0,M,'modeh',Const).

get_modeb(M,BL):-
		  findall((R,B),M:modeb(R,B),BL).

get_modeh(M,BL):-
         findall((R,B),M:modeh(R,B),BL).

% per ogni mode controlla quante variabili sono segnate con -# e # e crea un nuovo mode(h/b)
% per ogni possibile istanziazione	
cycle_modex([],_,_,_).

cycle_modex([(A,P)|T],M,Type,Const):-
	P=..[F|Args],
	count_values(Args,NL),
	NL>0,!,
	ModeR=..[Type,A,P],
	retract(M:ModeR),!,
	(M:local_setting(bottom_clause,no) ->
        findall(Modex,create_new_modex_no_bc(Type,M,A,F,Args,Modex,Const),_)
      ;
        findall(Modex,create_new_modex(Type,M,A,F,Args,Modex,Const),_)
    ),
	cycle_modex(T,M,Type,Const).

cycle_modex([(A,P)|T],M,Type,Const):-
	ModeR=..[Type,A,P],
	retract(M:ModeR),!,
	assert(M:ModeR),
	%Modex=..[Type,A,P],
	%assert(Modex),
	cycle_modex(T,M,Type,Const).
	
% conta # e -#	
count_values([],0).

count_values([-#_|TP],N):-
	!,
	count_values(TP,N0),
	N is N0+1.
	
count_values([#_|TP],N):-
	!,
	count_values(TP,N0),
	N is N0+1.

count_values([_|TP],N):-
	count_values(TP,N).

% crea e asserisce nuovi mode(h/b)
% non funziona per predicati builtin come quelli aritmetici
create_new_modex(Type,M,A,F,Args,Modex,Const):-
	length(Args,N),
	length(Args1,N),
	P0=..[F|Args1],
  (builtin(P0)->
    P=P0
  ;
  	P=..[F,_|Args1]
  ),
	replace_values(Args1,Args,Args2,Const),
  call(M:P),
	NewP=..[F|Args2],
	Modex=..[Type,A,NewP],
  \+ call(M:Modex),
	assert(M:Modex).

% crea e asserisce nuovi mode(h/b)
% non funziona per predicati builtin come quelli aritmetici
create_new_modex_no_bc(Type,M,A,F,Args,Modex,Const):-
	length(Args,N),
	length(Args1,N),
	P0=..[F|Args1],
  (builtin(P0)->
    P=P0
  ;
  	P=..[F,_|Args1]
  ),
	replace_values_no_bc(Args1,Args,Args2,Const),
  call(M:P),
	NewP=..[F|Args2],
	Modex=..[Type,A,NewP],
  \+ call(M:Modex),
	assert(M:Modex).

	
replace_values([],[],[],_Const).

replace_values([H|T1],[# Type|T],[H|T2],Const):-
	!,
  member((Type,Con),Const),
  member(H,Con),
	replace_values(T1,T,T2,Const).

replace_values([H|T1],[-#_|T],[H|T2],Const):-!,
	replace_values(T1,T,T2,Const).

replace_values([H|T1],[+ Type|T],[+Type|T2],Const):-
	!,
  member((Type,Con),Const),
  member(H,Con),
	replace_values(T1,T,T2,Const).

replace_values([_H|T1],[- Type|T],[-Type|T2],Const):-
	!,
	replace_values(T1,T,T2,Const).

replace_values([H|T1],[H|T],[H|T2],Const):-
	replace_values(T1,T,T2,Const).


replace_values_no_bc([],[],[],_Const).

replace_values_no_bc([H|T1],[# Type|T],[H|T2],Const):-
	!,
  member((Type,Con),Const),
  member(H,Con),
	replace_values_no_bc(T1,T,T2,Const).

replace_values_no_bc([H|T1],[-# Type|T],[H|T2],Const):-
	!,
  member((Type,Con),Const),
  member(H,Con),
	replace_values_no_bc(T1,T,T2,Const).

replace_values_no_bc([H|T1],[+ Type|T],[+Type|T2],Const):-
	!,
  member((Type,Con),Const),
  member(H,Con),
	replace_values_no_bc(T1,T,T2,Const).

replace_values_no_bc([_H|T1],[- Type|T],[-Type|T2],Const):-
	!,
	replace_values_no_bc(T1,T,T2,Const).

replace_values_no_bc([H|T1],[H|T],[H|T2],Const):-
	replace_values_no_bc(T1,T,T2,Const).

get_const_types(M,Const):-
  findall(Types,get_types(M,Types),LT),
  append(LT,T),
  remove_duplicates(T,T1),
  get_constants(T1,M,Const).


get_types(M,Types):-
  M:modeh(_,At),
  At=..[_|Args],
  get_args(Args,Types).

get_types(M,Types):-
  M:modeb(_,At),
  At=..[_|Args],
  get_args(Args,Types).


get_args([],[]).

get_args([+H|T],[H|T1]):-!,
  get_args(T,T1).

get_args([-H|T],[H|T1]):-!,
  get_args(T,T1).

get_args([#H|T],[H|T1]):-!,
  get_args(T,T1).

get_args([-#H|T],[H|T1]):-!,
  get_args(T,T1).

get_args([_|T],T1):-
  get_args(T,T1).



get_constants([],_Mod,[]).

get_constants([Type|T],Mod,[(Type,Co)|C]):-
  find_pred_using_type(Type,Mod,LP),
  find_constants(LP,Mod,[],Co),
  get_constants(T,Mod,C).

find_pred_using_type(T,M,L):-
  (setof((P,Ar,A),pred_type(T,M,P,Ar,A),L)->
    true
  ;
    L=[]
  ).

pred_type(T,M,P,Ar,A):-
  M:modeh(_,S),
  S=..[P|Args],
  length(Args,Ar),
  scan_args(Args,T,1,A).

pred_type(T,M,P,Ar,A):-
  M:modeb(_,S),
  S=..[P|Args],
  length(Args,Ar),
  scan_args(Args,T,1,A).

scan_args([+T|_],T,A,A):-!.

scan_args([-T|_],T,A,A):-!.

scan_args([#T|_],T,A,A):-!.

scan_args([-#T|_],T,A,A):-!.

scan_args([_|Tail],T,A0,A):-
  A1 is A0+1,
  scan_args(Tail,T,A1,A).

find_constants([],_Mod,C,C).

find_constants([(P,Ar,_)|T],Mod,C0,C):-
  functor(G,P,Ar),
  builtin(G),!,
  find_constants(T,Mod,C0,C).

find_constants([(P,Ar,A)|T],Mod,C0,C):-
  gen_goal(1,Ar,A,Args,ArgsNoV,V),
  G0=..[P|Args],
  (builtin(G0)->
    G=G0
  ;
    G=..[P,_|Args]
  ),
  (setof(V,ArgsNoV^call_goal(Mod,G),LC)->
    true
  ;
    LC=[]
  ),
  append(C0,LC,C1),
  remove_duplicates(C1,C2),
  find_constants(T,Mod,C2,C).

call_goal(M,G):-
  M:G.

gen_goal(Arg,Ar,_A,[],[],_):-
  Arg =:= Ar+1,!.

gen_goal(A,Ar,A,[V|Args],ArgsNoV,V):-!,
  Arg1 is A+1,
  gen_goal(Arg1,Ar,A,Args,ArgsNoV,V).

gen_goal(Arg,Ar,A,[ArgV|Args],[ArgV|ArgsNoV],V):-
  Arg1 is Arg+1,
  gen_goal(Arg1,Ar,A,Args,ArgsNoV,V).





% in caso di setting(bottom_clause,no) invece di creare le bottom clause genera 
% clause vuote - per compatibilità con setting(bottom_clause,yes) -
init_theory(0,[]).

init_theory(N,[rule(bottom_pos,(([],[]):-([],[])),0.5),rule(bottom_neg,(([],[]):-([],[])),0.5)|Theory]):-
	N1 is N - 1,
	init_theory(N1, Theory).
	

covering_loop(_Pos,[],[],Rules,Rules,_S):-!.

/* some eminus still to cover: generate a new clause */
covering_loop(Eplus,Eminus,EminusRem,NP,NN,NR,NR2,Rulesin,Rulesout,S):-
        print_ex_rem(Eplus,Eminus),
  /* INPUT initialize_agenda/6
		% Eplus=lista ex pos; Eminus=lista ex neg; NP=Num Pos; NN=Num Neg
		% Agenda=(H,HL):-(B,BL) con H=B=[], HL=lista atomi dal .l per testa, BL=lista atomi dal .l per body,BestClause=(null,null,0,0,_,_,_) [(NameOut,BCOut,HeurOut,DetOut)]*/
		initialize_agenda(Eplus,Eminus,NP,NN,Agenda,BestClause),
		specialize(Agenda,Eplus,Eminus,NP,NN,0,BestClause,(Name,BestClauseOut,Heur,(NC,PC,Emc,Epnc))), %corrisponde a FindBestIC  %Agenda rimane invariato (vedi commento sopra)
		% NC= Num ex neg ruled out
		% PC = Num Pos Covered
		% Emc = lista ex neg ruled out da BestClauseOut, lunga NC
		% Epnc =lista ex pos not covered da BestClauseOut
        (BestClauseOut=null->
            format("No more significant clauses.~n~n",[]),
            print_ex_rem(Eplus,Eminus),
            Rulesout=Rulesin,
            NR2=NR,
            EminusRem=Eminus
        ;
            set_output(S),
            write_clause(BestClauseOut),
            NR1 is NR+1,	    
            %MODIFICATO
	    %numbervars(Name,0,_,[functor_name(xarg)]),
            numbervars(Name,0,_),
            format("/* Rule n. ~d ",[NR1]), 
            write_term(Name,[numbervars(true)]),
            format(" ~p ~p ~p ~n",[acc(Heur), negcov(NC), poscov(PC)]),
            format("Neg traces ruled out:#~p */~n~n~n",[Emc]),
            %format("/* Rule n. ~d ~p ~p ~p ~p */~n",[NR1,Name,acc(Heur),negcov(NC),poscov(PC)]),
            %test_body(BestClauseOut,Eplus,NBODY,S),
            %total_number(NBODY,0,NB),
            %format("/* Positivi ~p */~n~n",[NB]),
            set_output(user_output),
            print_new_clause(Name,BestClauseOut,Heur,NC,PC,Emc,Epnc),
            flush_output(S),
            remove_cov_examples(Emc,Eminus,EminusOut), %tolgo da Eminus la lista Emc di ex negativi esclusi dalla clausola; gli ex neg rimanenti vanno in EminusOut
            length(EminusOut,NN1), %NN1=num ex neg rimasti (ho tolto quelli esclusi dalla clausola BestClauseOut)
            Rulesout=[rule(Name,BestClauseOut,(heur(Heur),negcov(NC),poscov(PC),emc(Emc),epnc(Epnc)))|Rules1],  %formato regola
            covering_loop(Eplus,EminusOut,EminusRem,NP,NN1,NR1,NR2,Rulesin,Rules1,S)
        ).


remove_cov_examples([],Eminus,Eminus):-!.
    
remove_cov_examples([Ex|Rest],Eminus,EminusOut):-
    delete(Eminus,Ex,Eminus1),
    remove_cov_examples(Rest,Eminus1,EminusOut).



/* MIO CODICE  */


covering_loop1(_Eplus,_Eminus,_M,_NP,_NN,0,Prog,Prog,LL,LL):-!.

/* some eminus still to cover: generate a new clause */
covering_loop1(Eplus,Eminus,M,NP,NN,MR,Prog0,Prog,LL0,LL):-
		% print_ex_rem(Eplus,Eminus),%gtrace,
		%		[([rule(bottom,  ([], []:-[], []), 0.5905797108904512)], -186.75453269193804)]
		%BestClauseRule  = rule(null,([], []:-[], []),(0,0,_,_,_)), %(Name,BestClause,(H,NN,NP,Emc,Epnc))
		BestClause  = (null,([], []:-[], []),(0,0,_,_,_)), %(Name,BestClause,(H,NN,NP,Emc,Epnc))
		findBestICS([BestClause],M,Eplus,Eminus,NP,NN,Prog0,Prog0,Prog1,LL0,LL1,0),
    write2(M,'New best theory: '),nl2(M),
    write_rules2(M,Prog1),nl2(M),
    write2(M,'Score '),write2(M,LL1),nl2(M),
        %read(_),
    MR1 is MR-1,
    (LL1=:=LL0->
      Prog=Prog0,
      LL=LL0
    ;
      covering_loop1(Eplus,Eminus,M,NP,NN,MR1,Prog1,Prog,LL1,LL)
    ).

     %Rule = rule(r, Clause, _Stat),
         %Rule = rule(r, Clause, _Stat),
		%		findBestICS([BestClause],Eplus,Eminus,NP,NN,0,[],Rulesout0),
		%length(Rulesout0,LRulesout0),
		%		format("~nL Rulesout0: ~d~n",[LRulesout0]),
		%	convert_rules_covering_loop1(Rulesout0,Rulesout).
		% NC= Num ex neg ruled out
		% PC = Num Pos Covered
		% Emc = lista ex neg ruled out da BestClauseOut, lunga NC
		% Epnc =lista ex pos not covered da BestClauseOut
		%((BestClauseOut=([], []:-[], []))->
        %    format("No more significant clauses.~n~n",[]),
        %    print_ex_rem(Eplus,Eminus),
        %    Rulesout=Rulesin,
        %    NR2=NR,
        %    EminusRem=Eminus
        %;
		%		%set_output(S),
        %    write_clause(BestClauseOut),   
        %    NR1 is NR+1,	    
            %MODIFICATO
	    	%numbervars(Name,0,_,[functor_name(xarg)]),
        %    numbervars(Name,0,_),
        %    format("/* Rule n. ~d ",[NR1]), 
        %    write_term(Name,[numbervars(true)]),
        %    format(" ~p ~p ~p ~n",[acc(Heur), negcov(NC), poscov(PC)]),
        %    format("Neg traces ruled out:#~p */~n~n~n",[Emc]),
            %format("/* Rule n. ~d ~p ~p ~p ~p */~n",[NR1,Name,acc(Heur),negcov(NC),poscov(PC)]),
            %test_body(BestClauseOut,Eplus,NBODY,S),
            %total_number(NBODY,0,NB),
            %format("/* Positivi ~p */~n~n",[NB]),
        %    set_output(user_output),
        %    print_new_clause(Name,BestClauseOut,Heur,NC,PC,Emc,Epnc), %******CHECK
			%flush_output(S),
        %    remove_cov_examples(Emc,Eminus,EminusOut), %tolgo da Eminus la lista Emc di ex negativi esclusi dalla clausola; gli ex neg rimanenti vanno in EminusOut
        %    length(EminusOut,NN1), %NN1=num ex neg rimasti (ho tolto quelli esclusi dalla clausola BestClauseOut)
			%format("********************** num ex neg rimasti ~d",[NN1]),
        %    Rulesout=[rule(Name,BestClauseOut,(heur(Heur),negcov(NC),poscov(PC),emc(Emc),epnc(Epnc)))|Rules1],  %formato regola
        %    covering_loop1(Eplus,EminusOut,EminusRem,NP,NN1,NR1,NR2,Rulesin,Rules1)
        %).

convert_rules_covering_loop1([],[]).

convert_rules_covering_loop1([(Name,BestClauseOut,Heur,(NC,PC,Emc,Epnc))|T],[rule(Name,BestClauseOut,(heur(Heur),negcov(NC),poscov(PC),emc(Emc),epnc(Epnc)))|T1]):-
	convert_rules_covering_loop1(T,T1).

%findBestICS([],_Ep,_Em,_NPT,_NNT,_N,BestClause,BestClause):-!.


findBestICS(_Ag,M,_Ep,_Em,_NPT,_NNT,_,Prog,Prog,LL,LL,N):-
		M:local_setting(max_nodes,NMax), %max num iterazioni 
        N>NMax,!.

/*findBestICS(_Ag,_Ep,_Em,_NPT,_NNT,_N,(Name,BestClause,H,(NN,NP,Emc,Epnc)),(Name,BestClause,H,(NN,NP,Emc,Epnc))):-
         H==1,  %regole con euristica 1 e copertura > del setting non vengono raffinate
         setting(min_coverage,MC),
	     NN>=MC,!.
*/

findBestICS(Agenda,M,Ep,Em,NPT,NNT,Prog00,Prog0,Prog,LL0,LL,N):-
		%	generate_new_agenda1(Ep,Em,NPT,NNT,Agenda,[],NewAgenda,BCIn,BC1),%raffina - Agenda è il beam corrente, NewAgenda quello aggiornato - BCIn = lista corrente di AllRefinements > minacc e > mincov
	format2(M,"Beam iteration ~d~n",[N]),
	generate_new_agenda1(Ep,Em,M,NPT,NNT,Agenda,[],NewAgenda,Prog00,Prog0,Prog1,LL0,LL1),%raffina - Agenda è il beam corrente, NewAgenda quello aggiornato - BCIn = lista corrente di AllRefinements > minacc e > mincov
	%	length(NewAgenda,LNA),%NewAgenda è il beam ordinato
	%	length(BC1,LBC1),
	%	format("~nlunghezza NewAgenda: ~d~n",[LNA]),
 %	format("lunghezza BC1: ~d~n",[LBC1]),
	N1 is N+1,!,
	%    findBestICS(NewAgenda,Ep,Em,NPT,NNT,N1,BC1,BCOut).
	findBestICS(NewAgenda,M,Ep,Em,NPT,NNT,Prog00,Prog1,Prog,LL1,LL,N1).

%raffina - Agenda è il beam corrente, NewAgenda quello aggiornato - BCIn = lista corrente di AllRefinements > minacc e > mincov

generate_new_agenda1(_Ep,_Em,_M,_NPT,_NNT,[],NewAg,NewAg,_,Prog,Prog,LL,LL):-!.    

generate_new_agenda1(Ep,Em,M,NPT,NNT,[Rule0|Rest],NAgIn,NAgOut,Prog00,Prog0,Prog,LL0,LL):-
	%    findall(NewClause,refine(Clause, NewClause),Ref),
	Rule0=(N,R0,P),
	Rule=rule(N,R0,P),
	format3(M,"Revision of one clause ",[]),nl3(M),
  write3(M,Rule),nl3(M),
	findall(RS, generalize_theory([Rule],M,RS),LRef), %LRef=lista di liste, 1 per clausola raffinata
  %maplist(writeln,LRef),
  %read(_),
	%	write(LRef),nl,
    evaluate_all_refinements(Ep,Em,M,NPT,NNT,LRef,NAgIn,NAg1,Prog00,Prog0,Prog1,LL0,LL1),!, 
    format3(M,"Current best theory\n",[]),
    write_rules3(M,Prog1),nl3(M),
    write3(M,'LL '),write3(M,LL1),nl3(M),

	%evaluate_all_refinements(Ep,Em,NPT,NNT,LRef,NAgIn,NAg1,BCIn,BC1),!, %NAg1=beam ordinato per heuristic; BC1=lista non ordinata
   %evaluate_all_refinements(Ep,Em,NPT,NNT,[HRef|TRef],Name,NAgIn,NAgOut,(NameIn,BCIn,HeurIn,DetIn),(NameOut,BCOut,HeurOut,DetOut)):-
    generate_new_agenda1(Ep,Em,M,NPT,NNT,Rest,NAg1,NAgOut,Prog00,Prog1,Prog,LL1,LL).

generalize_theory(Theory,M,Ref):-
  member(rule(N,R0,P0),Theory),
  (M:local_setting(bottom_clause,no) ->
    refine_no_bc(R0,M,R)%gtrace,
   ;
    refine(R0,M,R)
  ),
  M:local_setting(max_refinements, NR),
  ( NR=none ->
    delete(Theory,rule(N,R0,P0),T0),
    append(T0,[rule(r,R,0.5)],Ref)
  ;
    random_between(0, 100, RandValue),
    RandValue > 30,
    delete(Theory,rule(N,R0,P),T0),
    append(T0,[rule(N,R,P)],Ref)
  ).

% body
% ([], []:-[], [])
refine_no_bc(((H,HL):-(B,BL)),M,((H1,HL):-(B1,BL))):-
  length(B,BN),
  M:local_setting(max_lengths,[BodyLength,_,_,_]),  
  BN<BodyLength,
  findall(BLB, M:modeb(_,BLB), BLS),   %raccolgo i modeb e specializzo il body come in slipcover
  specialize_rule_body(BLS,(H:-B),M,(H1:-B1)).      %corrisponde a specialize_rule/5 - H:-B corrisponde a Rule

% head
refine_no_bc(((H,HL):-(B,BL)),M,((H1,HL):-(B1,BL))):-
%  length(H,HN),
%  setting(max_lengths,_,NDisj,NPlus,NMinus),  
%  HN=<NDisj,
  findall(HLH , M:modeh(_,HLH), HLS),%gtrace,   %raccolgo i modeh per la testa
  refine_head_no_bc(HLS,(H:-B),M,(H1:-B1)).      %corrisponde a specialize_rule/5 fatta su testa - H:-B corrisponde a Rule

specialize_rule_body([Lit|_RLit],(H:-B),M,(H:-BL1)):-  %Lit contiene modeb
  M:local_setting(lookahead,yes),
  check_recall(modeb,M,Lit,B),
  extract_lits_from_head(H,HL),
  append(HL,B,ALL),
  (	M:lookahead(Lit,LLit1)
  ;
	M:lookahead_cons(Lit,LLit1)
  ),
  specialize_rule_la(LLit1,M,HL,B,LLitOut),
  specialize_lit([Lit|LLitOut],M,ALL,SLitList),
  remove_copies(SLitList,ALL,SLitList1),
  append(B,SLitList1,BL1),
  linked_ic_nb(BL1,M,H).

specialize_rule_body([Lit|_RLit],(H:-B),M,(H:-BL1)):-  %Lit contiene modeb  
  check_recall(modeb,M,Lit,B),
  extract_lits_from_head(H,HL),
  append(HL,B,ALL),
  specialize_lit([Lit],M,ALL,[SLit]),
  not_member(SLit,ALL),
  append(B,[SLit],BL1),
  linked_ic_nb(BL1,M,H).

specialize_rule_body([_|RLit],Rule,M,SpecRul):-
  specialize_rule_body(RLit,Rule,M,SpecRul).

not_member(X,List):-
  \+member(X,List),!.

not_member(X,List):-
  X=..[P|Args],
  length(Args,N),
  length(Args1,N),
  C=..[P|Args1],
  member(C,List),
  not_eq_vars(Args,Args1).

not_eq_vars([],[]):-!,fail.

not_eq_vars([H|T],[H1|T1]):-
  ( (H==H1) -> 
     (!,not_eq_vars(T,T1))
    ;
     !,true
  ).

remove_copies([],_,[]):-!.

remove_copies([H|T],ALL,T1):-
  member(H,ALL),!,
  remove_copies(T,ALL,T1).

remove_copies([H|T],ALL,[H|T1]):-
  remove_copies(T,ALL,T1).

specialize_rule_la([],_M,_LH1,BL1,BL1).

specialize_rule_la([Lit1|T],M,LH1,BL1,BL3):-
  copy_term(Lit1,Lit2),
  M:modeb(_,Lit2),
  append(BL1,[Lit2],BL2),
  specialize_rule_la(T,M,LH1,BL2,BL3).

specialize_lit([],_,_,[]):-!.

specialize_lit(Lits,M,Rule,SpecLits):-
  extract_type_vars(Rule,M,TypeVars0),
  remove_duplicates(TypeVars0,TypeVars),
  specialize_lit_list(Lits,M,TypeVars,SpecLits).
  
specialize_lit_list([],_,_,[]).

specialize_lit_list([Lit|RLits],M,TypeVars,[SLit|RSLits]):-%gtrace,
  Lit =.. [Pred|Args],
  take_var_args(Args,TypeVars,Args1),
  SLit =.. [Pred|Args1],
  extract_type_vars([SLit],M,TypeVars0),
  append(TypeVars,TypeVars0,TypeVars1),
  remove_duplicates(TypeVars1,TypeVars2),
  specialize_lit_list(RLits,M,TypeVars2,RSLits).

remove_duplicates([],[]).

remove_duplicates([H|T],T1):-
  member_eq(H,T),!,
  remove_duplicates(T,T1).

remove_duplicates([H|T],[H|T1]):-
  remove_duplicates(T,T1).

refine_head_no_bc(Modehs,(H:-B),M,(HL1:-B)):- 
		%trace,
		%  write("refine_head_no_bc"),nl,
  length(H,NDisjInH),
  extract_lits_from_head(H,HL),
  M:local_setting(max_lengths,[_,NDisj,NPlus,NMinus]),
  %append(HL,B,ALL),
  (
     (
       NDisjInH<NDisj,
       (  % genera +
	    (
	      get_recall_modeh2(Modehs,M,Lits), %Lits= lista con N letterali per ogni modeh, N recall del modeh
	      length(Lits,NLits),
	      get_number_of_samples(NLits,M,NPlus,NSamp),
	      sample_possible_heads(NSamp,M,NLits,Lits,R),
	      member(Disj,R),
	      specialize_lit(Disj,M,B,SLits),
	      append(H,[(+,SLits,[])],HL1),
	      linked_ic_nb(B,M,HL1),
	      check_absence(+,SLits,H)
	    )
	;% genera -
		(NMinus>0,
	      member(Lit,Modehs),
	      check_recall(modeh,M,Lit,HL),
	      specialize_lit([Lit],M,B,SLit),
	      append(H,[(-,SLit,[])],HL1),
	      linked_ic_nb(B,M,HL1),
	      check_absence(-,SLit,H)
	    )
       )
       
     )
   ;% raffina da +/-
     ( 
       H\=[],
       member((S,Lits,[]),H),
       append(Lits,B,ALL),
       refine_single_disj_no_bc(S,Lits,Modehs,M,SLits,HL,ALL),
       delete(H,(S,Lits,[]),H1),
       ( dif(SLits,[]) ->  
            (append(H1,[(S,SLits,[])],HL1),
	         check_absence(S,SLits,H1)
	        )
         ;
            HL1=H1
       ),
       linked_ic_nb(B,M,HL1)
     )
  ).

check_absence(S,L,H):-
  \+check_absence_int(S,L,H),!.

check_absence_int(_S,L,H):-
  member((_,L1,[]),H),
  length(L,N),
  length(L1,N),
  check_lits(L,L1),!.

check_lits([],_):-!.

check_lits([H|T],L1):-
  H=..[P|Args],
  length(Args,N),
  length(Args1,N),
  C=..[P|Args1],
  member(C,L1),!,
  eq_vars(Args,Args1),
  check_lits(T,L1).

eq_vars([],[]):-!.

eq_vars([H|T],[H1|T1]):-
  H==H1,!,
  eq_vars(T,T1).

extract_lits_from_head([],[]).

extract_lits_from_head([(_,H,_)|HL],HRes):-
  extract_lits_from_head(HL,HRes0),
  append(H,HRes0,HRes1),
  remove_duplicates(HRes1,HRes).
  
check_recall(Mode,M,Lit,_Lits):-
  get_recall(Mode,M,Lit,*),!.

check_recall(Mode,M,Lit,Lits):-
  Lit=.. [Pred|_Args],
  count_lit(Pred,Lits,N),
  get_recall(Mode,M,Lit,R),
  R > N.
  
count_lit(_,[],0):-!.

count_lit(P,[H|T],N):-
  H=..[P|_Args1],!,
  count_lit(P,T,N0),
  N is N0 + 1.

count_lit(P,[_H|T],N):-
  count_lit(P,T,N).

extract_type_vars([],_,[]).

extract_type_vars([Lit|RestLit],M,TypeVars):-
  Lit =.. [Pred|Args],
  length(Args,L),
  length(Args1,L),
  Lit1 =.. [Pred|Args1],
  take_mode(Lit1,M),
  type_vars(Args,Args1,Types),
  extract_type_vars(RestLit,M,TypeVars0),
  !,
  append(Types,TypeVars0,TypeVars).

get_recall_modeh2([],_M,[]).

get_recall_modeh2([H|T],Mo,Samples):-
  H=..[_Pred|Args],
  length(Args,N),
  count_pmc1(Args,N,_P,M,_C),
  Mo:modeh(R,H),!,
  get_recall_modeh2_int(M,Mo,R,H,T,Samples).

% caso con solo + ->  M  
get_recall_modeh2_int(0,M,_,H,T,[H|Samples]):-
  !,
  get_recall_modeh2(T,M,Samples).

% caso - e non #  ->  M
get_recall_modeh2_int(_,M,R,H,T,Samples):-
  duplicate_all_modeh1([H],M, R, ModehSampled),
  get_recall_modeh2(T,M,Samples0),
  append(ModehSampled,Samples0,Samples).

count_pmc1([],N,0,0,N).
count_pmc1([+_|T],N,P,M,C):-!,
  count_pmc1(T,N,P0,M,C0),
  P is P0 + 1,
  C is C0 - 1.
count_pmc1([-_|T],N,P,M,C):-!,
  count_pmc1(T,N,P,M0,C0),
  M is M0 + 1,
  C is C0 - 1.
count_pmc1([_|T],N,P,M,C):-
  count_pmc1(T,N,P,M,C).
  
duplicate_all_modeh1([],_,_,[]).

duplicate_all_modeh1(L,M,*,Modehs):-!,
  M:local_setting(max_length, MaxL),
  random_between(0,MaxL,R),
  duplicate_all_modeh1(L,M,R,Modehs).

duplicate_all_modeh1([H|T],M,R,Modehs):-
  duplicate_modeh1(H,R,Modehs0),
  duplicate_all_modeh1(T,M,R,Modehs1),
  append(Modehs0,Modehs1,Modehs).
  
% inserisce r modeh dove r è il valore della recall
duplicate_modeh1(_,0,[]):- !.

% inserisce r modeh dove r è il valore della recall
duplicate_modeh1(Modeh, R, [Modeh|Modehs]) :-
  R0 is R - 1,
  duplicate_modeh1(Modeh, R0, Modehs).



get_recall(modeh,M,Lit,R):-
  M:modeh(R,Lit),!.

get_recall(modeb,M,Lit,R):-
  M:modeb(R,Lit),!.

take_mode(modeh,M,Lit):-
  %input_mod(M),
  M:modeh(_,Lit),!.%M:modeh(_,Lit),!.

take_mode(modeb,M,Lit):-
  %input_mod(M),
  %M:modeb(_,Lit),!.
  M:modeb(_,Lit),!.

take_mode(Lit,M):-
  %input_mod(M),
  M:modeh(_,Lit),!.%M:modeh(_,Lit),!.

take_mode(Lit,M):-
  %input_mod(M),
  %M:modeb(_,Lit),!.
  M:modeb(_,Lit),!.

/*
take_mode(Lit):-
  input_mod(M),
  M:mode(_,Lit),!.
*/

type_vars([],[],[]).

type_vars([V|RV],[+T|RT],[V=T|RTV]):-
  !,
  type_vars(RV,RT,RTV).

type_vars([V|RV],[-T|RT],[V=T|RTV]):-atom(T),!,
  type_vars(RV,RT,RTV).

type_vars([_V|RV],[_T|RT],RTV):-
  type_vars(RV,RT,RTV).

take_var_args([],_,[]).

take_var_args([+T|RT],TypeVars,[V|RV]):-
  !,
  member(V=T,TypeVars),
  take_var_args(RT,TypeVars,RV).

take_var_args([-T|RT],TypeVars,[_V|RV]):-
  atom(T),
  take_var_args(RT,TypeVars,RV).

take_var_args([-T|RT],TypeVars,[V|RV]):-
  member(V=T,TypeVars),
  take_var_args(RT,TypeVars,RV).

take_var_args([T|RT],TypeVars,[T|RV]):-
  T\= + _,(T\= - _; T= - A,number(A)),
  take_var_args(RT,TypeVars,RV).
  
  
/*
linked_ic_nb(B,H0):-
  extract_lits_from_head(H0,H),
  linked_ic(B,H).
*/

linked_ic_nb(B,M,_) :-
  linked_clause(B,M).
 
linked_clause(X,M):-
  linked_clause(X,M,[]).

linked_clause([],_,_).

linked_clause([L|R],M,PrevLits):-
  term_variables(PrevLits,PrevVars),
  input_variables(L,M,InputVars),
  linked(InputVars,PrevVars),!,
  linked_clause(R,M,[L|PrevLits]).


linked([],_).

linked([X|R],L) :-
  member_eq(X,L),
  !,
  linked(R,L).
  

input_variables(\+ LitM,M,InputVars):-
  !,
  LitM=..[P|Args],
  length(Args,LA),
  length(Args1,LA),
  Lit1=..[P|Args1],
  copy_term(LitM,Lit0),
  M:modeb(_,Lit1),
  Lit1 =.. [P|Args1],
  convert_to_input_vars(Args1,Args2),
  Lit2 =.. [P|Args2],
  input_vars(Lit0,Lit2,InputVars).

input_variables(LitM,M,InputVars):-
  LitM=..[P|Args],
  length(Args,LA),
  length(Args1,LA),
  Lit1=..[P|Args1],
  M:modeb(_,Lit1),
  input_vars(LitM,Lit1,InputVars).

input_head_variables(LitM,InputVars):-
  LitM=..[P|Args],
  length(Args,LA),
  length(Args1,LA),
  Lit1=..[P|Args1],
  modeh(_,Lit1),
  input_vars(LitM,Lit1,InputVars).

input_vars(Lit,Lit1,InputVars):-
  Lit =.. [_|Vars],
  Lit1 =.. [_|Types],
  input_vars1(Vars,Types,InputVars).


input_vars1([],_,[]).

input_vars1([V|RV],[+_T|RT],[V|RV1]):-
  !,
  input_vars1(RV,RT,RV1).

input_vars1([_V|RV],[_|RT],RV1):-
  input_vars1(RV,RT,RV1).

convert_to_input_vars([],[]):-!.

convert_to_input_vars([+T|RT],[+T|RT1]):-
  !,
  convert_to_input_vars(RT,RT1).

convert_to_input_vars([-T|RT],[+T|RT1]):-
  convert_to_input_vars(RT,RT1).


% Raffino una E togliendo un vincolo
refine_single_disj_no_bc(+,D,_,_,D1,_,_):-
  member(E,D),
  delete(D,E,D1).

% Raffino un EN aggiungendo un vincolo
refine_single_disj_no_bc(-,D,DL,M,D1,DL1,ALL):-
  M:local_setting(max_lengths,[_,_,_,NMinus]),
  length(D,LengthD),
  LengthD<NMinus,
  member(E,DL),
  check_recall(modeh,M,E,DL1),   
  specialize_lit([E],M,ALL,[E1]),
  append(D,[E1],D1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GENERATE HEADS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Campiona N teste di lunghezza Dim
% Dim è il minimo tra il numero di possibili letterali 
% e il numero massimo di letterali inseribili in testa + dato da max_length M:local_setting
sample_possible_heads(N,M,NLits,L,R):-
  M:local_setting(max_lengths,[_,_,NPlus,_]),
  (NPlus > NLits -> Dim = NLits ; Dim = NPlus),
  sample_possible_heads1(N,Dim,L,R,[]).
 
sample_possible_heads1(0,_,_,X,X):-!.

sample_possible_heads1(R,Dim,L,T,X):-
  sample(Dim,L,N0),
  sort(N0,N),
  ( member(N,X) ->
      sample_possible_heads1(R,Dim,L,T,X)
   ;
     (!,R0 is R-1,
      sample_possible_heads1(R0,Dim,L,T,[N|X])
     )
  ).
  

sample(0,List,[],List):-!.

sample(N,List,List,[]):-
  length(List,L),
  L=<N,!.

sample(N,List,[El|List1],Li):-
  length(List,L),
  random(0,L,Pos),
  nth0(Pos,List,El,Rest),
  N1 is N-1,
  sample(N1,Rest,List1,Li).

sample(0,_List,[]):-!.

sample(N,List,List):-
  length(List,L),
  L=<N,!.

sample(N,List,[El|List1]):-
  length(List,L),
  random(0,L,Pos),
  nth0(Pos,List,El,Rest),
  N1 is N-1,
  sample(N1,Rest,List1).


get_number_of_samples(NLits,M,NtoS,NSamp):-
  NLits > NtoS,!,
  M:local_setting(num_samples,NS),
  possible_combinations(NLits,NtoS,Res),
  (NS>Res ->
    NSamp = Res
   ;
    NSamp = NS
  ).

get_number_of_samples(NLits,M,_NtoS,NSamp):-
  M:local_setting(num_samples,NS),
  possible_combinations(NLits,NLits,Res),
  (NS>Res ->
    NSamp = Res
   ;
    NSamp = NS
  ).
  
% Possibili combinazioni di lunghezza NtoS creabili con NLits diversi
% NLits!/(NLits-NtoS)!NtoS!
possible_combinations(NLits,NtoS,Res):-
  comb(NLits,NtoS,R1),
  comb(NtoS,NtoS,R2),
  Res is R1/R2. 

comb(_,0,1):-!.
comb(A,B,R):-
  B0 is B - 1,
  A0 is A - 1,
  comb(A0,B0,R0),
  R is A*R0.

%*************************************************************************************%
/* stopping criterion (1): empty agenda 

specialize([],_Ep,_Em,_NPT,_NNT,_N,BestClause,BestClause):-!.
       
specialize(_Ag,_Ep,_Em,_NPT,_NNT,N,BestClause,BestClause):-
  setting(max_nodes,NMax),
  N>NMax,!.

specialize(_Ag,_Ep,_Em,_NPT,_NNT,_N,(Name,BestClause,H,(NN,NP,Emc,Epnc)),(Name,BestClause,H,(NN,NP,Emc,Epnc))):-
  H=1.0,
  setting(min_coverage,MC),
  NN>=MC,!.

specialize(Agenda,Ep,Em,NPT,NNT,N,BCIn,BCOut):-
    generate_new_agenda(Ep,Em,NPT,NNT,Agenda,[],NewAgenda,BCIn,BC1),%raffina
    N1 is N+1,!,
    specialize(NewAgenda,Ep,Em,NPT,NNT,N1,BC1,BCOut).
    
generate_new_agenda(_Ep,_Em,_NPT,_NNT,[],NewAg,NewAg,BC,BC):-!.    

generate_new_agenda(Ep,Em,NPT,NNT,[(Name,Node,_Heur,_NN)|Rest],NAgIn,NAgOut,BCIn,BCOut):-
    findall(NewNode,refine(Node, NewNode),Ref), 
    evaluate_all_refinements(Ep,Em,NPT,NNT,Ref,Name,NAgIn,NAg1,BCIn,BC1),!,
    generate_new_agenda(Ep,Em,NPT,NNT,Rest,NAg1,NAgOut,BC1,BCOut).
*/

evaluate_all_refinements(_Ep,_Em,_M,_NPT,_NNT,[],/*_Name,*/NAg,NAg,_,Prog,Prog,LL,LL):-!.

evaluate_all_refinements(Ep,Em,M,NPT,NNT,[[HRef]|TRef],/*Name,*/NAgIn,NAgOut,Prog00,Prog0,Prog,LL0,LL):-
  already_scored(M,[HRef|Prog00],Score),!,
  write3(M,'Already scored ref, score: '),write3(M,Score),write3(M,'\n'),
  write_rules3(M,[HRef|Prog00]),
  evaluate_all_refinements(Ep,Em,M,NPT,NNT,TRef,NAgIn,NAgOut,Prog00,Prog0,Prog,LL0,LL).

evaluate_all_refinements(Ep,Em,M,NPT,NNT,[[HRef]|TRef],/*Name,*/NAgIn,NAgOut,Prog00,Prog0,Prog,LL0,LL):-
	HRef=rule(Name,HRef1,_Stat),
  write3(M,'New ref '),write3(M,'\n'),
  write_rules3(M,[HRef|Prog00]),  
  learn_param([HRef|Prog00],M,Ep,Em,Prog1,NewL1),
  write3(M,'Score: '),write3(M,NewL1),write3(M,'\n'),
  write_rules3(M,Prog1),
       % Det1 = (NN,NP,Emc,Epnc),
	   %	append([(Name,HRef1,Heuristic,Det1)],AllRefinementsIn,AllRefinementsIn1)
	    M:local_setting(beamsize,BS),
        print_ref(Name,M,HRef,NewL1,_,_,_,_),
        insert_in_order((Name,HRef1,NewL1,_),BS,NAgIn,NAg1),
        store_prog(M,Prog1,NewL1),
	( NewL1>LL0->
    LL1=NewL1,
    Prog2=Prog1
  ;
    LL1=LL0,
    Prog2=Prog0    
    ),
    evaluate_all_refinements(Ep,Em,M,NPT,NNT,TRef,NAg1,NAgOut,Prog00,Prog2,Prog,LL1,LL).

	/* VERSIONE PRECEDENTE FATTA DA RICCARDO CON 2 LISTE: AllRefinementsOut (tiene tutti i raffinamenti con euristica>MA e NN>MC, quindi alla fine può contenere decine di migliaia di clausole) e NAgOut (beam ordinato in ordine descrescente di euristica con massima dimensione beamsize settata all'inizio; però le clausole sono inserite nel beam senza verificare i setting MA e MC)
	 ((%Heuristic > HeurIn, % non verifico che il raffinamento sia migliore del precedente HeurIn ma solo che abbia accuratezza >del setting MA
    Heuristic>MA
%    ,statistically_significant(NP,NN,NNTC,NPT,NNT)
    ,NN>=MC)->
        Det1 = (NN,NP,Emc,Epnc),
		append([(Name,HRef1,Heuristic,Det1)],AllRefinementsIn,AllRefinementsIn1)
    ;
        %Name1=NameIn,
        %BC1=BCIn,
        %Heur1 = HeurIn,
        %Det1 = DetIn
		AllRefinementsIn1 = AllRefinementsIn
    ),
    (prune(NN,Heuristic,HeurIn)->  %qui non entra mai
		NAg1=NAgIn
    ;
        setting(beamsize,BS),
        print_ref(Name,HRef,Heuristic,NN,NP,Emc,Epnc),
        insert_in_order((Name,HRef1,Heuristic,NN),BS,NAgIn,NAg1)
		%format("~nInserted in beam~n")
    ),!,
    evaluate_all_refinements(Ep,Em,NPT,NNT,TRef,NAg1,NAgOut,AllRefinementsIn1,AllRefinementsOut).
*/

%test_body(((H,_HL):-(B,_BL)),Ep,NBODY,S):-
%	generate_query((([],[]):-(B,_BL)),Query,VI),
%	set_output(S),
%	total_number(Ep,0,Nep),
%	format("/* Query ~d ~p ~p */~n~n",[Nep,Query,VI]),
%	test_clause_pos(Ep,Query,VI,0,NP,[],NBODY).

store_prog(M,Ref,Score):-
  assert(M:ref_th(Ref,Score)).

elab_clause_ref(((H,_HL):-(B,_BL)),rule(H1,B1)):-
  copy_term((H,B),(H1,B1)).

already_scored(M,Prog,Score):-
  M:ref_th(P,Score),
  length(P,NR),
  length(Prog,NR),
  already_scored_clause(Prog,P).

already_scored_clause([],[]).

already_scored_clause([R|RT],[rule(H1,B1)|RT0]):-
  elab_ref([R],[rule(H,B)]),
  permutation(B,B1),
  perm_head(H,H1),
  already_scored_clause(RT,RT0).

perm_head([],_H1).

perm_head([(Sign,Lit,_DL)|T],H1):-
  member((Sign,Lit1,_),H1),
  permutation(Lit,Lit1),
  perm_head(T,H1).

elab_ref([],[]).

elab_ref([rule(_NR,((H,_HL):-(B,_BL)),_Lits)|T],[rule(H1,B1)|T1]):-!,
  copy_term((H,B),(H1,B1)),
  numbervars((H1,B1),0,_N),
  elab_ref(T,T1).

generate_query(((H,_HL):-(B,_BL)),QA,VI):-
  process_head(H,HA,VI),
  add_int_atom(B,B1,VI),
  append(B1,HA,Q),
  list2and(Q,QA).

process_head([],[],_VI).  

process_head([(+,D,_DL)|T],[\+(DA)|T1],VI):-
  add_int_atom(D,D1,VI),
  list2and(D1,DA),
  process_head(T,T1,VI).
  
process_head([(+=,D,_DL)|T],[\+(DA)|T1],VI):-
  add_int_atom(D,D1,VI),
  list2and(D1,DA),
  process_head(T,T1,VI).

process_head([(-,D,_DL)|T],[\+(\+(DA))|T1],VI):-
  add_int_atom(D,D1,VI),
  list2and(D1,DA),
  process_head(T,T1,VI).
  
process_head([(-=,D,_DL)|T],[\+(\+(DA))|T1],VI):-
  add_int_atom(D,D1,VI),
  list2and(D1,DA),
  process_head(T,T1,VI).

add_int_atom([],[],_VI).

add_int_atom([H|T],[H|T1],VI):-
  builtin(H),!,
  add_int_atom(T,T1,VI).

add_int_atom([H|T],[H1|T1],VI):-
  H=..[F|Args],
  H1=..[F,VI|Args],
  add_int_atom(T,T1,VI).



/*    
prune(_NPT,_NNT,NN,_H,HeurBestClause):-
    M:local_setting(pruning,true),
    BestHeur is (NN+1)/(NN+2),
    BestHeur<HeurBestClause.
    
prune(NPT,NNT,NN,_H,_HeurBestClause):-
    M:local_setting(pruning,true),
    BestLR is -2*NN*log10(NNT/(NNT+NPT)),
    M:local_setting(significance_level,X),
    sig_threshold(X,T),
    BestLR<T.
*/
statistically_significant(NP,NN,_NNTC,NPT,NNT):-
    PMinusCMinus is NN/(NN+NPT-NP),
    (PMinusCMinus=:=0->
        LikelihoodRatio is 2*(NN+NPT-NP)*log10(1/(NPT/(NPT+NNT)))
    ;
        (PMinusCMinus=:=1->
            LikelihoodRatio is 2*(NN+NPT-NP)*log10(1/(NNT/(NPT+NNT)))
        ;
            LikelihoodRatio is 2*(NN+NPT-NP)*
                (PMinusCMinus*log10(PMinusCMinus/(NNT/(NPT+NNT))) +
                (1-PMinusCMinus)*log10((1-PMinusCMinus)/(NPT/(NPT+NNT))))
        )
    ),
    setting(significance_level,X),
    sig_threshold(X,T),
    LikelihoodRatio>T.

statistically_significant_pos(NP,NN,NNTC,NPT,NNT):-
    PPlusC is NP/(NP+NNTC-NN),
    (PPlusC=:=0->
        LikelihoodRatio is 2*(NP+NNTC-NN)*log10(1/(NNT/(NPT+NNT)))/log10(2)
    ;
        (PPlusC=:=1->
            LikelihoodRatio is 2*(NP+NNTC-NN)*log10(1/(NPT/(NPT+NNT)))/log10(2)
        ;
            LikelihoodRatio is 2*(NP+NNTC-NN)*
                (PPlusC*log10(PPlusC/(NPT/(NPT+NNT)))/log10(2) +
                (1-PPlusC)*log10((1-PPlusC)/(NNT/(NPT+NNT)))/log10(2))
        )
    ),
    setting(significance_level,X),
    sig_threshold(X,T),
    LikelihoodRatio>T.

sig_threshold(0.995,7.88).    
sig_threshold(0.99,6.64).
sig_threshold(0.975,5.02).
sig_threshold(0.95,3.84).
sig_threshold(0.90,2.71).
sig_threshold(0.75,1.32).
sig_threshold(0,0).

list2andHead([],false):-!.

list2andHead(HeadList,Head):-
    list2and(HeadList,Head).

list2andBody([],true):-!.

list2andBody(BodyList,Body):-
    list2and(BodyList,Body).
    
/*
initialize_agenda(Pos,Neg,NP,NN,L,(NameOut,BCOut,HeurOut,DetOut)):-
  findall( (Name,[],H,B,[]),template(dynamic,fixed,Name,H,B),L_dynamic_fixed1),
  evaluate_fixed(Pos,Neg,NP,NN,L_dynamic_fixed1,[],L_dynamic_fixed,(null,null,0,(0,_,_,_)),(NameOut1,BCOut1,HeurOut1,DetOut1)),
  findall( (Name,H,[],B,[]),template(fixed,fixed,Name,H,B),L_fixed_fixed1),
  evaluate_fixed(Pos,Neg,NP,NN,L_fixed_fixed1,L_dynamic_fixed,L_fixed_fixed,(NameOut1,BCOut1,HeurOut1,DetOut1),(NameOut2,BCOut2,HeurOut2,DetOut2)),
  findall( (Name,H,[],[],B),template(fixed,dynamic,Name,H,B),L_fixed_dynamic1),
  evaluate_fixed(Pos,Neg,NP,NN,L_fixed_dynamic1,L_fixed_fixed,L_fixed_dynamic,(NameOut2,BCOut2,HeurOut2,DetOut2),(NameOut,BCOut,HeurOut,DetOut)),
  findall( (Name,(([],H):-([],B)),0,NN ),template(dynamic,dynamic,Name,H,B),L_dynamic_dynamic),
  append(L_fixed_dynamic,L_dynamic_dynamic,L).

evaluate_fixed(_Pos,_Neg,_NPT,_NNT,[],Ag,Ag,BC,BC).
 
evaluate_fixed(Ep,Em,NPT,NNT,[(Name,H,HL,B,BL)|TRef],NAgIn,NAgOut,(NameIn,BCIn,HeurIn,DetIn),
        (NameOut,BCOut,HeurOut,DetOut)):-
    extract_disj(H,H1),
    generate_query(((H1,HL):-(B,BL)),Query,VI),
    test_clause_pos(Ep,Query,VI,0,NP,[],Epc),
    test_clause_neg(Em,Query,VI,0,NN,[],Emc),
    deleteall(Ep,Epc,Epnc),
%    length(Em,NNTC),
    setting(min_coverage,MC),
    setting(min_accuracy,MA),
    (setting(heur,laplace)->
        Heuristic is (NN+1)/(NPT-NP+NN+2)
    ;
      Den is (NPT-NP+NN),
      (Den=0->
        Heuristic is 0     
      ;         
        Heuristic is (NN)/Den      
      )
    ),
    DetIn=(NNIn,_,_,_),
    (((Heuristic > HeurIn;Heuristic=:=HeurIn,NN>NNIn),Heuristic>MA
%    ,statistically_significant(NP,NN,NNTC,NPT,NNT)
    ,NN>=MC)->
        Name1=Name,
        BC1 = ((H1,HL):-(B,BL)),
        Heur1 = Heuristic,
        Det1 = (NN,NP,Emc,Epnc)
    ;
        Name1=NameIn,
        BC1=BCIn,
        Heur1 = HeurIn,
        Det1 = DetIn
    ),
    print_ref(Name,((H1,HL):-(B,BL)),Heuristic,NN,NP,Emc,Epnc),
    (prune(NPT,NNT,NN,Heuristic,HeurIn)->
        NAg1=NAgIn
    ;
        insert_in_order((Name,((H1,HL):-(B,BL)),Heuristic,NN),10000000,NAgIn,NAg1)
    ),!,
    evaluate_fixed(Ep,Em,NPT,NNT,TRef,NAg1,NAgOut,(Name1,BC1,Heur1,Det1),
        (NameOut,BCOut,HeurOut,DetOut)).
*/

extract_disj([],[]).

extract_disj([(S,D)|T],[(S,D,[])|T1]):-
	extract_disj(T,T1).  
/*
initialize_clause(NAME,[],H,B,[]):-
	template(fixedbody,NAME,H,B).
	

initialize_clause(NAME,[],H,[],B):-
	template(dynamicbody,NAME,H,B).
*/
	
  
  

gen_cov_eminus([],[]):-!.

gen_cov_eminus([H|T],[(H,[])|T1]):-  
    gen_cov_eminus(T,T1).
    
print_ex_rem(Eplus,Eminus):-
        setting(verbosity,V),
        V>0,
        length(Eplus,Lp),
        format("Positive examples remaining: ~d~N~p~N~N",[Lp,Eplus]),
        length(Eminus,Lm),
        format("Negative examples remaining: ~d~N~p~N~N",[Lm,Eminus]).

insert_in_order(C,BeamSize,[],[C]):-
        BeamSize>0,!.

insert_in_order(_NewClauseItem,0,Beam,Beam):-!.


insert_in_order((Name,HRef,Heuristic,NN),BeamSize,
        [(Name1,HRef1,Heuristic1,NN1)|RestBeamIn],
        BeamOut):-
    (Heuristic>Heuristic1),!,
    % bigger heuristic, insert here
    NewBeam=[(Name,HRef,Heuristic,NN),(Name1,HRef1,Heuristic1,NN1)|RestBeamIn],
    length(NewBeam,L),
    (L>BeamSize->
        nth1(L,NewBeam,_Last,BeamOut)
        
    ;
        BeamOut=NewBeam
    ).

insert_in_order((Name,HRef,Heuristic,NN),BeamSize,
        [(Name1,HRef1,Heuristic1,NN1)|RestBeamIn],
        [(Name1,HRef1,Heuristic1,NN1)|RestBeamOut]):-
    BeamSize1 is BeamSize -1,
	%	format("beamsize = ~d~n",[BeamSize1]),
    insert_in_order((Name,HRef,Heuristic,NN),BeamSize1,RestBeamIn,
                RestBeamOut).


        



/* test_clause_pos(PosEx,(Head:-Body),NIn,NOut,CovIn,CovOut) returns in NOut
the number of ex where the clause is true and in CovOut a list of covered examples */    
test_clause_pos([],_Mo,_Q,_VI,N,N,Ec,Ec):-!.

test_clause_pos([Module|Rest],Mo,Q,VI,NIn,NOut,EcIn,EcOut):-
  copy_term(r(Q,VI),r(Q1,VI1)),
  VI1=Module,
    (call(Mo:Q1)->
        N is NIn,
        Ec=EcIn
    ;
      (Mo:mult(Module,M)->
        N is NIn+M
      ;
        N is NIn + 1
      ),
        Ec =[Module|EcIn]
    ),
    test_clause_pos(Rest,Mo,Q,VI,N,NOut,Ec,EcOut).                

test_clause_neg([],_Mo,_Q,_VI,N,N,Ec,Ec):-!.

test_clause_neg([Module|Rest],Mo,Q,VI,NIn,NOut,EcIn,EcOut):-
  copy_term(r(Q,VI),r(Q1,VI1)),
  VI1=Module,
    (call(Mo:Q1)->
      (Mo:mult(Module,M)->
        N is NIn+M
      ;
        N is NIn + 1
      ),
        Ec =[Module|EcIn]
    ;
        N is NIn,
        Ec=EcIn
    ),
    test_clause_neg(Rest,Mo,Q,VI,N,NOut,Ec,EcOut).                

distribute_not(L,\+ L):-
    L\=(_,_),!.

distribute_not((L,RestL),(\+ L ,NewRestL)):-
    distribute_not(RestL,NewRestL).

remove_red(_Pos,[],P,P).

remove_red(Pos,[rule(Name,C,Stat)|T],PIn,POut):-
  reduce_clause(Pos,C,CRed),
  append(PIn,[rule(Name,CRed,Stat)],P1),
  remove_red(Pos,T,P1,POut).  

reduce_clause(Pos,((H,HL):-(B,BL)),((HR,HL):-(B,BL))):-
  reduce_head(B,Pos,H,[],HR).

reduce_head(_B,_Pos,[],Head,Head).
  
reduce_head(B,Pos,[H|T],HeadIn,HeadOut):-
  generate_query((([H],_):-(B,_)),Q,VI),
  test_clause_pos(Pos,Q,VI,0,NP,[],Epc),
  (NP=0->
    Head1=HeadIn,
    Pos1=Pos
  ;
    append(HeadIn,[H],Head1),
    deleteall(Pos,Epc,Pos1)
  ),
  reduce_head(B,Pos1,T,Head1,HeadOut).


deleteall(L,[],L).

deleteall(L,[H|T],LOut):-
  delete(L,H,L1),
  deleteall(L1,T,LOut).

get_pos_neg(DB,Mod,Pos,Neg):-
  (Mod:local_setting(examples,keys(P))->
    AtomP=..[P,M,pos],
    Atom=..[P,M],
    (current_predicate(Mod:P/1)->
      (current_predicate(Mod:P/2)->
        findall(M,(member(M,DB),(Mod:AtomP;Mod:Atom)),Pos0),
        findall(M,(member(M,DB),\+ Mod:AtomP,\+ Mod:Atom),Neg)
      ;
        findall(M,(member(M,DB),Mod:Atom),Pos0),
        findall(M,(member(M,DB),\+ Mod:Atom),Neg)
      )
    ;
      findall(M,(member(M,DB),Mod:AtomP),Pos0),
      findall(M,(member(M,DB),\+ Mod:AtomP),Neg)
    )
  ;
    AtomP=..[pos,M],
    findall(M,(member(M,DB),Mod:AtomP),Pos0),
    findall(M,(member(M,DB),\+ Mod:AtomP),Neg)
  ),
  remove_duplicates(Pos0,Pos).

    
load_models(File,HB,Pos,Neg):-
  (setting(examples,keys(P))->
    reconsult(File),
    AtomP=..[P,M,pos],
    AtomN=..[P,M,neg],
    findall(M,AtomP,Pos),
    findall(M,AtomN,Neg),
    HB=[]
  ;
    open(File,read,Stream),
    read_models(Stream,[],HB,ModulesList),
    close(Stream),
    divide_pos_neg(ModulesList,[],Pos,[],Neg)
  ). %nomrmale

read_models(Stream,HB0,HB,[Name1|Names]):-
    read(Stream,begin(model(Name))),!,
    (number(Name)->
        name(Name,NameStr),
        append("i",NameStr,Name1Str),
        name(Name1,Name1Str)
    ;
        Name1=Name
    ),
    read_all_atoms(Stream,HB0,HB1,Name1),
    read_models(Stream,HB1,HB,Names).

read_models(_S,HB,HB,[]).

read_all_atoms(Stream,HB0,HB,Name):-
    read(Stream,Atom),
    Atom \=end(model(_Name)),!,
    Atom=..[Pred|Args],
    Atom1=..[Pred,Name|Args],
    assertz(Atom1),
    functor(Atom1,F,A),
    (member(F/A,HB0)->
    	HB1=HB0
    ;
    	HB1=[F/A|HB0]
    ),
    read_all_atoms(Stream,HB1,HB,Name).    

    
read_all_atoms(_S,HB,HB,_N).


/*
load_models(File,HB,ModulesList):-
    open(File,read,Stream),
    read_models(Stream,[],HB,ModulesList),
    close(Stream).
*/

title(File,Stream):-
        max_spec_steps(Spec),
        der_depth(Der),
        beamsize(B),
        ver(V),
        setting(verbosity,Ver),
        verapp(Vapp),
        min_cov(MC),
        format(Stream,"~N ~N/*~NACL1 ver ~a AbdProofProc. ver ~a~NFile name: ~a~N",
                [V,Vapp,File]),
        format(Stream,"Max spec steps=~d, Beamsize=~d,~NDerivation depth=~d, Verbosity=~d, Minimum coverage=~d~N*/~N",
                        [Spec,B,Der,Ver,MC]).
  

list2and([],true):-!.

list2and([X],X):-!.

list2and([H|T],(H,Ta)):-
        list2and(T,Ta).

and2list(true,[]):-!.


and2list((H,Ta),[H|T]):-!,
        and2list(Ta,T).

and2list(X,[X]).

print_list([]):-!.

print_list([rule(Name,C,Stat)|Rest]):-
    numbervars(C,0,_M),
    write_clause(C),
	format("/* ~p ~p */~n~n",[Name,Stat]),    
	%format("/* P = ~p */~n~n",[Stat]),    
    print_list(Rest).

print_list1([],[]):-!.

print_list1([rule(Name,C,Stat)|Rest],[P|Par]):-
    numbervars(C,0,_M),
    format("~f :: ",[P]),
    write_clause(C),
	format("/* ~p ~p */~n~n",[Name,Stat]),    
    print_list1(Rest,Par).

print_list1([],_N,_Par):-!.

print_list1([rule(Name,C0,Stat,_P)|Rest],N,Par):-
    copy_term(C0,C),
    numbervars(C,0,_M),
    member([N,[P,_]],Par),
    format("~f :: ",[P]),
    write_clause(C),
    format("/* ~p ~p */~n~n",[Name,Stat]),    
    N1 is N+1,
    print_list1(Rest,N1,Par).

print_list1([]):-!.

print_list1([rule(_Name,C0,P)|Rest]):-
    copy_term(C0,C),
    numbervars(C,0,_M),
    format("~f :: ",[P]),
    write_clause(C),
	%format("/* ~p */~n~n",[Name]),    
    print_list1(Rest).



% CODICE PER SWI
load_bg(FileBG):-
  (exists_file(FileBG)->
    open(FileBG,read,S), 
    read_all_atoms_bg(S),
    close(S)
  ;
    true
  ).  

% CODICE PER YAP
%load_bg(FileBG):-
%  (file_exists(FileBG)->
%    open(FileBG,read,S), 
%    read_all_atoms_bg(S),
%    close(S)
%  ;
%    true
%  ). 

%prune(NN,_H,HeurBestClause):-
%    setting(pruning,true),
%    BestHeur is (NN+1)/(NN+2),
%    BestHeur<HeurBestClause.
    
%prune(NN,_H,_HeurBestClause):-
%    setting(pruning,true),
%    nnt(NNT),
%    npt(NPT),
%    BestLR is -2*NN*log(10,NNT/(NNT+NPT)),
%    setting(significance_level,X),
%    sig_threshold(X,T),
%    BestLR<T.

 

process((H:-B),(H1:-B1)):-!,
  add_int_atom([H],[H1],VI),
  and2list(B,BL),
  add_int_atom(BL,BL1,VI),
  list2and(BL1,B1).  
      
process(H,H1):-!,
  add_int_atom([H],[H1],_VI).


learn_param([],M,_,_,[],MInf):-!,
  M:local_setting(minus_infinity,MInf).

learn_param(Program0,M,Pos,Neg,Program,NewL1):-
  M:local_setting(learning_algorithm,lbfgs),!,  
  format3(M,"Parameter learning by lbfgs~n",[]),
  convert_prob(Program0,Pr1),
%  gen_par(0,NC,Par0),
  length(Program0,N),
  length(Pos,NPos),
  length(Neg,NNeg),
  NEx is NPos+NNeg,
  gen_initial_counts(N,MIP0), %MIP0=vettore di N zeri
  test_theory_pos_prob(Pos,M,Pr1,MIP0,MIP), %MIP=vettore di N zeri
  test_theory_neg_prob(Neg,M,Pr1,N,MI), %MI = [[1, 1, 1, 1, 1, 1, 1|...], [0, 0, 0, 0, 0, 0|...]
%  flush_output,
%  optimizer_set_parameter(max_step,0.001),
  optimizer_initialize(N,pascal,evaluate,[M,MIP,MI,NEx],progress,[M]),
  M:local_setting(max_initial_weight,R),
  R0 is R*(-1),
  random(R0,R,R1),  %genera val random in (-1,1)
  format3(M,"Starting parameters: ~f",[R1]),nl3(M),
  init_par(N,R1),
  evaluate_L(MIP,MI,M,L),
  IL is -L,
  format3(M,"~nInitial L ~f~n",[IL]),
  optimizer_run(_LL,Status),
  interpret_return_value(Status,Mess),
  format3(M,"Status ~p ~s~n",[Status,Mess]),
  update_theory(Program0,0,Program),
  evaluate_L(MIP,MI,M,NewL),
  NewL1 is -NewL,
  format3(M,"Final L ~f~n~n",[NewL1]),
  optimizer_finalize.

learn_param(Program0,M,Pos,Neg,Program,NewL1):-
  M:local_setting(learning_algorithm,gradient_descent),!,
  format3(M,"Parameter learning by gradient descent~n",[]),
  M:local_setting(random_restarts_number,NR),
  %write_to_file(Nodes,NR),
  convert_prob(Program0,Pr1),
  %  gen_par(0,NC,Par0),
  length(Program0,N),
  gen_initial_counts(N,MIP0), %MIP0=vettore di N zeri
  test_theory_pos_prob(Pos,M,Pr1,MIP0,MIP), %MIP=vettore di N zeri
  test_theory_neg_prob(Neg,M,Pr1,N,MI), %MI = [[1, 1, 1, 1, 1, 1, 1|...], [0, 0, 0, 0, 0, 0|...]
  length(Pos,NPos),
  length(Neg,NNeg),
  NEx is NPos+NNeg,
  random_restarts(NR,N,M,MIP,MI,NEx,1e20,Score,initial,PH),
  (PH=initial ->
    Program=Program0
  ;
    PH=..[_|LW],
    update_theory_w(Program0,LW,Program)
  ),
  NewL1 is -Score.

sigma_vec(W,SW):-
  W=..[F|ArgW],
  maplist(sigma,ArgW,ArgSW),
  SW=..[F|ArgSW].

sigma(W,S):-S is 1/(1+e^(-W)).

random_restarts(0,_NR,_MN,_MIP,_MI,_NEx,Score,Score,Par,Par):-!.

random_restarts(N,NR,M,MIP,MI,NEx,Score0,Score,Par0,Par):-
  M:local_setting(random_restarts_number,NMax),
  Num is NMax-N+1,
  format3(M,"Restart number ~d~n~n",[Num]),
  initialize_weights(NR,M,W),
  M:local_setting(gd_iter,Iter),
  M:local_setting(minus_infinity,MInf),
  gradient_descent(0,Iter,M,W,MIP,MI,NEx,NR,-MInf),
  evaluate_w(MIP,MI,W,M,_LN,ScoreR),
  format3(M,"Random_restart: Score ~f~n",[ScoreR]),
  N1 is N-1,
  (ScoreR<Score0->
    random_restarts(N1,NR,M,MIP,MI,NEx,ScoreR,Score,W,Par)
  ;
    random_restarts(N1,NR,M,MIP,MI,NEx,Score0,Score,Par0,Par)
  ).

initialize_weights(NR,M,W):-
  M:local_setting(fixed_parameters,L0),
  (is_list(L0)->
    L=L0
  ;
    length(L,NR)
  ),
  length(WA,NR),
  W=..[w|WA],
  M:local_setting(max_initial_weight,MW),
  maplist(random_weight(MW),WA,L).


random_weight(MW,W,FW):-
  var(FW),!,
  Min is -MW,
  random(Min,MW,W).

random_weight(_,FW,FW).

gradient_descent(I,I,_,_,_MIP,_MI,_NEx,_NR,_LL0):-!.

gradient_descent(Iter,MaxIter,M,W,MIP,MI,NEx,NR,LL0):-
  evaluate_w(MIP,MI,W,M,LN,LL),
  Diff is LL0-LL,
  Ratio is Diff/abs(LL0),
  M:local_setting(epsilon_em,EM),
  M:local_setting(epsilon_em_fraction,EMF),
  ((Diff<EM;Ratio<EMF)->
    write3(M,end(Diff,Ratio,LL,LL0)),nl3(M),
    true
  ;
    duplicate_term(W,WC),
    format3(M,"Gradient descent iteration ~d, LL ~f, old LL ~f~n",[Iter,LL,LL0]),
    length(GA,NR),
    G=..[g|GA],
    maplist(g_init,GA),
    M:local_setting(regularizing_constant,C),
    M:local_setting(regularization,R),
    compute_grad_w(MIP,W,G,1,MI,M,LN,NEx,R,C),
    format3(M,"Gradient:",[]),write3(M,G),nl3(M),
    format3(M,"Weights:",[]),write3(M,W),nl3(M),
    learning_rate(M,Iter,Eta),
    format3(M,"Learning rate ~f~n",[Eta]),  
    nl3(M),
    update_weights(M,W,G,Eta),
    Iter1 is Iter+1,
    assertz(M:p(WC,LL)),
    gradient_descent(Iter1,MaxIter,M,W,MIP,MI,NEx,NR,LL)
  ).

g_init(0.0).

update_weights(M,W,G,Eta):-
  functor(W,_,NR), 
  M:local_setting(fixed_parameters,FP0),
  (is_list(FP0)->
    FP=FP0
  ;
    length(FP,NR)
  ),
  numlist(1,NR,L),
  maplist(update_w(W,G,Eta),L,FP).

update_w(W,G,Eta,NR,F):-
  var(F),!,
  arg(NR,G,G0),
  arg(NR,W,W0),
  New_W0 is W0-Eta*G0,
  setarg(NR,W,New_W0).

update_w(_W,_G,_Eta,_NR,_F).

learning_rate(M,_Iter,Eta):-
  M:local_setting(learning_rate,fixed(Eta)),!.

learning_rate(M,Iter,Eta):-
  M:local_setting(learning_rate,decay(Eta_0,Eta_tau,Tau)),
  (Iter>Tau ->
    Eta = Eta_tau
  ;
    Alpha is Iter/Tau,
    Eta is (1.0-Alpha)*Eta_0+Alpha*Eta_tau
  ).
  
evaluate(L,N,_Step,M,MIP,MI,NEx):-
%		format("~nEVALUATE~n",[]),
		%  write(init_ev),nl,
		%  %  write(Step),nl,
		compute_likelihood_pos(MIP,0,0,LP),
		%format("~nlikelihood_pos: ~f",[LP]),
		%    %  write(lpos),nl,
		compute_likelihood_neg(MI,LN),
		%		format("~nlikelihood_neg:",[]), write(LN),nl,
		%      %  write(lneg),nl,
		compute_likelihood(LN,LP,M,L),
	%	format("~nL: ~f~n",[L]),
		length(MIP,LMIP),
		compute_weights(0,LMIP,LW),
    write3(M,"Weights "),write3(M,LW),nl3(M),
	 %   format("~nPesi ",[]),write(LW),nl,
		%        %  NL is -L,
		%        %  write(l),nl,
    M:local_setting(regularizing_constant,C),
    M:local_setting(regularization,R),
    compute_grad(MIP,0,MI,M,R,C,NEx,LN),
    store_hist(M,N,L).
		
compute_weights(_I,0,[]):-!.

compute_weights(I,LMIP,[P|Rest]):-
  optimizer_get_x(I,W0), 
  P is 1/(1+exp(-W0)),
  I1 is I+1,
  LMIP1 is LMIP-1,
  compute_weights(I1,LMIP1,Rest).


progress(FX,X_Norm,G_Norm,Step,_N,Iteration,Ls,0,M) :-
  format3(M,'~d. Iteration :  f(X)=~4f  |X|=~4f  |g(X)|=~4f  Step=~4f  Ls=~4f~n',[Iteration,FX,X_Norm,G_Norm,Step,Ls]),
  true.

store_hist(M,N,FX):-
  get_weights(0,N,WA),
  W=..[w|WA],
  assertz(M:p(W,FX)).

get_weights(I,I,[]):-!.

get_weights(I,N,[W0|Rest]):-
  optimizer_get_x(I,W0), 
  I1 is I+1,
  get_weights(I1,N,Rest).

convert_prob([],[]).

convert_prob([rule(_,H,_P)|T],[(Q,VI)|T1]):-
  generate_query_prob(H,Q,VI),
  convert_prob(T,T1).

generate_query_prob(((H,_HL):-(B,_BL)),QA,VI):-
  process_head(H,HA,VI),
  add_int_atom(B,B1,VI),
  append(B1,HA,Q),
  list2and(Q,QA).



test_theory_pos_prob([],_,_Theory,MIP,MIP).

test_theory_pos_prob([Module|Rest],M,Th,MIP0,MIP):-
  test_clause_prob(Th,M,Module,MIP0,MIP1),
  test_theory_pos_prob(Rest,M,Th,MIP1,MIP).
  
test_clause_prob([],_Mo,_M,MIP,MIP).

test_clause_prob([(Q,VI)|Rest],Mo,M,[MIPH0|MIPT0],[MIPH|MIPT]):-
  copy_term(r(Q,VI),r(Q1,VI1)),
  VI1=M,
  %  write(before),nl,
  % setting(approx_pl,N),
  %(N=none ->
   	findall(Q1,Mo:Q1,L),
	%;
	%	findnsols(N,Q1,Q1,L)
	%),
  %write(after),nl,
  length(L,MIP),
%  succeeds_n_times(Q1,MIP),
  MIPH is MIPH0+MIP,
  test_clause_prob(Rest,Mo,M,MIPT0,MIPT).                

test_theory_neg_prob([],_,_Theory,_N,[]).

test_theory_neg_prob([Module|Rest],M,Th,N,[MI|LMI]):-
  gen_initial_counts(N,MI0),
  test_clause_prob(Th,M,Module,MI0,MI),
  test_theory_neg_prob(Rest,M,Th,N,LMI).

test_clause_neg_prob([],_Mo,_M,B,B).

test_clause_neg_prob([(Q,VI,BDDV)|Rest],Mo,M,B0,B):-
  copy_term(r(Q,VI,BDDV),r(Q1,VI1,BDDV1)),
  VI1=M,
  findall(BDDV1,Mo:Q1,L),
  or_list(L,BDD0),
  or(BDD0,B0,B1),
   test_clause_neg_prob(Rest,Mo,M,B1,B).                

or_list([],Zero):-
  zero(Zero).

or_list([H],H):-!.

or_list([H|T],B):-
  or_list1(T,H,B).


or_list1([],B,B).

or_list1([H|T],B0,B1):-
  or(B0,H,B2),
  or_list1(T,B2,B1).

init_par(0,_):-!.

init_par(I,R1):-
  I1 is I-1,
  optimizer_set_x(I1,R1),
  init_par(I1,R1).


compute_grad_w([],_W,_G,_N,_MI,_M,_LN,_NEx,_R,_C):-!.

compute_grad_w([HMIP|TMIP],W,G,N0,MI,M,LN,NEx,R,C):-
  N00 is N0-1,
  compute_sum_neg(MI,LN,N00,M,0,S),
  arg(N0,W,W0),
  P is 1/(1+exp(-W0)),
%  optimizer_get_x(N0,W0),
  G0 is R*C*P^R*(1-P)+(HMIP-S)*P/NEx,
  setarg(N0,G,G0),
  %  optimizer_set_g(N0,G),
  N1 is N0+1,
  compute_grad_w(TMIP,W,G,N1,MI,M,LN,NEx,R,C).

evaluate_L(MIP,MI,M,L):-
  compute_likelihood_pos(MIP,0,0,LP),
  compute_likelihood_neg(MI,LN), %MI lista di liste
  compute_likelihood(LN,LP,M,L). %LN=[6.931471805599453, 0.0, 6.931471805599453, 0.0, 0.0, 0.0, 0.0, 0.0|...]

compute_likelihood([],L,_M,L).

compute_likelihood([HP|TP],L0,M,L):-
  %write(hp),write(HP),nl,
  A is 1.0-exp(-HP),
  M:local_setting(zero,Zero),
  (A=<0.0->
    A1 is Zero
  ;
    A1=A
  ),
  L1 is L0-log(A1),
  compute_likelihood(TP,L1,M,L).

compute_likelihood_neg([],[]).

compute_likelihood_neg([HMI|TMI],[HLN|TLN]):- %HMI=lista
  compute_likelihood_pos(HMI,0,0,HLN),
  compute_likelihood_neg(TMI,TLN).

compute_likelihood_pos([],_,LP,LP).%LP=0 alla fine

compute_likelihood_pos([HMIP|TMIP],I,LP0,LP):- %primo arg=vettore di 0 MI
  optimizer_get_x(I,W0), 
  P is 1/(1+exp(-W0)), %P=sigma(w)=1/(1+exp(-W0))
  LP1 is LP0-log(1-P)*HMIP,
  I1 is I+1,
  compute_likelihood_pos(TMIP,I1,LP1,LP).

compute_grad([],_N,_MI,_M,_R,_C,_NEx,_LN):-!.

compute_grad([HMIP|TMIP],N0,MI,M,R,C,NEx,LN):-
  compute_sum_neg(MI,LN,N0,M,0,S),
  optimizer_get_x(N0,W0),
  P is 1/(1+exp(-W0)),
  G is (HMIP-S)*P/NEx+R*C*P^R*(1-P),
  optimizer_set_g(N0,G),
  N1 is N0+1,
  compute_grad(TMIP,N1,MI,M,R,C,NEx,LN).

compute_sum_neg([],_LN,_I,_M,S,S).

compute_sum_neg([HMI|TMI],[HLN|TLN],I,M,S0,S):-
%  write(HMI),write(hmi),nl,
%  write(I),write('I'),nl,
  nth0(I,HMI,MIR),
%  write(MIR),write(mir),nl,
%  write(HLN),write(hln),nl,
  Den is 1.0-exp(-HLN),
  M:local_setting(zero,Zero),
  (Den=<0.0->
    Den1 is Zero
  ;
    Den1 = Den
  ),
  S1 is S0+MIR*exp(-HLN)/Den1,
  compute_sum_neg(TMI,TLN,I,M,S1,S).

gen_initial_counts(0,[]):-!.

gen_initial_counts(N0,[0|MIP0]):-
  N1 is N0-1,
  gen_initial_counts(N1,MIP0).

update_theory([],_N,[]):-!.

update_theory([rule(Name,C,_P)|Rest],N,[rule(Name,C,P)|Rest1]):-
    optimizer_get_x(N,W0),
    P is 1/(1+exp(-W0)),
    N1 is N+1,
    update_theory(Rest,N1,Rest1).


update_theory_w([],[],[]):-!.

update_theory_w([rule(Name,C,_P)|Rest],[W0|WR],[rule(Name,C,P)|Rest1]):-
    P is 1/(1+exp(-W0)),
    update_theory_w(Rest,WR,Rest1).

print_new_clause(Name,M,C,Heur,NC,PC,_Emc,_Epnc):-
        M:local_setting(verbosity,V),
        V>0,
        format(" ~N ~NGenerated clause:~n",[]),
        write_clause(C),
        nl,
        copy_term(Name,Name1),
        numbervars(Name1,0,_),
        format("Name:~p~n",[Name1]),
        format("Heuristic:~p~n",[Heur]),
        format("Neg ex ruled out:#~p~n",[NC]),
%        format("Neg ex ruled out:#~p~n",[Emc]),
        format("Covered pos ex:#~p~n",[PC]),
%        format("Covered pos ex:#~p~n",[Epnc]),
%%        format("correct: ~a, Np=~d, Npa=~d, Nm=~d, Nma=~d\c
%                ~NPos ex cov: ~p~NNeg ex cov: ~p~NAbduced literals: ~p~N ~N",
%                [C,Np,Npa,Nm,Nma,
%                Epluscovered,Eminuscovered,NewDelta]),
        (V>3->
                get0(_)
        ;
                true
        ).

write_clause(((H,_HL):-(B,_BL))):-
  copy_term(c(H,B),c(H1,B1)),
  numbervars((H1,B1),0,_M),
    write('\t'),
    (B1=[]->
      write(true)
    ;
      write_list(B1)
    ),
    nl,
    write('--->'),
    nl,
    write_head(H1).

write_head([]):-
  write('\t'),
  write('false.'),nl.

write_head([(Sign,[A|T],_DL)]):-!,
  write('\t'),
  ((Sign = '-';Sign = '-=') ->
  	write('not(')
  ;
   	true
  ),
    %  write(A),
	write_term(A,[numbervars(true)]),
  (T=[]->
    ((Sign='-';Sign='-=')->
      write(')')
    ;
      true
    )
  ;
    ((Sign='-';Sign='-=')->
      write(')\n\t/\\')
    ;
      write('\n\t/\\')
    ),
    write_list(T)
  ),
  write('.'),
  nl.

write_head([(Sign,[A|T],_DL)|HT]):-!,
  write('\t'),
  ((Sign = '-';Sign = '-=') ->
  	write('not(')
  ;
   	true
  ),
    %  write(A),
	write_term(A,[numbervars(true)]),
  (T=[]->
    ((Sign='-';Sign='-=')->
      write(')')
    ;
      true
    )
  ;
    ((Sign='-';Sign='-=')->
      write(')\n\t/\\')
    ;
      write('\n\t/\\')
    ),
    write_list(T)
  ),
  nl,
  write('\\/'),nl,
  write_head(HT).



/*
write_head([(Sign,[h(Ev,Time)|T],_DL)]):-!,
   write('\t'),
  (Sign = '+' ->
    write('E(')
  ;
    write('EN(')
  ),
  write(Ev),
  write(','),
  write(Time),
  write(')\n\t/\\'),
  write_list(T),
  nl.

write_head([(Sign,[h(Ev,Time)|T],_DL)|HT]):-!,
   write('\t'),
  (Sign= '+' ->
    write('E(')
  ;
    write('EN(')
  ),
	%MODIFICA
    %  write(Ev),
	write_term(Ev,[numbervars(true)]),
  write(','),
  write(Time),
  write(')\n\t/\\'),
  write_list(T),nl,
  write('\\/'),nl,
  write_head(HT).
*/

write_list([H]):-!,
  (H=h(E,Time)->
    write('H('),
	%MODIFICA
    %write(E),
	write_term(E,[numbervars(true)]),
    write(','),
    write(Time),
    write(')')
  ;
	%MODIFICA
    %write(H)
	write_term(H,[numbervars(true)])
  ).

write_list([H|T]):-
  (H=h(E,Time)->
    write('H('),
	%MODIFICA
    %write(E),
	write_term(E,[numbervars(true)]),
    write(','),
    write(Time),
    write(')')
  ;
	%MODIFICA
    %write(H)
	write_term(H,[numbervars(true)])
  ),
  write('\n\t/\\'),
  write_list(T).



write2(M,A):-
  M:local_setting(verbosity,Ver),
  (Ver>1->
    write(A)
  ;
    true
  ).

write3(M,A):-
  M:local_setting(verbosity,Ver),
  (Ver>2->
    write(A)
  ;
    true
  ).

nl2(M):-
  M:local_setting(verbosity,Ver),
  (Ver>1->
    nl
  ;
    true
  ).

nl3(M):-
  M:local_setting(verbosity,Ver),
  (Ver>2->
    nl
  ;
    true
  ).

format2(M,A,B):-
  M:local_setting(verbosity,Ver),
  (Ver>1->
    format(A,B)
  ;
    true
  ).

format3(M,A,B):-
  M:local_setting(verbosity,Ver),
  (Ver>2->
    format(A,B)
  ;
    true
  ).

write_rules2(M,A):-
  M:local_setting(verbosity,Ver),
  (Ver>1->
    print_list1(A)
  ;
    true
  ).

write_rules3(M,A):-
  M:local_setting(verbosity,Ver),
  (Ver>2->
    print_list1(A)
  ;
    true
  ).

print_ref(_Name,M,C,Heur,_NC,_PC,_Emc,_Epnc):-
        M:local_setting(verbosity,V),
        (V>1->
        format("Refinement:~n",[]),
		C = rule(r,C1,_),
        write_clause(C1),
		%non scrivo il nome della regola
		%        copy_term(Name,Name1),
		%numbervars(Name1,0,_),
		%format("Name:~p~n",[Name1]),
        format("Heuristic:~p~n",[Heur]),
%        format("Neg ex ruled out:#~p~n",[NC]),
%        format("Covered pos ex:#~p~n",[PC]),nl,
        (V>3->
                get0(_)
        ;
                true
        )
      ;
        true
        ).

/*
generate_file_names(File,FileKB,FileBG,FileOut,FileL):-
        name(File,FileString),
        append(FileString,".kb",FileStringKB),
        name(FileKB,FileStringKB),
        append(FileString,".bg",FileStringBG),
        name(FileBG,FileStringBG),
        append(FileString,".l",FileStringL),
        name(FileL,FileStringL),
        append(FileString,".icl.out",FileOutString),
        name(FileOut,FileOutString).
*/
% refinement operator for bodies
%
%
% Se non scelgo i raffinamento ottimale o raffino il body o la testa
% Head la testa attuale
% HeadList la testa presa a partire dal template
% Body il body attuale
% BodyList il body preso dal template
%
refine(((H,HL):-(B,BL)),M,((H1,HL1):-(B1,BL1))):-
  length(H,HN),
  length(B,BN),
  N is HN+BN,
  M:local_setting(max_length,ML),
  N=<ML,
  (M:local_setting(optimal,no)->
    ((refine_body_no(B,BL,B1,BL1),H1=H,HL1=HL)
    ;
      (refine_head_no(H,HL,M,H1,HL1),B1=B,BL1=BL)
     )  
  ;
    refine(B,BL,B1,BL1,M,H,HL,H1,HL1)
  ).

% raffino il body aggiungendo uno dei possibili 
refine_body_no(B,BL,NewB,NewBL):-
  member(E,BL),
  delete(E,BL,NewBL),
%  \+ member_eq(E,B),
  append(B,[E],NewB).

% posso raffinare il body
refine(B,BL,B1,BL1,_M,H,HL,H,HL):-
  refine_body(B,BL,B1,BL1).

% se raffino la testa non posso pi� raffinare il body quindi metto BL a []
refine(B,_BL,B,[],M,H,HL,H1,HL1):-
  refine_head(H,HL,M,H1,HL1).

% raffino il body aggiungendo un elemento e quindi riducendo la BL
refine_body(B,[H|T],NewB,T):-
  append(B,[H],NewB).

% posso raffinare il body anche non aggiungendo nulla
refine_body(B,[_H|T],NewB,BL):-
  refine_body(B,T,NewB,BL).

% Raffinamento della testa aggiungendo un disjoint
% [(+,[HD|TD],TD)] significa che per gli E inizio aggiungendo tutti i vincoli e  mi segno in TD quali sono cos� li posso eliminare
% [(+,[HD|TD],TD)] significa che per gli EN inizio mettendo solo l'EN e mi segno in TD quali sono i vincoli da aggiungere
%
% Originale
%refine_head_no(H,HL,NewH,HL):-
%  member(HH,HL),
%  (HH=(+,[HD|TD])->
%    append(H,[(+,[HD|TD],TD)],NewH)
%  ;
%    HH=(-,[HD|TD]),
%    append(H,[(-,[HD],TD)],NewH)
%  ).

refine_head_no(H,HL,_M,NewH,NewHL):-
  member(HH,HL),
  delete(HH,HL,NewHL),
  (HH=(+,[HD|TD])->
    append(H,[(+,[HD|TD],TD)],NewH)
  ;
    (HH=(-,[HD|TD])->
    	append(H,[(-,[HD],TD)],NewH)
    ;
    	(HH=(+=,[HD|TD])->
    		append(H,[(+=,[HD|TD],[])],NewH)
    	;
    		HH=(-=,[HD|TD]),
    		append(H,[(-=,[HD|TD],[])],NewH)
    	)	
    )	
  ).

% Raffinamento della testa, raffinando un disjoint
refine_head_no(H,HL,M,NewH,HL):-
  refine_disj(H,M,NewH).



refine_head(H,HL,_M,H1,HL1):-
  add_disj(H,HL,H1,HL1).

refine_head(H,_HL,M,NewH,[]):-
  refine_disj(H,M,NewH).
  
% Originale  
%add_disj(H,[HH|T],NewH,T):-
%  (HH=(+,[HD|TD])->
%    append(H,[(+,[HD|TD],TD)],NewH)
%  ;
%    HH=(-,[HD|TD]),
%    append(H,[(-,[HD],TD)],NewH)
%  ).

add_disj(H,[HH|T],NewH,T):-
  (HH=(+,[HD|TD])->
    append(H,[(+,[HD|TD],TD)],NewH)
  ;
    (HH=(-,[HD|TD])->
    	append(H,[(-,[HD],TD)],NewH)
    ;
    	(HH=(+=,[HD|TD])->
    		append(H,[(+=,[HD|TD],[])],NewH)
    	;
    		HH=(-=,[HD|TD]),
    		append(H,[(-=,[HD|TD],[])],NewH)
    	)	
    )	
  ).
  
  

add_disj(H,[_HH|T],NewH,HL):-
  add_disj(H,T,NewH,HL).


% Raffinamento del disjoint nella testa 
%
refine_disj([(Sign,D,DL)|T],M,[(Sign,D1,DL1)|T]):-
  (M:local_setting(optimal,no)->
    refine_single_disj_no(Sign,D,DL,D1,DL1)
  ;
    refine_single_disj(Sign,D,DL,D1,DL1)
  ).

% Raffinamento di un disjoint interno
refine_disj([D|T],M,[D|T1]):-
  refine_disj(T,M,T1).


% Raffino una E togliendo un vincolo
refine_single_disj_no(+,D,DL,D1,DL):-
  member(E,D),
  delete(D,E,D1).

% Raffino un EN agiungendo un vincolo
refine_single_disj_no(-,D,DL,D1,DL1):-
  member(E,DL),
  delete(E,DL,DL1),
%  \+ member_eq(E,D),
  append(D,[E],D1).
  
% Gli elementi con += vanno lasciati intonsi
%refine_single_disj_no(+=,D,DL,D,DL). 

% Gli elementi con -= vanno lasciati intonsi
%refine_single_disj_no(-=,D,DL,D,DL). 


refine_single_disj(+,D,[H|T],D1,T):-
  delete(D,H,D1).

refine_single_disj(+,D,[_H|T],D1,DL1):-
  refine_single_disj(+,D,T,D1,DL1).

refine_single_disj(-,D,[H|T],D1,T):-
  append(D,[H],D1).

refine_single_disj(-,D,[_H|T],D1,DL1):-
  refine_single_disj(-,D,T,D1,DL1).

% Gli elementi con += vanno lasciati intonsi
%refine_single_disj(+=,D,DL,D,DL).

% Gli elementi con -= vanno lasciati intonsi
%refine_single_disj(-=,D,DL,D,DL).



number(+inf,Inf):-
    Inf is inf, !.
number(-inf,MInf):-
    MInf is -inf, !.
number(X,Y):-
    Y is X, !.



%--------------
aleph_member1(H,[H|_]):- !.
aleph_member1(H,[_|T]):-
    aleph_member1(H,T).

aleph_member2(X,[Y|_]):- X == Y, !.
aleph_member2(X,[_|T]):-
    aleph_member2(X,T).

aleph_member3(A,A-B):- A =< B.
aleph_member3(X,A-B):-
    A < B,
    A1 is A + 1,
    aleph_member3(X,A1-B).

aleph_member(X,[X|_]).
aleph_member(X,[_|T]):-
    aleph_member(X,T).

%----------------
goals_to_list((true,Goals),T):-
    !,
    goals_to_list(Goals,T).
goals_to_list((Goal,Goals),[Goal|T]):-
    !,
    goals_to_list(Goals,T).
goals_to_list(true,[]):- !.
goals_to_list(Goal,[Goal]).

list_to_goals([Goal],Goal):- !.
list_to_goals([Goal|Goals],(Goal,Goals1)):-
    list_to_goals(Goals,Goals1).


prune(_):-fail.

in((Head:-true),Head):- !.
in((Head:-Body),L):-
    !,
    in((Head,Body),L).
in((L1,_),L1).
in((_,R),L):-
    !,
    in(R,L).
in(L,L).

in((L1,L),L1,L).
in((L1,L),L2,(L1,Rest)):-
    !,
    in(L,L2,Rest).
in(L,L,true).

member_eq(A,[H|_T]):-
  A==H,!.

member_eq(A,[_H|T]):-
  member_eq(A,T).

clear_kb([]).

clear_kb([F/A|T]):-
	abolish(F,A),
	clear_kb(T).

/**
 * builtin(+Goal:atom) is det
 *
 * Succeeds if Goal is an atom whose predicate is defined in Prolog
 * (either builtin or defined in a standard library).
 */
builtin(G):-
  builtin_int(G),!.

builtin_int(average(_L,_Av)).
builtin_int(G):-
  predicate_property(G,built_in).
builtin_int(G):-
  predicate_property(G,imported_from(lists)).
builtin_int(G):-
  predicate_property(G,imported_from(apply)).
builtin_int(G):-
  predicate_property(G,imported_from(nf_r)).
builtin_int(G):-
  predicate_property(G,imported_from(matrix)).
builtin_int(G):-
  predicate_property(G,imported_from(clpfd)).

average(L,Av):-
        sum_list(L,Sum),
        length(L,N),
        Av is Sum/N.


/**
 * set_pascal(:Parameter:atom,+Value:term) is det
 *
 * The predicate sets the value of a parameter
 * For a list of parameters see
 * https://github.com/friguzzi/pascal/blob/master/doc/manual.pdf or
 */
set_pascal(M:Parameter,Value):-
  retract(M:local_setting(Parameter,_)),
  assert(M:local_setting(Parameter,Value)).

/**
 * setting_pascal(:Parameter:atom,-Value:term) is det
 *
 * The predicate returns the value of a parameter
 * For a list of parameters see
 * https://github.com/friguzzi/pascal/blob/master/doc/manual.pdf or
 */
setting_pascal(M:P,V):-
  M:local_setting(P,V).
  
/*
portray(xarg(N)) :-
    format('X~w',[N]).
*/


assert_all([],_M,[]).

assert_all([H|T],M,[HRef|TRef]):-
  assertz(M:H,HRef),
  assert_all(T,M,TRef).

assert_all([],[]).

assert_all([H|T],[HRef|TRef]):-
  assertz(slipcover:H,HRef),
  assert_all(T,TRef).


retract_all([],_):-!.

retract_all([H|T],M):-
  erase(M,H),
  retract_all(T,M).

retract_all([]):-!.

retract_all([H|T]):-
  erase(H),
  retract_all(T).

make_dynamic(M):-
  M:(dynamic int/1),
  findall(O,M:output(O),LO),
  findall(I,M:input(I),LI),
  findall(I,M:input_cw(I),LIC),
  findall(D,M:determination(D,_DD),LDH),
  findall(DD,M:determination(_D,DD),LDD),
  findall(DH,(M:modeh(_,_,_,LD),member(DH,LD)),LDDH),
  append([LO,LI,LIC,LDH,LDD,LDDH],L0),
  remove_duplicates(L0,L),
  maplist(to_dyn(M),L).

to_dyn(M,P/A):-
  A1 is A+1,
  M:(dynamic P/A1),
  A2 is A1+2,
  M:(dynamic P/A2),
  A3 is A2+1,
  M:(dynamic P/A3).




pascal_expansion((:- begin_bg), []) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  assert(M:bg_on).

pascal_expansion(C, M:bgc(C)) :-
  prolog_load_context(module, M),
  C\= (:- end_bg),
  pascal_input_mod(M),
  M:bg_on,!.

pascal_expansion((:- end_bg), []) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  retractall(M:bg_on),
  findall(C,M:bgc(C),L),
  retractall(M:bgc(_)),
  (M:bg(BG0)->
    retract(M:bg(BG0)),
    append(BG0,L,BG),
    assert(M:bg(BG))
  ;
    assert(M:bg(L))
  ).

pascal_expansion((:- begin_in), []) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  assert(M:in_on).

pascal_expansion(rule(C,P), M:inc(rule(C,P))) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),
  M:in_on,!.

pascal_expansion(ic(String), M:inc(rule((Head:-Body),P))) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),
  M:in_on,!,
  parse_ics_string(String,ICs),
  add_var(ICs,[rule(((Head,_):-(Body,_)),0,P)]).

pascal_expansion((:- end_in), []) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  retractall(M:in_on),
  findall(C,M:inc(C),L),
  retractall(M:inc(_)),
  (M:in(IN0)->
    retract(M:in(IN0)),
    append(IN0,L,IN),
    assert(M:in(IN))
  ;
    assert(M:in(L))
  ).

pascal_expansion(begin(model(I)), []) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  retractall(M:model(_)),
  assert(M:model(I)),
  assert(M:int(I)).

pascal_expansion(end(model(_I)), []) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  retractall(M:model(_)).

pascal_expansion(At, A) :-
  prolog_load_context(module, M),
  pascal_input_mod(M),
  M:model(Name),
  At \= (_ :- _),
  At \= end_of_file,
  (At=neg(Atom)->
    Atom=..[Pred|Args],
    Atom1=..[Pred,Name|Args],
    A=neg(Atom1)
  ;
    (At=prob(Pr)->
      A=prob(Name,Pr)
    ;
      At=..[Pred|Args],
      Atom1=..[Pred,Name|Args],
      A=Atom1
    )
  ).




:- thread_local pascal_file/1.

user:term_expansion((:- pascal), []) :-!,
  prolog_load_context(source, Source),
  asserta(pascal_file(Source)),
  prolog_load_context(module, M),
  retractall(M:local_setting(_,_)),
  findall(local_setting(P,V),default_setting_pascal(P,V),L),
  assert_all(L,M,_),
  assert(pascal_input_mod(M)),
  retractall(M:rule_sc_n(_)),
  assert(M:rule_sc_n(0)),
  M:dynamic((modeh/2,mult/2,modeb/2,
    lookahead/2,
    lookahead_cons/2,lookahead_cons_var/2,
    bg_on/0,bg/1,bgc/1,in_on/0,in/1,inc/1,int/1,
    p/2,model/1,ref_th/2,fold/2)),
  style_check(-discontiguous).


user:term_expansion(end_of_file, C) :-
  pascal_file(Source),
  prolog_load_context(source, Source),
  retractall(pascal_file(Source)),
  prolog_load_context(module, M),
  pascal_input_mod(M),!,
  retractall(pascal_input_mod(M)),
  C=[(:- style_check(+discontiguous)),end_of_file].

user:term_expansion(In, Out) :-
  \+ current_prolog_flag(xref, true),
  pascal_file(Source),
  prolog_load_context(source, Source),
  pascal_expansion(In, Out).


