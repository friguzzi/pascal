.. pascal documentation master file, created by
   sphinx-quickstart on Thu May 30 15:47:04 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========================================
Pascal Manual
===========================================

.. toctree::
	:maxdepth: 2


Introduction
============
Pasccal is an algorithm for learning probabilistic integrity constraints. It was 
proposed in :cite:`BelAlbRig20-TPLP-IJ`. 
It contains modules for both structure and parameter learning.

Pascal is also available in the cplint on SWISH web application at `<http://cplint.eu>`_.

Installation
============
Pascal is distributed as a `pack <http://www.swi-prolog.org/pack/list?p=pack>`_ of `SWI-Prolog <http://www.swi-prolog.org/>`_. 
To install it, use ::

.. code:: prolog

	?- pack_install(pack).

Requirements
-------------
It requires the pack

 * `lbfgs <https://github.com/friguzzi/lbfgs>`_
 
It is installed automatically when installing pack `pascal` or can be installed manually as ::

	$ swipl
	?- pack_install(lbfgs).

`lbfgs` uses a foreign library and contains the library binaries for 32 and 64 bits Linux. If you want to recompile the foreign library you can use ::

	?- pack_rebuild(lbfgs).

On 32 and 64 bits Linux this should work out of the box. 



You can upgrade the pack with ::

    $ swipl
    ?- pack_upgrade(pack).

Note that the pack on which `pascal` depends is not upgraded automatically in this case so it needs to be upgraded manually.

Example of use
---------------
::

    $ cd <pack>/pascal/prolog/examples
    $ swipl
    ?- [bongardkeys].
    ?- induce_pascal([train]),T).

Testing the installation
------------------------
::

    $ swipl
    ?- [library(test_pascal)].
    ?- test_pascal.

Datasets
--------

Other machine learning datasets are available in pack `cplint_datasets <https://github.com/friguzzi/cplint_datasets>`_.

Support
-------

Use the Google group `<https://groups.google.com/forum/#!forum/cplint>`_.

Language
========
A Probabilistic Constraint Logic Theory (PCLT) is a set of Probabilistic Integrity Constraints (PIC) of the form

.. math:: 

  p \ ::\ L_1,\ldots,L_b\rightarrow \exists(P_1);\ldots;\exists(P_n);\forall\neg(N_1);\ldots;\forall\neg(N_m)

where :math:`p` is a probability, each :math:`L_i` is a literal and each :math:`P_j` and :math:`N_j` is a conjunction of literals.
We  call each :math:`P_j`  a *P conjunction*  and  each :math:`N_k` an 
*N conjunction*.  We  call each
:math:`\exists(P_j)` a *P disjunct* and each :math:`\forall\neg(N_k)` an *N disjunct*.

The variables that occur in the body are quantified universally with
scope the PIC. The variables in the head that do not occur in the body
are quantified existentially if they occur in a P disjunct and
universally if they occur in an N disjunct, with scope the disjunct
they occur in.

An example of a PIC for the Bongard problems of :cite:`RaeLae95-ALT95`

.. _`example of a PIC`:

.. math::

  0.5\ ::\ triangle(T),square(S),in(T,S)\rightarrow \exists(circle(C),in(C,S));\forall\neg(circle(C),in(C,T))

which states
that if there is a triangle inside a square then either there exists a circle inside the square or
there doesn't exist a circle inside the triangle. This constraint
has probability 0.5.



Use
==================
The following learning algorithms are available:

- Parameter learning
- Structure learning

Input
-----
To execute the learning algorithms, prepare a Prolog file divided in five parts

- preamble
- background knowledge, i.e., knowledge valid for all interpretations
- PCLT for you which you want to learn the parameters (optional)
- language bias information
- example interpretations

The preamble must come first, the order of the other parts can be changed.

For example, consider the Bongard problems of :cite:`RaeLae95-ALT95`. 
`bongardkeys.pl <http://cplint.eu/example/pascal/bongardkeys.pl>`__ represents a Bongard problem for Pascal.

Preamble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the preamble, the Pascal library is loaded with (`bongardkeys.pl <http://cplint.eu/e/pascal/bongardkeys.pl>`__):

.. code:: prolog

  :- use_module(library(pascal)).

Now you can initialize pascal with

.. code:: prolog

	:- pascal.

At this point you can start setting parameters for Pascal such as for example


.. code:: prolog

  :-set_pascal(examples,keys(pos)).
  :-set_pascal(learning_algorithm,gradient_descent).
  :-set_pascal(learning_rate,fixed(0.5)).
  :-set_pascal(verbosity,1).

We will see later the list of available parameters.

A parameter that is particularly important for Pascal is :code:`verbosity`: if set to 1, 
nothing is printed and learning is fastest, 
if set to 3 much information is printed and learning is slowest, 2 is in between. This ends the preamble.

Background and Initial PCLT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now you can specify the background knowledge by including a set of Prolog clauses
in a section between :code:`:- begin_bg.` and :code:`:- end_bg.` 
For example

.. code:: prolog

  :- begin_bg.
  in(A,B) :- inside(A,B).
  in(A,D) :- inside(A,C),in(C,D).
  :- end_bg.

Moreover, you can specify an initial PCLT in a section between :code:`:- begin_in.` and :code:`:- end_in.`.
The initial program is used in parameter learning for providing the structure. 
In the section, facts for the predicates :code:`rule/2` or :code:`ic/1` can be given.

Facts for :code:`rule/2` take the form :code:`rule(ic,prob)` where :code:`prob` is a probability and 
:code:`ic` is a term of the form :code:`head:-body`. In it, :code:`body` is a list of literals and :code:`head`
is a list of head disjuncts. Each head disjunct is couple :code:`(sign,conjunction)` where sign is 
either :code:`(+)` for P disjuncts or :code:`(-)` for N disjuncts and :code:`conjunction` is a list of literals.

The `example of a PIC`_ above can be expressed as

.. code:: prolog

	:- begin_in.
	rule(([((+),[circle(C),in(C,S)]),((-),[circle(C),in(C,T)])]:-
	  [triangle(T),square(S),in(T,S)]),0.5).
	:- end_in.

Facts for the predicate :code:`ic/1` take the form :code:`ic(string)` where :code:`string` is a Prolog
string wher a constraint is encoded as :code:`prob::body--->head`.
:code:`body` is a conjuntion of literals where the conjunction symbol is :code:`/\ `.
:code:`head` is a disjunction where the disjunction symbol is :code:`\/`. 
Each disjunct is either a conjunction of literals, in the case of a P disjunct, or of the form
:code:`not(conjunction)` where :code:`conjunction` is a conjunction of literals. 

The `example of a PIC`_ above can be expressed as

.. code::

  :- begin_in.
  ic("0.5 :: triangle(T)/\\square(S)/\\in(T,S)
  --->
  circle(C)/\\in(C,S)
  \\/
  not(circle(C)/\\in(C,T)).").
  :- end_in.


Language Bias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The language bias part is specified by means of mode declarations in the style of `Progol <https://www.doc.ic.ac.uk/~shm/progol.html>`_. ::

	modeh(<recall>,<predicate>(<arg1>,...)).

specifies the atoms that can appear in the head of clauses, while ::

	modeb(<recall>,<predicate>(<arg1>,...)).

specifies the atoms that can appear in the body of clauses. :code:`<recall>` can be an integer or :code:`*`. 
:code:`<recall>` indicates how many atoms for the predicate specification are considered. 
:code:`*` stands for all those that are found. 
Otherwise the indicated number is randomly chosen.

Arguments of the form ::

	+<type>

specifies that the argument should be an input variable of type :code:`<type>`, i.e., a variable replacing a :code:`+<type>` argument in the head or a :code:`-<type>` argument in a preceding literal in the current hypothesized clause.

Another argument form is ::

	-<type>

for specifying that the argument should be a output variable of type :code:`<type>`. 
Any variable can replace this argument, either input or output. 
The only constraint on output variables is that those in the head of the current hypothesized clause must appear as output variables in an atom of the body.

Other forms are ::

	#<type>

for specifying an argument which should be replaced by a constant of type :code:`<type>` in the bottom clause but should not be used for replacing input variables of the following literals when building the bottom clause or ::

	-#<type>

for specifying an argument which should be replaced by a constant of type :code:`<type>` in the bottom clause and that should be used for replacing input variables of the following literals when building the bottom clause. ::

	<constant>

for specifying a constant.

An example of language bias for the Bongard domain is

.. code:: prolog

  modeh(*,triangle(+obj)).
  modeh(*,square(+obj)).
  modeh(*,circle(+obj)).
  modeh(*,in(+obj,-obj)).
  modeh(*,in(-obj,+obj)).
  modeh(*,in(+obj,+obj)).
  modeh(*,config(+obj,-#dir)).
  modeb(*,triangle(-obj)).
  modeb(*,square(-obj)).
  modeb(*,circle(-obj)).
  modeb(*,in(+obj,-obj)).
  modeb(*,in(-obj,+obj)).
  modeb(*,config(+obj,-#dir)).


Example Interpretations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The last part of the file contains the data. 
You can specify data with two modalities: models and keys. 
In the models type, you specify an example model (or interpretation or megaexample) as a list of Prolog facts initiated by :code:`begin(model(<name>)).` and terminated by :code:`end(model(<name>)).` as in

.. code:: prolog

	begin(model(2)).
	pos.
	triangle(o5).
	config(o5,up).
	square(o4).
	in(o4,o5).
	circle(o3).
	triangle(o2).
	config(o2,up).
	in(o2,o3).
	triangle(o1).
	config(o1,up).
	end(model(2)).


The facts in the interpretation are loaded in SWI-Prolog database by adding an extra initial argument equal to the name of the model. 
After each interpretation is loaded, a fact of the form :code:`int(<id>)` is asserted, where :code:`id` is the name of the interpretation. 
This can be used in order to retrieve the list of interpretations.

Alternatively, with the keys modality, you can directly write the facts and the first argument will be interpreted as a model identifier. 
The above interpretation in the keys modality is

.. code:: prolog

	pos(2).
	triangle(2,o5).
	config(2,o5,up).
	square(2,o4).
	in(2,o4,o5).
	circle(2,o3).
	triangle(2,o2).
	config(2,o2,up).
	in(2,o2,o3).
	triangle(2,o1).
	config(2,o1,up).

which is contained in the `bongardkeys.pl <http://cplint.eu/e/bongardkeys.pl>`_. 
This is also how model :code:`2` above is stored in SWI-Prolog database. 
The two modalities, models and keys, can be mixed in the same file. 
Facts for :code:`int/1` are not asserted for interpretations in the key modality but can be added by the user explicitly.

Then you must indicate how the examples are divided in folds with facts of the form: :code:`fold(<fold_name>,<list of model identifiers>)`, as for example

.. code:: prolog

	fold(train,[2,3,...]).
	fold(test,[490,491,...]).

As the input file is a Prolog program, you can define intensionally the folds as in

.. code:: prolog

	fold(all,F):-
  	findall(I,int(I),F).

:code:`fold/2` is dynamic so you can also write (`registration.pl <http://cplint.eu/e/registration.pl>`_)

.. code:: prolog

	:- fold(all,F),
		sample(4,F,FTr,FTe),
    assert(fold(rand_train,FTr)),
    assert(fold(rand_test,FTe)).


which however must be inserted after the input interpretations otherwise the facts for :code:`int/1` will not be available and the fold :code:`all` would be empty. 

Commands
--------

Parameter Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To execute parameter learning, prepare an input file as indicated above and call

.. code:: prolog

	?- induce_par_pascal(<list of folds>,T).

where :code:`<list of folds>` is a list of the folds for training and :code:`T` will contain the input theory with updated parameters.

For example `bongardkeys.pl <http://cplint.eu/e/pascal/bongardkeys.pl>`__, you can perform parameter learning on the :code:`train` fold with

.. code:: prolog

	?- induce_par_pascal([train],P).

Structure Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To execute structure learning, prepare an input file in the editor panel as indicated above and call

.. code:: prolog

	induce(+List_of_folds:list,-T:list) is det

where :code:`List_of_folds` is a list of the folds for training and :code:`T` will contain the learned PCLT.

For example `bongardkeys.pl <http://cplint.eu/e/pascal/bongardkeys.pl>`__, you can perform structure learning on the :code:`train` fold with

.. code:: prolog

	?- induce([train],P).

A PCLT can also be tested on a test set with :code:`test_pascal/7` or :code:`test_prob_pascal/6` as described below.

Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A PCLT can also be tested on a test set with

.. code:: prolog

	test_pascal(+T:list,+List_of_folds:list,-LL:float,-AUCROC:float,-ROC:list,-AUCPR:float,-PR:list) is det

or ::

	test_prob_pascal(+T:list,+List_of_folds:list,-NPos:int,-NNeg:int,-LL:float,-ExampleList:list) is det

where :code:`T` is a list of terms representing clauses and :code:`List_of_folds` is a list of folds.

:code:`test_pascal/7` returns the log likelihood of the test examples in :code:`LL`, the Area Under the ROC curve in :code:`AUCROC`, a dictionary containing the list of points (in the form of Prolog pairs :code:`x-y`) of the ROC curve in :code:`ROC`, the Area Under the PR curve in :code:`AUCPR`, a dictionary containing the list of points of the PR curve in :code:`PR`.

:code:`test_prob_pascal/6` returns the log likelihood of the test examples in :code:`LL`, the numbers of positive and negative examples in :code:`NPos` and :code:`NNeg` and the list :code:`ExampleList` containing couples :code:`Prob-Ex` where :code:`Ex` is :code:`a` for :code:`a` a positive example and :code:`\+(a)` for :code:`a` a negative example and :code:`Prob` is the probability of example :code:`a`.

Then you can draw the curves in :code:`cplint` on SWISH using C3.js using ::

	compute_areas_diagrams(+ExampleList:list,-AUCROC:float,-ROC:dict,-AUCPR:float,-PR:dict) is det

(from pack `auc.pl <http://www.swi-prolog.org/pack/list?p=auc>`_) that takes as input a list :code:`ExampleList` of pairs probability-literal of the form that is returned by :code:`test_prob_pascal/6`.

For example, to test on fold :code:`test` the program learned on fold :code:`train` you can run the query

.. code:: prolog

	?- induce_par([train],P),
  	test(P,[test],LL,AUCROC,ROC,AUCPR,PR).

Or you can test the input program on the fold :code:`test` with ::

.. code:: prolog

	?- in(P),test(P,[test],LL,AUCROC,ROC,AUCPR,PR).

In :code:`cplint` on SWISH, by including ::

.. code:: prolog

	:- use_rendering(c3).
	:- use_rendering(lpad).

in the code before :code:`:- pascal.` the curves will be shown as graphs using C3.js and the output program will be pretty printed.


Parameters for Learning
-----------------------
Parameters are set with commands of the form ::

	:- set_pascal(<parameter>,<value>).

The available parameters are:

- :code:`specialization`: (values: :code:`{bottom,mode}`, default value: :code:`bottom`, valid for SLIPCOVER) specialization mode.
- :code:`depth_bound`: (values: :code:`{true,false}`, default value: :code:`true`) if :code:`true`, the depth of the derivation of the goal is limited to the value of the :code:`depth` parameter.
- :code:`depth` (values: integer, default value: 2): depth of derivations if :code:`depth_bound` is set to :code:`true`
- :code:`single_var` (values: :code:`{true,false}`, default value: :code:`false`): if set to :code:`true`, there is a random variable for each clause, instead of a different random variable for each grounding of each clause
- :code:`epsilon_em` (values: real, default value: 0.1): if the difference in the log likelihood in two successive parameter EM iteration is smaller than :code:`epsilon_em`, then EM stops
- :code:`epsilon_em_fraction` (values: real, default value: 0.01): if the difference in the log likelihood in two successive parameter EM iteration is smaller than :code:`epsilon_em_fraction*(-current log likelihood)`, then EM stops
- :code:`iter` (values: integer, defualt value: 1): maximum number of iteration of EM parameter learning. If set to -1, no maximum number of iterations is imposed
- :code:`iterREF` (values: integer, defualt value: 1, valid for SLIPCOVER and LEMUR): maximum number of iteration of EM parameter learning for refinements. If set to -1, no maximum number of iterations is imposed.
- :code:`random_restarts_number` (values: integer, default value: 1, valid for EMBLEM, SLIPCOVER and LEMUR): number of random restarts of parameter EM learning
- :code:`random_restarts_REFnumber` (values: integer, default value: 1, valid for SLIPCOVER and LEMUR): number of random restarts of parameter EM learning for refinements
- :code:`seed` (values: seed(integer) or seed(random), default value :code:`seed(3032)`): seed for the Prolog random functions, see `SWI-Prolog manual <http://www.swi-prolog.org/pldoc/man?predicate=set_random/1>`__ 
- :code:`c_seed` (values: unsigned integer, default value 21344)): seed for the C random functions
- :code:`logzero` (values: negative real, default value :math:`\log(0.000001)`: value assigned to :math:`\log(0)`
- :code:`max_iter` (values: integer, default value: 10, valid for SLIPCOVER): number of interations of beam search
- :code:`max_var` (values: integer, default value: 4, valid for SLIPCOVER and LEMUR): maximum number of distinct variables in a clause
- :code:`beamsize` (values: integer, default value: 100, valid for SLIPCOVER): size of the beam
- :code:`megaex_bottom` (values: integer, default value: 1, valid for SLIPCOVER): number of mega-examples on which to build the bottom clauses
- :code:`initial_clauses_per_megaex` (values: integer, default value: 1, valid for SLIPCOVER): number of bottom clauses to build for each mega-example (or model or interpretation)
- :code:`d` (values: integer, default value: 1, valid for SLIPCOVER): number of saturation steps when building the bottom clause
- :code:`mcts_beamsize` (values: integer, default value: 3, valid for LEMUR): size of the Monte-Carlo tree search beam
- :code:`mcts_visits` (values: integer, default value: +1e20, valid for LEMUR): maximum number of visits
- :code:`max_iter_structure` (values: integer, default value: 10000, valid for SLIPCOVER): maximum number of theory search iterations
- :code:`mcts_max_depth` (values: integer, default value: 8, valid for LEMUR): maximum depth of default policy search
- :code:`mcts_c` (values: real, default value: 0.7, valid for LEMUR): value of parameter :math:`C` in the computation of UCT
- :code:`mcts_iter` (values: integer, default value: 20, valid for LEMUR): number of Monte-Carlo tree search iterations
- :code:`mcts_maxrestarts` (values: integer, default value: 20, valid for LEMUR): maximum number of Monte-Carlo tree search restarts
- :code:`neg_ex` (values: :code:`given`, :code:`cw`, default value: :code:`cw`): if set to :code:`given`, the negative examples in training and testing are taken from the test folds interpretations, i.e., those examples :code:`ex` stored as :code:`neg(ex)`; if set to :code:`cw`, the negative examples in training and testing are generated according to the closed world assumption, i.e., all atoms for target predicates that are not positive examples. The set of all atoms is obtained by collecting the set of constants for each type of the arguments of the target predicate, so the target predicates must have at least one fact for :code:`modeh/2` or :code:`modebb/2` also for parameter learning.
- :code:`alpha` (values: floating point :math:`\geq 0`, default value: 0): parameter of the symmetric Dirichlet distribution used to initialize the parameters. If it takes value 0, a truncated Dirichlet process is used to sample parameters: the probability of being true of each Boolean random variable used to represent multivalued random variables is sampled uniformly and independently in [0,1]. If it takes a value :math:`\geq 0`, the parameters are sampled from a symmetric Dirichlet distribution, i.e. a Dirichlet distribution with vector of parameters :math:`\alpha,\ldots,\alpha`.
- :code:`verbosity` (values: integer in [1,3], default value: 1): level of verbosity of the algorithms.



Example Files
=============
The :code:`pack/pascal/prolog/examples` folder in SWI-Prolog home contains some example programs. 
The :code:`pack/pascal/docs` folder contains this manual in latex, html and pdf.

Manual in PDF
==================
A PDF version of the manual is available at `http://friguzzi.github.io/pascal/_build/latex/pascal.pdf <http://friguzzi.github.io/pascal/_build/latex/pascal.pdf>`_.

License
=======
Pascal follows the BSD 2-Clause License that you can find in the root folder. 
The copyright is by Fabrizio Riguzzi.


References
==========
.. bibliography:: newbib.bib
