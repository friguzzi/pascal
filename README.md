PASCAL
======


Pasccal (probabilistic inductive constraint logic) is an algorithm for learning probabilistic integrity constraints. It was 
proposed in 

Fabrizio Riguzzi, Elena Bellodi, Riccardo Zese, Marco Alberti, and Evelina Lamma. Probabilistic inductive constraint logic. Machine Learning, 110:723–754, 2021. 
[doi:10.1007/s10994-020-05911-6](https://doi.org/10.1007/s10994-020-05911-6)

It contains modules for both structure and parameter learning.


You can find the manual at [http://friguzzi.github.io/pascal/](http://friguzzi.github.io/pascal/).

You can try it online at [http://cplint.eu](http://cplint.eu).

Installation
------------
This is an [SWI-Prolog](http://www.swi-prolog.org/) pack.

It can be installed with `pack_install/1`

    $ swipl
    ?- pack_install(pascal).

Requirements
-------------
It requires the pack `lbfgs` [https://github.com/friguzzi/lbfgs](https://github.com/friguzzi/lbfgs)

It is installed automatically when installing pack `pascal` or can installed manually as

    $ swipl
    ?- pack_install(lbfgs).

You can upgrade the pack with

    $ swipl
    ?- pack_upgrade(pascal).


Example of use
---------------

    $ cd <pack>/pascal/prolog/examples
    $ swipl
    ?- [bongardkeys].
    ?- induce_pascal([train]),T).

Testing the installation
------------------------

    $ swipl
    ?- [library(test_pascal)].
    ?- test_pascal.

Support
-------

Use the Google group [https://groups.google.com/forum/#!forum/cplint](https://groups.google.com/forum/#!forum/cplint).