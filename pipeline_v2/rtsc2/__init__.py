"""rtsc2: a compact rebuild of the train -> evaluate -> select loop.

Stages (each a plain function, cached on a hash of its inputs):

    data      labelled Materials Project table -> feature matrix, element tokens
    models    boosted-tree bag and a batched deep-ensemble network
    evaluate  the repository's standard folds, so numbers compare one-to-one
    agent     acquisition policies, scored by replaying hidden labels
    cli       one command for the whole loop
"""

__version__ = "0.1.0"
