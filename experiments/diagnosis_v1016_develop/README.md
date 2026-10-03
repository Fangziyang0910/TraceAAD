# Equal development of new designs

Late in a search, is a new design behind because it is worse, or because it has not been developed? `develop.py` gives the training best and two late Explore designs (similarity to the best ≤ 0.5, chosen by a fixed rule) the same 12 Refine generations with V10.16's own prompts, evaluation and repair, each in its own copy of a completed run; `analyze.py` summarizes.

Result (2026-10-03, 10 runs, 30 arms): the best improved in 2/10 arms (tiny); new designs improved in 17/20 (median +1.44%), mostly within 3-5 generations, and closed gaps of 1-25% to within about 1%; 2/20 passed the best on training. Reading: the search judges a new design by its first version, which is far below its developed level; a few generations of development reveal that level. Whether a design then wins depends on the design itself. Data: `experiments_result/diagnosis_v1016_develop/`.
