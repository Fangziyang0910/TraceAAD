# Equal-budget refinement of sampled programs

How much does further refinement improve programs sampled late in a search? `develop.py` gives the training best and two late Explore programs (code similarity to the best ≤ 0.5, chosen by a fixed rule) the same 12 Refine generations with V10.16's own prompts, evaluation and repair, each in its own copy of a completed run; `analyze.py` summarizes.

Result (2026-10-03, 10 runs, 30 arms): the best improved in 2/10 arms (tiny); the sampled Explore programs improved in 17/20 (median +1.44%), mostly within 3-5 generations, and closed gaps of 1-25% to within about 1%; 2/20 passed the best on training. Reading: the first measured score of these programs was often below the score reached after a few refinements. Inspect the edits to see which computations were retained, repaired, combined or replaced, and use the resulting gains to assess further effort. Data: `experiments_result/diagnosis_v1016_develop/`.
