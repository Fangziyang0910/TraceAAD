# Transfer of training progress

`evaluate.py` evaluates every program that raised a run's training best (V10.15-6 and V10.16, 565 programs) on the run's selection set; `analyze.py` compares training and selection gains along the frontier.

Result (2026-10-03): after attempt 300 the selection set keeps 58-96% of the training gain on every task (CVRP 18.1/24.7%, OBP 5.4/5.6%, OP 7.3/12.6%, TSP 14.9/22.5%, VRPTW 23.8/29.0%). Late steps are tiny (median about 0.1%) and about half go down on the selection set, but late search is not overfitting overall. Data: `experiments_result/diagnosis_v1016_frontier/`.
