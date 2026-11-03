"""Execution limits owned by this version."""

INSTANCE_SECONDS = 120
FUNCTION_SECONDS = 2
FUNCTION_LIMITS = {
    "tsp_construct": 2, "cvrp_aco": 2, "fssp_gls": 2,
    "graph_colouring": 10, "jssp_construct": 10, "op_aco": 2,
}
INSTANCE_LIMITS = {task: 120 for task in FUNCTION_LIMITS}
INSTANCE_MEMORY_BYTES = 2 << 30
