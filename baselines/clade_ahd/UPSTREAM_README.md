# Clade-AHD: Monte Carlo Tree Search with Clade Metaproductivity for Automatic Heuristic Design

This repository contains the implementation of **Clade-AHD**, an enhanced framework for LLM-based automatic heuristic design that integrates **Clade Metaproductivity** into Monte Carlo Tree Search (MCTS).

## Key Innovation: Clade Metaproductivity

Clade-AHD extends traditional MCTS with a novel selection mechanism inspired by hierarchical genetic concepts. Instead of evaluating individual nodes, our framework assesses the **metaproductivity of entire subtrees (clades)**, enabling:

- **Deep Exploration**: Identify "stepping stone" nodes that may have mediocre individual performance but lead to excellent descendants
- **Robust Selection**: Clade-based Thompson Sampling aggregates evidence from entire subtrees, reducing selection bias from early random fluctuations
- **Adaptive Balance**: Dynamic temperature annealing and pseudo evaluations naturally balance exploration-exploitation throughout the search process
- **Efficient Resource Allocation**: Dynamic pruning mechanism freezes unpromising subtrees, focusing computational budget on high-potential regions

### Core Mechanisms

1. **Clade Aggregation**: Recursively aggregate performance evidence from all descendants
2. **Thompson Sampling on Clades**: Sample from aggregated Beta distributions to select promising branches
3. **Pseudo Evaluations**: Stabilize sampling for sparsely-visited nodes with representative prior knowledge
4. **Adaptive Temperature (Cool-down)**: Gradually shift from exploration to exploitation as search progresses
5. **Dynamic Pruning**: Freeze nodes whose clade performance significantly lags behind the best

![Clade-AHD Framework](流程图2.pdf)

## Framework Overview

Clade-AHD employs Monte Carlo Tree Search for LLM-based heuristic evolution while preserving all generated heuristics in a tree structure. The Clade Metaproductivity mechanism enables comprehensive exploration of the heuristic function space while maintaining focus on high-performing regions.

### Workflow

1. **Selection**: Traverse the tree using Clade-based Thompson Sampling, evaluating entire subtrees rather than individual nodes
2. **Expansion**: Execute all evolutionary operators in parallel to generate diverse offspring
3. **Simulation**: Evaluate newly generated heuristics on problem instances
4. **Backpropagation**: Update Bayesian beliefs along the ancestor path with credit decay
5. **Pruning**: Periodically freeze underperforming subtrees to optimize resource allocation

## Supported Tasks

### NP-hard Combinatorial Optimization Problems

#### Step-by-step Construction Framework
- **Travelling Salesman Problem (TSP)**
- **0-1 Knapsack Problem (KP)**
- **Online Bin Packing Problem (Online BPP)**
- **Admissible Set Problem (ASP)**

#### Ant Colony Optimization (ACO)
- **TSP** (with Black-box settings)
- **Capacitated Vehicle Routing Problem (CVRP)** (with Black-box settings)
- **Multiple Knapsack Problem (MKP)** (with Black-box settings)
- **Offline Bin Packing Problem (Offline BPP)** (with Black-box settings)

#### Guided Local Search
- **Large-scale TSP**

### Other Complex Tasks
- **Bayesian Optimization (BO)**: Cost-aware Function Design in Active Learning
- **Reinforcement Learning**: Mountain Car optimization

**Total: 15 scenarios provided**

## Getting Started

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd Clade-AHD

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Set up LLM client**: Configure your LLM API in `cfg/llm_client/` (OpenAI, DeepSeek, or Zhipu)
2. **Select problem**: Choose or modify problem configuration in `cfg/problem/`
3. **Adjust hyperparameters**: Edit `cfg/config.yaml` for MCTS-AHD parameters

### Key Hyperparameters

```yaml
# Clade Metaproductivity parameters
num_pseudo_evals: 10          # Pseudo evaluations for stability
cool_down_enabled: true        # Enable adaptive temperature
cool_down_beta: 1.0           # Temperature decay rate
pruning_enabled: true         # Enable dynamic pruning
pruning_threshold: 0.3        # Performance threshold (30% of best)
pruning_min_visits: 10        # Minimum visits before pruning
backprop_lambda: 0.9          # Credit assignment decay

# Bayesian prior
bayesian_prior_alpha: 1.0
bayesian_prior_beta: 1.0
```

### Running Experiments

```bash
# Basic run
python main.py

# Specify problem
python main.py problem=tsp_aco

# Specify LLM client
python main.py llm_client=openai

# Custom configuration
python main.py problem=cvrp_aco llm_client=deepseek
```

### Running Multiple Evaluations Simultaneously

To run several evaluations in parallel:

1. Create multiple environments with copied `problems/` and `prompts/` directories
2. Copy and rename problem config in `cfg/problem/` (see `tsp_constructive_copy` for reference)
3. Change the problem name in the copied config
4. Run each instance in separate terminals or processes

### Problem-Specific Notes

- **Online BPP**: Set `max_fe = 2000` for reproducing reported results
- **ACO Black-box settings**: Set `init_pop_size = 10` for reproducing reported results
- **Bayesian Optimization**: Install `botorch` according to `requirements.txt`

## Project Structure

```
Clade-AHD/
├── main.py                    # Main entry point
├── ahd_adapter.py             # High-level MCTS-AHD adapter
├── problem_adapter.py         # Problem-specific evaluation interface
├── cfg/                       # Configuration files
│   ├── config.yaml           # Main configuration
│   ├── problem/              # Problem-specific configs
│   └── llm_client/           # LLM API configurations
├── source/                    # Core algorithms
│   ├── mcts.py               # MCTS with Clade selection
│   ├── bayesian_node.py      # Bayesian node with clade methods
│   ├── mcts_ahd.py           # MCTS-AHD main loop
│   └── score_normalizer.py  # Adaptive score normalization
├── problems/                  # Problem implementations
├── prompts/                   # Task-specific prompts
└── utils/                     # Utility functions
```

## Key Files

- **`source/mcts.py`**: Core MCTS implementation with `select_clade_thompson()` method
- **`source/bayesian_node.py`**: BayesianNode class with `get_clade_scores()` for subtree aggregation
- **`source/mcts_ahd.py`**: Main MCTS-AHD loop integrating all components
- **`ahd_adapter.py`**: High-level interface managing MCTS-AHD execution

## Extending to New Problems

Clade-AHD can be easily extended to new optimization scenarios:

1. Create evaluation script in `problems/your_problem/eval.py`
2. Define problem configuration in `cfg/problem/your_problem.yaml`
3. Design appropriate prompts in `prompts/your_problem/`
4. Implement problem-specific heuristic interface

The framework does not require high-performance seed functions and can bootstrap from simple initial heuristics.

## Output and Visualization

Results are saved in the `outputs/` directory, including:
- **Heuristic Evolution History**: Complete MCTS tree with all generated heuristics
- **Performance Logs**: Objective values and convergence curves
- **Visualizations**: Tree structure, node beliefs, and search progress
- **Best Heuristics**: Top-performing heuristic functions

Each problem directory contains a `gpt.py` file with a leading heuristic function discovered by Clade-AHD.

## Algorithm Pseudocode

Complete algorithm pseudocode suitable for academic papers is provided in:
- `algorithm_pseudocode.tex`: Modular version with sub-procedures
- `algorithm_appendix.tex`: Unified version suitable for appendix

## Technical Highlights

### vs. Traditional MCTS
- **Selection**: Clade aggregation vs. individual node UCB
- **Exploration**: Thompson Sampling with pseudo evaluations vs. fixed exploration constant
- **Adaptivity**: Dynamic temperature and pruning vs. static parameters
- **Efficiency**: Focused search through freezing vs. uniform resource allocation

### vs. Other AHD Methods
- **vs. FunSearch**: Structured evolution with credit assignment vs. flat population
- **vs. ReEvo**: Comprehensive tree exploration vs. greedy best-first
- **vs. EoH**: Bayesian belief steering vs. heuristic ranking

## Citation

If you use this code in your research, please cite:

```bibtex
@article{clade-ahd2026,
  title={Monte Carlo Tree Search with Clade Metaproductivity for Automatic Heuristic Design},
  year={2026}
}
```

## Acknowledgments

This work builds upon and extends the MCTS-AHD framework with novel Clade Metaproductivity mechanisms. We acknowledge the implementations and insights from EoH and ReEvo projects.

## License

This project is provided for academic research purposes.
