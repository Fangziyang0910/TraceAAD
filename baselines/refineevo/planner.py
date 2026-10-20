"""RefineEvo 7de015d5 planner_interface.py: strategy decisions and operator refinement.

Model calls go through the search's `ask(messages, stage)`, which logs them and returns None on a
failed call. Local adaptations are documented in README.md.
"""

import json

import numpy as np

from .utils import extract_json_block, messages, text


class Planner:
    # Operator classification
    EXPLORATION_OPERATORS = ["op1", "op2"]  # Exploration: Create new algorithms
    EXPLOITATION_OPERATORS = ["op3", "op4", "op5"]  # Exploitation: Improve existing algorithms

    def __init__(self, ask, operator_descriptions):
        self._ask = ask
        # History records
        self.history = {
            "decisions": [],  # Decision history
            "population_stats": [],  # Population statistics history
            "operator_performance": {op: [] for op in self.EXPLORATION_OPERATORS + self.EXPLOITATION_OPERATORS}
        }
        # Upstream gives the planner a copy of the evolver's descriptions.
        self.operator_descriptions = dict(operator_descriptions)
        # Continuous exploitation threshold: if last k rounds all used "exploitation", force "exploration" this round
        self.explore_after_k_exploit = 5
        # Why the last decision fell back to the heuristic rule; None when the LLM decided.
        self.fallback = None

    def analyze_population(self, population):
        """
        Analyze population state and compute key metrics
        """

        # Extract valid objective values
        objectives = [ind["objective"] for ind in population if ind["objective"] is not None]
        objectives = np.array(objectives)

        # Basic statistics
        stats = {
            "size": len(population),
            "mean_objective": float(np.mean(objectives)),
            "std_objective": float(np.std(objectives)),
            "best_objective": float(np.min(objectives)),
            "worst_objective": float(np.max(objectives)),
        }

        # Diversity: normalized standard deviation
        obj_range = stats["worst_objective"] - stats["best_objective"]
        if obj_range > 1e-9:
            stats["diversity"] = float(stats["std_objective"] / obj_range)
        else:
            stats["diversity"] = 0.0

        # Convergence rate: computed based on historical data
        if len(self.history["population_stats"]) > 0:
            prev_best = self.history["population_stats"][-1]["best_objective"]
            if prev_best is not None and prev_best != 0:
                improvement = (prev_best - stats["best_objective"]) / abs(prev_best)
                stats["convergence_rate"] = float(improvement)
            else:
                stats["convergence_rate"] = 0.0
        else:
            stats["convergence_rate"] = 0.0

        # Improvement potential: based on diversity and convergence rate
        stats["improvement_potential"] = float(
            0.6 * stats["diversity"] + 0.4 * (1.0 if stats["convergence_rate"] > 0.01 else 0.5)
        )

        return stats

    def decide_strategy(self, population, generation):
        """
        Decide whether to explore or exploit in current round
        """
        # Analyze population state
        stats = self.analyze_population(population)
        self.history["population_stats"].append(stats)

        # Call LLM for decision
        decision = self._query_llm_for_strategy(stats, generation)
        # If last k rounds all used "exploitation", force "exploration" this round
        k = self.explore_after_k_exploit
        prev = self.history.get("decisions", [])
        if k > 0 and len(prev) >= k:
            last_k = prev[-k:]
            if all(d.get("decision", {}).get("strategy") == "exploitation" for d in last_k):
                decision = {
                    "strategy": "exploration",
                    "reasoning": f"Auto-switch due to {k} consecutive exploitation",
                    "selected_operators": self.EXPLORATION_OPERATORS,
                    "confidence": 0.7,
                }

        # Record decision
        self.history["decisions"].append({
            "generation": generation,
            "stats": stats,
            "decision": decision
        })

        return decision

    def _build_pop_stats(self, stats, generation):
        """Build context information for LLM decision-making"""
        context_parts = [
            f"## Generation {generation} Population Statistics:",
            f"- Population Size: {stats['size']}",
            f"- Best Objective: {stats['best_objective']:.6f}" if stats['best_objective'] is not None else "- Best Objective: N/A",
            f"- Mean Objective: {stats['mean_objective']:.6f}" if stats['mean_objective'] is not None else "- Mean Objective: N/A",
            f"- Std Deviation: {stats['std_objective']:.6f}" if stats['std_objective'] is not None else "- Std Deviation: N/A",
            f"- Diversity Score: {stats['diversity']:.4f}",
            f"- Convergence Rate: {stats['convergence_rate']:.4f}",
            f"- Improvement Potential: {stats['improvement_potential']:.4f}",
            f"",
        ]

        # Add historical trend
        if len(self.history["population_stats"]) > 1:
            context_parts.append("## Historical Trend (last 3 generations):")
            for hist_stat in self.history["population_stats"][-3:]:
                context_parts.append(
                    f"- Best: {hist_stat['best_objective']:.6f}, "
                    f"Diversity: {hist_stat['diversity']:.4f}"
                )
            context_parts.append("")

        return "\n".join(context_parts)

    def _build_operator_context(self):
        """Build operator-related context information"""
        context_parts = []
        context_parts.append("- Exploration Operators (Create new search directions):")
        for op in self.EXPLORATION_OPERATORS:
            desc = self.operator_descriptions.get(op, "No description available")
            context_parts.append(f"- **{op}**: {desc}")
        context_parts.append("")

        context_parts.append("- Exploitation Operators (Refine existing solutions):")
        for op in self.EXPLOITATION_OPERATORS:
            desc = self.operator_descriptions.get(op, "No description available")
            context_parts.append(f"- **{op}**: {desc}")
        context_parts.append("")

        return "\n".join(context_parts)

    def _query_llm_for_strategy(self, stats, generation):
        """
        Call LLM to make exploration/exploitation decision
        """
        self.fallback = None
        pop_stats = self._build_pop_stats(stats, generation)
        operator_context = self._build_operator_context()

        user_prompt = text("user_sel_ops").format(
            pop_stats=pop_stats,
            ops_desc=operator_context,
            exploration_ops=self.EXPLORATION_OPERATORS,
            exploitation_ops=self.EXPLOITATION_OPERATORS
        )
        response = self._ask(messages(text("system_planner"), user_prompt), "planner")
        if response is None:
            return self._heuristic_decision("call_failed")

        # Upstream lets a malformed reply end the run; here it takes the upstream heuristic fallback.
        try:
            # Parse JSON response
            json_text = extract_json_block(response)
            decision_list = json.loads(json_text)
            decision = decision_list[0]

            # Validate decision format
            if "strategy" not in decision or "selected_operators" not in decision:
                return self._heuristic_decision("missing_fields")

            # Ensure selected operators are valid
            valid_ops = self.EXPLORATION_OPERATORS + self.EXPLOITATION_OPERATORS
            decision["selected_operators"] = [
                op for op in decision["selected_operators"] if op in valid_ops
            ]
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return self._heuristic_decision("unparsed_reply")

        if not decision["selected_operators"]:
            return self._heuristic_decision("no_valid_operator")

        return decision

    def _heuristic_decision(self, reason):
        """Heuristic rule-based decision (fallback solution)"""
        self.fallback = reason
        if not self.history["population_stats"]:
            return {
                "strategy": "exploration",
                "reasoning": "Initial generation, need exploration",
                "selected_operators": ["op1", "op2"],
                "confidence": 0.8
            }

        stats = self.history["population_stats"][-1]
        diversity = stats["diversity"]
        convergence = stats["convergence_rate"]

        # Simple rules:
        # - Low diversity or slow convergence -> Exploration
        # - High diversity and fast convergence -> Exploitation
        if diversity < 0.4 or convergence < 0.03:
            return {
                "strategy": "exploration",
                "reasoning": f"Low diversity ({diversity:.3f}) or slow convergence ({convergence:.3f})",
                "selected_operators": ["op1", "op2"],
                "confidence": 0.7
            }
        else:
            return {
                "strategy": "exploitation",
                "reasoning": f"High diversity ({diversity:.3f}) and good convergence ({convergence:.3f})",
                "selected_operators": ["op3", "op4"],
                "confidence": 0.7
            }

    def improve_operator(self, operator_id, population):
        """
        Dynamically improve operator description based on population characteristics and performance history
        """
        if operator_id not in (self.EXPLORATION_OPERATORS + self.EXPLOITATION_OPERATORS):
            return None

        # Analyze population characteristics
        stats = self.analyze_population(population)

        # Build improvement context
        context_dict = self._build_improvement_context(operator_id, stats, population)

        # Call LLM to improve operator
        improvement_result = self._query_llm_for_improvement(operator_id, context_dict)

        if improvement_result is None:
            return None

        # Update operator description
        self.operator_descriptions[operator_id] = improvement_result["new_description"]

        return {
            "operator": operator_id,
            "improvement_type": improvement_result["improvement_type"],
            "new_description": improvement_result["new_description"],
            "reasoning": improvement_result["reasoning"],
            "confidence": improvement_result["confidence"],
            "population_context": stats
        }

    @staticmethod
    def _performance_lines(recent_history):
        history_lines = []
        for i, perf in enumerate(recent_history):
            gen = perf.get("generation", i)
            mean = perf.get("mean")
            best = perf.get("best")
            success_rate = perf.get("success_rate", 0.0)

            if mean is not None:
                history_lines.append(
                    f"Gen {gen}: Mean={mean:.6f}, Best={best:.6f}, Success Rate={success_rate:.2%}"
                )
            else:
                history_lines.append(
                    f"Gen {gen}: No valid offspring (Success Rate={success_rate:.2%})"
                )

        return "\n".join(history_lines) if history_lines else "No performance history available"

    @staticmethod
    def _performance_status(recent_history):
        if len(recent_history) >= 3:
            recent_means = [p["mean"] for p in recent_history[-3:] if p["mean"] is not None]
            if len(recent_means) >= 2:
                if recent_means[-1] < recent_means[0]:
                    return "improving performance"
                elif recent_means[-1] > recent_means[0] * 1.05:
                    return "degrading performance"
                else:
                    return "stagnating performance"
            return "insufficient valid results"
        return "insufficient history"

    @staticmethod
    def _population_context(stats):
        return "\n".join([
            f"Current Population Size: {stats['size']}",
            f"Best Objective: {stats['best_objective']:.6f}" if stats['best_objective'] is not None else "Best Objective: N/A",
            f"Mean Objective: {stats['mean_objective']:.6f}" if stats['mean_objective'] is not None else "Mean Objective: N/A",
            f"Diversity Score: {stats['diversity']:.4f}",
            f"Convergence Rate: {stats['convergence_rate']:.4f}",
        ])

    def _build_improvement_context(self, operator_id, stats, population):
        """Build context for operator improvement"""
        # Collect characteristics of best performing individuals in population
        valid_pop = [ind for ind in population if ind["objective"] is not None]
        valid_pop.sort(key=lambda x: x["objective"])
        top_individuals = valid_pop[:5] if len(valid_pop) >= 5 else valid_pop

        # Build performance history string
        perf_history = self.history["operator_performance"].get(operator_id, [])
        recent_history = perf_history[-10:] if len(perf_history) > 10 else perf_history

        # Build top algorithms string
        top_algo_parts = []
        for i, ind in enumerate(top_individuals, 1):
            algo_desc = ind.get("algorithm", "No description")
            # Truncate algorithm description to first 150 characters
            algo_summary = algo_desc[:150] + "..." if len(algo_desc) > 150 else algo_desc
            top_algo_parts.append(f"**Algorithm {i}** (Objective: {ind['objective']:.6f})")
            top_algo_parts.append(f"{algo_summary}")
            top_algo_parts.append("")

        top_algorithms = "\n".join(top_algo_parts) if top_algo_parts else "No successful algorithms yet"

        return {
            "performance_history": self._performance_lines(recent_history),
            "performance_status": self._performance_status(recent_history),
            "history_length": len(recent_history),
            "population_context": self._population_context(stats),
            "top_algorithms": top_algorithms
        }

    def _query_llm_for_improvement(self, operator_id, context_dict):
        """Call LLM to improve operator description"""

        current_desc = self.operator_descriptions.get(operator_id, "")
        if not current_desc:
            return None

        user_prompt = text("user_improve_operator").format(
            operator_id=operator_id,
            current_description=current_desc,
            performance_status=context_dict["performance_status"],
            history_length=context_dict["history_length"],
            performance_history=context_dict["performance_history"],
            population_context=context_dict["population_context"],
            top_algorithms=context_dict["top_algorithms"]
        )
        response = self._ask(messages(text("system_operator_improver"), user_prompt), "operator_improve")
        if response is None:
            return None
        try:
            # Extract JSON response
            json_text = extract_json_block(response)
            improvement_result = json.loads(json_text)

            # Validate response format
            required_fields = ["improvement_type", "reasoning", "new_description", "confidence"]
            if not all(field in improvement_result for field in required_fields):
                return None
            # Upstream logs new_description[:100] inside this try; a non-text description fails here.
            improvement_result["new_description"][:100]
            return improvement_result
        except Exception:
            return None

    def update_operator_performance(self, operator_id, offsprings, generation=None):
        """
        Update operator performance statistics
        """
        if operator_id not in self.history["operator_performance"]:
            return

        # Compute performance metrics for this operator
        valid_objs = [off["objective"] for off in offsprings if off["objective"] is not None]

        if valid_objs:
            performance = {
                "generation": generation,
                "mean": float(np.mean(valid_objs)),
                "best": float(np.min(valid_objs)),
                "worst": float(np.max(valid_objs)),
                "std": float(np.std(valid_objs)),
                "success_rate": len(valid_objs) / len(offsprings) if offsprings else 0.0,
                "num_valid": len(valid_objs),
                "num_total": len(offsprings)
            }
        else:
            performance = {
                "generation": generation,
                "mean": None,
                "best": None,
                "worst": None,
                "std": None,
                "success_rate": 0.0,
                "num_valid": 0,
                "num_total": len(offsprings)
            }

        self.history["operator_performance"][operator_id].append(performance)

    def _query_llm_for_operator_check(self, operator_id, perf_history, population=None):
        """
        Call LLM to determine if operator needs improvement
        """
        current_desc = self.operator_descriptions.get(operator_id, "No description available")

        # Build performance history string
        recent_history = perf_history[-10:] if len(perf_history) > 10 else perf_history
        if population is not None:
            population_context = self._population_context(self.analyze_population(population))
        else:
            population_context = "Population context not available"

        # Build user prompt
        user_prompt = text("user_check_operator").format(
            operator_id=operator_id,
            current_description=current_desc,
            performance_history=self._performance_lines(recent_history),
            performance_status=self._performance_status(recent_history),
            history_length=len(recent_history),
            population_context=population_context
        )
        response = self._ask(messages(text("system_operator_evaluator"), user_prompt), "operator_check")
        if response is None:
            return None
        try:
            # Extract JSON response
            json_text = extract_json_block(response)
            decision = json.loads(json_text)

            # Validate response format
            required_fields = ["needs_improvement", "reasoning", "confidence", "improvement_urgency"]
            if not all(field in decision for field in required_fields):
                return None

            return decision
        except Exception:
            return None

    def check_operator_needs_improvement(self, operator_id, population=None):
        """
        Check if operator needs improvement; upstream always asks the LLM (use_heuristic=False).
        """
        if operator_id not in self.history["operator_performance"]:
            return False, "No performance history available"

        perf_history = self.history["operator_performance"][operator_id]

        # Need at least 3 records to make judgment
        if len(perf_history) < 3:
            return False, "Insufficient performance history (need at least 3 generations)"

        # Call LLM for decision
        decision = self._query_llm_for_operator_check(operator_id, perf_history, population)

        if decision is None:
            return False, "LLM evaluation failed"

        needs_improvement = decision.get("needs_improvement", False)
        reasoning = decision.get("reasoning", "")

        if needs_improvement:
            return True, reasoning
        else:
            return False, reasoning

    def get_summary(self):
        """Get execution summary of planner agent"""
        counts = {"exploration": 0, "exploitation": 0}
        usage = {}
        for decision in self.history["decisions"]:
            strategy = str(decision["decision"].get("strategy"))
            counts[strategy] = counts.get(strategy, 0) + 1
            for op in decision["decision"].get("selected_operators", []):
                usage[op] = usage.get(op, 0) + 1
        return {"total_decisions": len(self.history["decisions"]),
                "strategy_distribution": counts, "operator_usage": usage}
