import pytest

from baselines.calm import CALM, CALMProfiler
from baselines.pathwise import PathWise, PathWiseProfiler
from baselines.shared.checkpoint import NAME, read_checkpoint, save_checkpoint
from baselines.shinka_evo import ShinkaEvo, ShinkaEvoProfiler
from core import LLM, Evaluation
from experiments.infra.resume_baseline import history, recover
from traceaad.storage import read_json, write_json


@pytest.fixture(autouse=True)
def toy_calm_seed(monkeypatch):
    monkeypatch.setattr(
        "baselines.calm.calm.load_seed",
        lambda task: (
            "def solve(x):\n    return 1\n",
            "The idea of the algorithm is to return one.",
        ),
    )


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def solve(x):\n    return 1\n",
            task_description="Minimize the returned value.",
            safe_evaluate=False,
        )
        self.evaluated = []

    def evaluate_program(self, code, function, **kwargs):
        self.evaluated.append(code)
        return float(function(1))


class ScriptedLLM(LLM):
    def __init__(self, values, *, stop=True):
        super().__init__()
        self.values = list(values)
        self.calls = []
        self.stop = stop

    def draw_sample(self, prompt, **kwargs):
        self.calls.append(prompt)
        text = str(prompt)
        if "You are controlling the evolutionary search" in text:
            return "PARENTS: [init_0]\nDIRECTIVE: reduce the returned value"
        if "Analyze heuristic evolution for" in text:
            return "Prefer smaller values."
        if "Phrase your hints as design principles" in text:
            return "Keep the return deterministic."
        if not self.values:
            if self.stop:
                raise KeyboardInterrupt
            pytest.fail("Search exceeded its remaining budget")
        value = self.values.pop(0)
        return (
            "{{The idea of the algorithm is to return a smaller deterministic value.}}\n"
            "Description: return a smaller value\nDerivation Rationale: minimize the objective\n"
            f"```python\ndef solve(x):\n    return {value}\n```"
        )


def make_method(name, directory, llm, *, budget=6):
    evaluation = TinyEvaluation()
    if name == "calm":
        method = CALM(
            llm,
            evaluation,
            CALMProfiler(directory),
            max_sample_nums=budget,
            task_key="tsp_construct",
            numeric_refine_variants=0,
            numeric_refine_initial_variants=0,
            population_size=2,
            n_prompts=1,
            n_generations=1,
            ub_crossover=0,
        )
    elif name == "pathwise":
        method = PathWise(
            llm,
            evaluation,
            PathWiseProfiler(directory),
            max_sample_nums=budget,
            pop_size=2,
            init_pop_size=2,
            num_actions=1,
            num_rollouts=1,
            max_inner_steps=1,
            use_prompt_perturbation=False,
        )
    else:
        method = ShinkaEvo(
            llm,
            evaluation,
            ShinkaEvoProfiler(directory),
            max_sample_nums=budget,
            num_generations=10,
            num_islands=2,
            patch_types=("full",),
            patch_type_probs=(1.0,),
            random_seed=0,
            embedding_fn=lambda code: (
                [float(f"return {-i}\n" in code) for i in range(1, 10)]
                + [float("return 1\n" in code)]
            ),
        )
    return method, evaluation


def setup_config(directory, name, budget=6):
    write_json(
        directory / "run_config.json",
        {
            "method": name,
            "task": "toy",
            "budget": budget,
            "seed": 0,
            "backend": "test",
            "run_name": "test",
        },
    )


def run_until_interrupt(method):
    try:
        method.run()
    except KeyboardInterrupt:
        pass


@pytest.mark.parametrize("name", ["calm", "pathwise", "shinka_evo"])
def test_interrupted_baseline_resumes_without_reevaluating_saved_programs(tmp_path, name):
    setup_config(tmp_path, name)
    first, _ = make_method(name, tmp_path, ScriptedLLM([-1]))
    run_until_interrupt(first)
    assert (tmp_path / NAME).exists()
    spent = history(tmp_path)[-1]
    assert spent == (1 if name == "pathwise" else 2)
    assert read_json(tmp_path / "summary.json")["status"] != "finished"

    resumed, evaluation = make_method(
        name, tmp_path, ScriptedLLM([-2, -3, -4, -5, -6, -7], stop=False)
    )
    report = recover(resumed, tmp_path)
    assert report["recovery"] == "search_checkpoint"
    assert resumed._tot_sample_nums == spent and evaluation.evaluated == []
    resumed.run()
    assert resumed._tot_sample_nums == 6
    assert len(evaluation.evaluated) == 6 - spent
    assert all("return 1\n" not in code for code in evaluation.evaluated)
    assert read_json(tmp_path / "summary.json")["status"] == "finished"


def test_failed_checkpoint_write_preserves_previous_snapshot(tmp_path, monkeypatch):
    import baselines.shared.checkpoint as checkpoint

    setup_config(tmp_path, "shinka_evo")
    method, _ = make_method("shinka_evo", tmp_path, ScriptedLLM([]))
    method._evaluate_seed()
    save_checkpoint(method, "shinka_evo", "generation")
    previous = (tmp_path / NAME).read_bytes()

    def interrupt(payload, handle, **kwargs):
        handle.write('{"partial":')
        raise RuntimeError("checkpoint interrupted")

    monkeypatch.setattr(checkpoint.json, "dump", interrupt)
    with pytest.raises(RuntimeError, match="checkpoint interrupted"):
        save_checkpoint(method, "shinka_evo", "generation")
    assert (tmp_path / NAME).read_bytes() == previous
    assert read_checkpoint(tmp_path)["budget_used"] == 1


def test_funsearch_resumes_from_recorded_profiles_without_reprofiling(tmp_path):
    from baselines.funsearch import FunSearch, FunSearchProfiler
    from baselines.funsearch.config import ProgramsDatabaseConfig

    setup_config(tmp_path, "funsearch", budget=4)

    def create(values, stop=True):
        task = TinyEvaluation()
        task._instance_execution = {"timeout_seconds": 3, "n_workers": 1}
        return FunSearch(
            ScriptedLLM(values, stop=stop), task, FunSearchProfiler(tmp_path),
            max_sample_nums=4, samples_per_prompt=1,
            database_config=ProgramsDatabaseConfig(num_islands=2),
        )

    first = create([-1])
    first.run()
    before = history(tmp_path)
    resumed = create([-2, -3], stop=False)
    report = recover(resumed, tmp_path)
    assert report["budget_used"] == 2 and report["boundary"] == "new_prompt_batch"
    assert resumed._database.best_score_per_island == first._database.best_score_per_island
    assert history(tmp_path) == before
    resumed.run()
    assert resumed._tot_sample_nums == 4
    assert read_json(tmp_path / "summary.json")["best"]["fitness"] == -3


def test_funsearch_checkpoint_preserves_reset_islands_and_next_prompts(tmp_path):
    from baselines.funsearch import FunSearch, FunSearchProfiler
    from baselines.funsearch.config import ProgramsDatabaseConfig

    setup_config(tmp_path, "funsearch", budget=4)
    def create(values):
        return FunSearch(ScriptedLLM(values), TinyEvaluation(), FunSearchProfiler(tmp_path),
                         max_sample_nums=4, samples_per_prompt=1,
                         database_config=ProgramsDatabaseConfig(num_islands=4))
    first = create([-1])
    first.run()
    assert read_checkpoint(tmp_path) is not None
    first._database.reset_islands()
    first._database._last_reset_time = first._database._clock()
    first._checkpoint_provenance = {'rng_reseeded': True, 'reconstruction': 'migration'}
    save_checkpoint(first, "funsearch", "new_prompt_batch")
    expected = [first._database.get_prompt().code for _ in range(4)]
    resumed = create([-2, -3])
    report = recover(resumed, tmp_path)
    assert report["recovery"] == "search_checkpoint" and report["budget_used"] == 2
    assert report['rng'] == 'reseeded_at_migration'
    assert resumed._database.num_resets == 1
    assert [i.num_programs for i in resumed._database.islands] == [i.num_programs for i in first._database.islands]
    assert [resumed._database.get_prompt().code for _ in range(4)] == expected
    resumed.run()
    assert resumed._tot_sample_nums == 4
    assert read_checkpoint(tmp_path)['provenance'] == first._checkpoint_provenance
