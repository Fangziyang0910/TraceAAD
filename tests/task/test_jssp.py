"""Check dispatch decisions against explicit schedules and independent constraints."""
import numpy as np
import pytest
from benchmarks.jssp_construct.evaluation import solve, validate_schedule
from benchmarks.jssp_construct import JSSPEvaluation
from benchmarks._seeded_data import generate_dataset, digest_arrays
from benchmarks.jssp_construct import dataset
from core.evaluate import InvalidEvaluationResult


def tiny():
    return {'processing_times': np.array([[3, 2], [2, 4]]),
            'machine_order': np.array([[0, 1], [1, 0]])}


def mwkr(t, machines, next_op, jobs, ready, candidates):
    return np.array([t[j, k:].sum() for j, k in candidates])


def test_conflict_set_uses_operation_indices_and_scheduler_respects_priority():
    data=tiny();seen=[]
    def choose_later_job(t, machines, next_op, jobs, ready, candidates):
        seen.append(candidates.copy())
        assert np.all(candidates[:,1] == next_op[candidates[:,0]])
        return candidates[:,0]
    starts, cost=solve(data, choose_later_job)
    assert np.array_equal(seen[0], [[1,0]])
    assert np.array_equal(seen[1], [[0,0],[1,1]])
    assert np.array_equal(starts, [[6,9],[0,2]])
    assert cost == 11
    better, best_cost=solve(data, mwkr)
    assert np.array_equal(better, [[0,3],[0,3]])
    assert best_cost == 7  # machine 0 requires exactly seven time units


@pytest.mark.parametrize('output',[None, np.array([np.nan]), np.array([[1.]]),np.ones(99)])
def test_dispatch_rejects_scores_outside_candidate_contract(output):
    with pytest.raises(InvalidEvaluationResult):
        solve(tiny(), lambda *args: output)


def test_independent_validator_rejects_precedence_and_machine_overlap():
    d=tiny()
    with pytest.raises(ValueError,match='precedence'):
        validate_schedule(d['processing_times'],d['machine_order'],np.array([[0,1],[0,3]]))
    with pytest.raises(ValueError,match='overlaps'):
        validate_schedule(d['processing_times'],d['machine_order'],np.array([[0,3],[0,2]]))


def test_input_mutation_does_not_change_the_instance_or_scheduler():
    d=tiny();before=digest_arrays(d)
    def mutate(t,m,n,j,r,c):
        ranking=mwkr(t,m,n,j,r,c);t[:]=0;m[:]=0;n[:]=999;j[:]=999;r[:]=999;c[:]=-1
        return ranking
    assert solve(d,mutate)[1] == 7
    assert digest_arrays(d)==before


def test_seeded_instance_reconstruction_and_split_independence():
    a,train=generate_dataset(dataset,'train',limit=2)
    b,repeated=generate_dataset(dataset,'train',limit=2)
    _,test=generate_dataset(dataset,'test',limit=2)
    assert a==b
    assert [digest_arrays(x)for x in train] == [digest_arrays(x)for x in repeated]
    assert not ({digest_arrays(x)for x in train} & {digest_arrays(x)for x in test})
    assert all(np.array_equal(np.sort(x['machine_order'],axis=1),np.tile(np.arange(20),(20,1)))for x in train)


def test_real_evaluation_recomputes_objective_and_has_zero_template_deviation():
    evaluation=JSSPEvaluation(limit=2,safe_evaluate=False)
    assert evaluation.evaluate_program(evaluation.TEMPLATE,None) == 0
    assert evaluation.n_instance==2 and evaluation.outer_settings=={}
