import hashlib
import gzip
import json

import pytest

from experiments.traceaad.harness import execute, main, read_rows


@pytest.mark.parametrize('compressed',[False,True])
def test_trace_reader_accepts_committed_rows_and_leaves_partial_tail(tmp_path,compressed):
    p=tmp_path/('attempts.jsonl.gz' if compressed else 'attempts.jsonl')
    raw=b'{"attempt":1}\n{"attempt":'
    p.write_bytes(gzip.compress(raw) if compressed else raw)
    assert list(read_rows(p))==[{'attempt':1}]


def test_trace_reader_exposes_corrupt_committed_evidence(tmp_path):
    p=tmp_path/'attempts.jsonl'
    p.write_bytes(b'{"attempt":1}\ncorrupt\n{"attempt":3}\n')
    with pytest.raises(json.JSONDecodeError):
        list(read_rows(p))


def test_replay_does_not_silently_keep_response_from_an_older_prompt(tmp_path):
    item={'arm':'candidate','case':'real_trace','prompt':'current input'}
    dest=tmp_path/'responses/candidate/real_trace_0.json'
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps({'prompt_sha256':hashlib.sha256(b'older input').hexdigest()}))
    with pytest.raises(ValueError,match='prompt identity'):
        execute((str(tmp_path),None,item,0,'unused'))


def test_missing_model_response_cannot_pass_evaluation_collection(tmp_path,monkeypatch):
    (tmp_path/'inputs_candidate.json').write_text(json.dumps([
        {'case':'real_trace','arm':'candidate','phase':'generation'}]))
    monkeypatch.setattr('sys.argv',['harness','evaluate','--out',str(tmp_path),'--arm','candidate'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==1
