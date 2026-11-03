import json

import pytest

from experiments.traceaad_v19_4.prepare import main, verify


def test_preparation_writes_separate_plans_without_launching(tmp_path,capsys):
    tokenizer=tmp_path/'tokenizer.json'
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    Tokenizer(WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(tokenizer))
    output=tmp_path/'prepared'
    main(['--output',str(output),'--batch-prefix','v194_test','--tokenizer',str(tokenizer),
          '--scheduler-socket','/tmp/test-scheduler.sock','--tasks','tsp_construct'])
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='prepared_not_launched'
    assert result['runs']=={'main':3,'neighborhood':3,'rank':3,'direct':3,'preflight':1}
    plans={cell:json.loads((output/'plans'/f'{cell}.json').read_text()) for cell in result['cells']}
    assert plans['main']['method_params']['selection_mode']=='trajectory'
    assert plans['neighborhood']['method_params']['selection_mode']=='neighborhood'
    assert plans['rank']['method_params']['selection_mode']=='rank'
    assert plans['direct']['method_params']['selection_mode']=='direct'
    assert plans['preflight']['method_params']['initial_size']==1
    assert plans['preflight']['method_params']['selection_mode']=='trajectory'
    assert plans['preflight']['budget_per_run']==5
    assert all('--launch' not in row['command'] for p in plans.values() for row in p['plan'])
    assert verify(output)['cells']==['main','neighborhood','rank','direct','preflight']
    (output/'assets/tokenizer.json').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        verify(output)
