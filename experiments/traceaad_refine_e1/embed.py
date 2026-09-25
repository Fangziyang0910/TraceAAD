"""CPU-only ONNX MiniLM baseline, pooling every code/summary token in chunks."""
import ast
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

OUT=Path(__file__).resolve().parent/'raw/refine_e1_20260907'
MODEL='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def main():
    start=time.time()
    tokenizer=Tokenizer.from_file(hf_hub_download(MODEL,'tokenizer.json',revision=REVISION))
    tokenizer.no_truncation();tokenizer.no_padding()
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    model=ort.InferenceSession(hf_hub_download(MODEL,'onnx/model.onnx',revision=REVISION),sess_options=options,providers=['CPUExecutionProvider'])
    texts={};text_ids={};runs=[]
    for run in json.loads((OUT/'snapshot.json').read_text())['runs']:
        folder=OUT/'snapshot'/run['run_name']
        nodes=json.loads((folder/'tree_state.json').read_text())['nodes']
        events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        parents={e['parent_id'] for e in events if e['requested_operator']==e['operator']=='Refine'}
        keys=[];ids=[]
        for n in nodes:
            if n['id'] not in parents:continue
            code=ast.unparse(ast.parse(n['code']))
            views=(n.get('idea','').strip(),code)
            if views not in text_ids:
                key=f'text_{len(text_ids):06d}'
                text_ids[views]=key;texts[key]=list(views)
            keys.append(text_ids[views]);ids.append(n['id'])
        runs.append((run['run_name'],ids,keys))
    cache=OUT/'embeddings';cache.mkdir(exist_ok=True)
    total_chunks=0
    for index,(key,views) in enumerate(texts.items()):
        path=cache/(key+'.npy')
        if path.exists():continue
        vectors=[]
        for text in views:
            if not text:continue
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            chunks=[ids[i:i+240] for i in range(0,len(ids),240)]
            summed=np.zeros(384,dtype=np.float64);weight=0
            for offset in range(0,len(chunks),32):
                batch=chunks[offset:offset+32];length=max(map(len,batch))+2
                tokens=np.zeros((len(batch),length),dtype=np.int64);mask=np.zeros_like(tokens)
                for i,chunk in enumerate(batch):
                    seq=[101]+chunk+[102];tokens[i,:len(seq)]=seq;mask[i,:len(seq)]=1
                feed={'input_ids':tokens,'attention_mask':mask,'token_type_ids':np.zeros_like(tokens)}
                outputs=model.run(None,{x.name:feed[x.name] for x in model.get_inputs()})
                states=outputs[0]
                pooled=(states*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                lens=np.array([len(x) for x in batch]);summed+=(pooled*lens[:,None]).sum(0);weight+=lens.sum()
                total_chunks+=len(batch)
            v=summed/max(weight,1);vectors.append(v/max(np.linalg.norm(v),1e-12))
        vec=np.mean(vectors,axis=0);vec/=max(np.linalg.norm(vec),1e-12)
        np.save(path,vec.astype(np.float32))
        if index%50==0:print('embedded',index+1,'/',len(texts),'chunks',total_chunks,'seconds',round(time.time()-start),flush=True)
    for name,ids,keys in runs:
        folder=OUT/'distances'/name;folder.mkdir(parents=True,exist_ok=True)
        vectors=np.stack([np.load(cache/(key+'.npy')) for key in keys])
        distance=np.clip(1-vectors@vectors.T,0,2);np.fill_diagonal(distance,0)
        np.save(folder/'embedding.npy',distance)
        (folder/'embedding_ids.json').write_text(json.dumps(ids))
    (OUT/'embedding_metadata.json').write_text(json.dumps(dict(model=MODEL,revision=REVISION,provider='CPUExecutionProvider',
        chunk_tokens=240,pooling='token-weighted chunk means, normalize each view, equal code/summary weights, normalize',
        unique_texts=len(texts),chunks_computed=total_chunks,seconds=time.time()-start,onnxruntime=ort.__version__),indent=2))

if __name__=='__main__':main()
