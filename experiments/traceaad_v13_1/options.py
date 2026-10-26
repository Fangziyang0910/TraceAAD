from traceaad.v13_1 import Config

FIELDS=dict(roots=int,init_attempt_limit=int,history_depth=int,prompt_tokens=int,max_model_calls=int,
            warmup=int,free_per_ten=int,diagnostic_per_ten=int,diagnostic_threshold=float,
            lineage_decay=float,improvement_delta=float,match_batch=int,max_conditions=int,library_path=str)


def add_arguments(parser):
    for name,kind in FIELDS.items():
        parser.add_argument('--'+name.replace('_','-'),type=kind,default=getattr(Config,name))


def values(args):
    return {name:getattr(args,name) for name in FIELDS}


def command_arguments(args):
    return [f'--{name.replace("_","-")}={getattr(args,name)}' for name in FIELDS
            if getattr(args,name) is not None]+[f'--output-tokens={args.output_tokens}']
