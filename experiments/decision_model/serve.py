"""Serve a trained outcome model to TraceAAD with authenticated JSON requests."""

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading

from data import ACTIONS, expected_gain, expected_request_cost, fit_record, questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if not args.token_file.exists():
        args.token_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(32))
    token = args.token_file.read_text().strip()
    if len(token) < 24:
        raise ValueError("API token must contain at least 24 characters")
    metadata = json.loads(args.metadata.read_text())
    from unsloth import FastDecisionModel
    from unsloth.models.clef import encode_record
    import torch

    model, processor = FastDecisionModel.from_pretrained(args.adapter, dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit, max_seq_length=metadata["max_seq_length"])
    FastDecisionModel.for_inference(model)
    model.eval().requires_grad_(False)
    tokenizer = getattr(processor, "tokenizer", processor)
    length = lambda row: len(encode_record(tokenizer, row, max_length=1000000).input_ids)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            if self.path != "/metadata":
                return self.respond(404, {"error": "unknown endpoint"})
            self.respond(200, {"adapter": args.adapter, "score_scales": metadata["score_scales"],
                               "horizons": metadata["horizons"], "max_seq_length": metadata["max_seq_length"],
                               "decision_horizon": metadata.get("decision_horizon", 1),
                               "prompt_policy": metadata["prompt_policy"]})

        def do_POST(self):
            if not self.authorized():
                return self.respond(401, {"error": "unauthorized"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if self.path != "/score" or not 0 < size <= 2**21:
                    return self.respond(400, {"error": "invalid request size or endpoint"})
                records = json.loads(self.rfile.read(size))["records"]
                if not isinstance(records, list) or not 1 <= len(records) <= 3:
                    raise ValueError("send one to three candidate requests")
                answers = []
                with lock:
                    for state in records:
                        task, horizon, action = state["task"], state["development_budget_candidates"], state["proposed_action"]
                        if action not in ACTIONS or horizon not in metadata["horizons"]:
                            raise ValueError("unsupported action or horizon")
                        if state["generation_policy"] != f"TraceAAD V10.{metadata['prompt_policy'][3:]}":
                            raise ValueError("generation prompt policy differs from training")
                        if abs(state["score_scale"] - metadata["score_scales"][task]) > 1e-9:
                            raise ValueError("score scale differs from training")
                        row = fit_record({"state": state, "questions": questions(horizon)}, length, metadata["max_seq_length"])
                        prediction = FastDecisionModel.predict(model, processor, row["state"], row["questions"])
                        reward = expected_gain(prediction["frontier_gain"], metadata["reward_support"][str(horizon)][task])
                        cost = (expected_request_cost(prediction["repair_used"], state["remaining_candidates"])
                                if horizon == 2 else float(horizon))
                        answers.append({"action": action, "expected_gain": reward, "expected_candidate_cost": cost,
                                        "gain_per_candidate": reward / cost, "answers": prediction})
                self.respond(200, {"choices": answers})
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(422, {"error": str(exc)})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ready": True, "host": args.host, "port": args.port, "adapter": args.adapter}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
