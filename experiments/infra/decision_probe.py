"""Replay saved System One requests against a local SGLang server, without searching."""

import argparse
import json
import math
import time
from pathlib import Path

import requests


def check_answer(question, answer):
    """Check the actual inference result, including normalization and option identity."""
    assert answer["type"] == question["type"], answer
    if question["type"] == "noul":
        assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1, answer
    else:
        probabilities = answer["probabilities"]
        assert len(probabilities) == len(question["criteria"]), answer
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()), answer
        assert math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5), answer
        if question["type"] == "choice":
            assert set(probabilities) == set(question["criteria"]), answer
            assert probabilities[answer["choice"]] == max(probabilities.values()), answer
        else:
            assert math.isfinite(answer["score"]) and 0 <= answer["score"] <= len(probabilities) - 1, answer
    if "x_label_mass" in answer:
        # GPU log-probability rounding can put total mass slightly above one.
        assert math.isfinite(answer["x_label_mass"]) and 0 <= answer["x_label_mass"] <= 1 + 1e-6, answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True, help="JSONL: id and request per line")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; existing files are refused")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/v1/systemone"
    with args.requests.open() as source, args.output.open("x") as output, requests.Session() as session:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            started = time.perf_counter()
            response = session.post(endpoint, json=row["request"], timeout=args.timeout)
            elapsed = time.perf_counter() - started
            if not response.ok:
                raise RuntimeError(f"{row['id']}: HTTP {response.status_code}: {response.text[:2000]}")
            result = response.json()
            assert set(result["answers"]) == set(row["request"]["questions"]), result
            for name, question in row["request"]["questions"].items():
                check_answer(question, result["answers"][name])
            assert result["usage"]["output_tokens"] == 0, result["usage"]
            output.write(json.dumps({**row, "endpoint": endpoint, "elapsed_seconds": elapsed,
                                     "response": result}, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{row['id']}: {len(result['answers'])} answers, {elapsed:.3f}s, zero generated tokens", flush=True)


if __name__ == "__main__":
    main()
