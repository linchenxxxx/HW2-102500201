#!/usr/bin/env python3
"""Audit that the report input is backed by the saved request-level records."""

import hashlib
import json
from pathlib import Path

from benchmark import COUNT, MAX_NEW_TOKENS, PREFIX_LEN, PROMPT_LEN, SAMPLING, summarize


ROOT = Path(__file__).resolve().parents[2] / "results" / "target1"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    comparison = json.loads((ROOT / "comparison.json").read_text(encoding="utf-8"))
    shared_prefix = None
    for group in ("dispersed_prefix", "shared_prefix"):
        folder = ROOT / group
        requests = read_jsonl(folder / "requests.jsonl")
        responses = read_jsonl(folder / "responses.jsonl")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        assert len(requests) == len(responses) == COUNT
        assert json.loads((folder / "flush.json").read_text())["success"]
        first_tokens = set()
        for request, response in zip(requests, responses):
            ids = request["input_ids"]
            assert len(ids) == PROMPT_LEN
            assert request["sampling_params"] == SAMPLING and request["stream"] is True
            assert response["request_id"] == request["request_id"]
            assert response["input_sha256"] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert response["success"] and response["prompt_tokens"] == PROMPT_LEN
            assert response["completion_tokens"] == MAX_NEW_TOKENS
            assert response["prefill_tokens"] == PROMPT_LEN - response["cached_tokens"]
            first_tokens.add(ids[0])
            if group == "shared_prefix":
                shared_prefix = shared_prefix or ids[:PREFIX_LEN]
                assert ids[:PREFIX_LEN] == shared_prefix
                assert response["cached_tokens"] >= PREFIX_LEN
            else:
                assert response["cached_tokens"] == 0
        if group == "dispersed_prefix":
            assert len(first_tokens) == COUNT
        else:
            assert (folder / "prewarm.json").is_file()
        computed = summarize(group, responses, summary["walltime_s"])
        for key, value in computed.items():
            if isinstance(value, float):
                assert abs(value - summary[key]) < 1e-10, (group, key)
            else:
                assert value == summary[key], (group, key)
        assert summary in comparison
        print(f"{group}: {COUNT}/{COUNT} records verified")


if __name__ == "__main__":
    main()
