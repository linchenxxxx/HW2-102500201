#!/usr/bin/env python3
"""Reproducible native /generate prefix-cache benchmark for SGLang 0.5.14."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


PROMPT_LEN = 2112
PREFIX_LEN = 2048
SUFFIX_LEN = 64
COUNT = 32
CONCURRENCY = 8
MAX_NEW_TOKENS = 16
SEED = 2026
SAMPLING = {
    "temperature": 0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "ignore_eos": True,
    "sampling_seed": SEED,
}


def make_workloads():
    rng = random.Random(SEED)
    # These are ordinary Qwen vocabulary IDs. Distinct first tokens prevent
    # cross-request prefix matches in the control group.
    common = [rng.randrange(1000, 50000) for _ in range(PREFIX_LEN)]
    shared = [common + [rng.randrange(1000, 50000) for _ in range(SUFFIX_LEN)]
              for _ in range(COUNT)]
    prewarm = common + [rng.randrange(50000, 100000) for _ in range(SUFFIX_LEN)]
    dispersed = [[1000 + i] + [rng.randrange(1000, 50000)
                              for _ in range(PROMPT_LEN - 1)] for i in range(COUNT)]
    assert all(len(row) == PROMPT_LEN for row in shared + dispersed)
    assert all(row[:PREFIX_LEN] == common for row in shared)
    assert len({row[0] for row in dispersed}) == COUNT
    return {"shared_prefix": shared, "dispersed_prefix": dispersed}, prewarm


def percentile(values, percentage):
    ordered = sorted(values)
    if not ordered:
        return None
    point = (len(ordered) - 1) * percentage / 100
    lower = int(point)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (point - lower)


async def generate(session, base_url, token_ids, request_id, group):
    payload = {"input_ids": token_ids, "sampling_params": SAMPLING, "stream": True}
    start = time.perf_counter()
    first_token_at = None
    last_event = None
    event_count = 0
    got_done = False
    error = None
    try:
        async with session.post(base_url + "/generate", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
            while True:
                line = await response.content.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    got_done = True
                    break
                event = json.loads(raw)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                event_count += 1
                last_event = event
                if first_token_at is None and event.get("meta_info", {}).get("completion_tokens", 0) > 0:
                    first_token_at = time.perf_counter()
        if not got_done:
            raise RuntimeError("SSE stream closed without [DONE]")
        if last_event is None:
            raise RuntimeError("SSE stream had no result")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    meta = (last_event or {}).get("meta_info", {})
    completion_tokens = meta.get("completion_tokens")
    prompt_tokens = meta.get("prompt_tokens")
    cached_tokens = meta.get("cached_tokens")
    ttft = (first_token_at - start) if first_token_at is not None else None
    e2e = end - start
    tpot = ((e2e - ttft) / (completion_tokens - 1)
            if ttft is not None and isinstance(completion_tokens, int) and completion_tokens > 1
            else None)
    success = (error is None and prompt_tokens == len(token_ids)
               and completion_tokens == MAX_NEW_TOKENS
               and isinstance(cached_tokens, int) and 0 <= cached_tokens <= prompt_tokens)
    if not success and error is None:
        error = f"Unexpected metadata: {meta}"
    return {
        "group": group, "request_id": request_id, "success": success,
        "error": error, "input_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens, "prefill_tokens":
        (prompt_tokens - cached_tokens) if isinstance(prompt_tokens, int) and isinstance(cached_tokens, int) else None,
        "ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e,
        "sse_events": event_count, "finish_reason": meta.get("finish_reason"),
    }


async def flush(session, base_url):
    async with session.post(base_url + "/flush_cache?timeout=30") as response:
        body = await response.text()
        if response.status != 200 or not body.startswith("Cache flushed."):
            raise RuntimeError(f"Cache flush failed: HTTP {response.status}: {body}")
        return body.strip()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def summarize(group, rows, elapsed):
    good = [row for row in rows if row["success"]]
    totals = {key: sum(row[key] for row in good) for key in
              ("prompt_tokens", "completion_tokens", "cached_tokens", "prefill_tokens")}
    summary = {
        "group": group, "request_count": len(rows), "success_count": len(good),
        "success_rate": len(good) / len(rows), "walltime_s": elapsed,
        "throughput_output_tokens_per_s": totals["completion_tokens"] / elapsed,
        "cache_hit_rate": totals["cached_tokens"] / totals["prompt_tokens"] if good else None,
        **{f"total_{key}": value for key, value in totals.items()},
    }
    for metric in ("ttft_s", "tpot_s", "e2e_s"):
        values = [row[metric] for row in good if row[metric] is not None]
        summary[f"{metric}_p50"] = percentile(values, 50)
        summary[f"{metric}_p95"] = percentile(values, 95)
    return summary


async def main(args):
    workloads, prewarm_ids = make_workloads()
    results = Path(args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    write_json(results / "run_metadata.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "sglang_version": version("sglang"), "ray_version": version("ray"),
        "gpu": gpu.stdout.strip(), "model": "Qwen/Qwen3-0.6B",
        "request_count_per_group": COUNT, "max_concurrency": CONCURRENCY,
        "prompt_tokens_per_request": PROMPT_LEN, "shared_prefix_tokens": PREFIX_LEN,
        "sampling_params": SAMPLING,
        "group_order": ["dispersed_prefix", "shared_prefix"],
    })
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Complete one short request before any measurement, then clear its KV.
        warm = await generate(session, args.base_url, [1001, 1002, 1003, 1004],
                              "service_warmup", "warmup")
        if not warm["success"]:
            raise RuntimeError(f"Service warmup failed: {warm['error']}")
        write_json(results / "service_warmup.json", warm)
        summaries = []
        for group in ("dispersed_prefix", "shared_prefix"):
            folder = results / group
            folder.mkdir(parents=True, exist_ok=True)
            flush_response = await flush(session, args.base_url)
            write_json(folder / "flush.json", {"response": flush_response, "success": True})
            if group == "shared_prefix":
                prewarm_result = await generate(session, args.base_url, prewarm_ids,
                                                "shared_prewarm", "prewarm")
                if not prewarm_result["success"]:
                    raise RuntimeError(f"Shared prefix prewarm failed: {prewarm_result['error']}")
                write_json(folder / "prewarm.json", prewarm_result)
            inputs = [{"request_id": f"{group}_{i:02d}", "input_ids": ids,
                       "sampling_params": SAMPLING, "stream": True}
                      for i, ids in enumerate(workloads[group])]
            write_jsonl(folder / "requests.jsonl", inputs)
            semaphore = asyncio.Semaphore(CONCURRENCY)

            async def limited(item):
                async with semaphore:
                    return await generate(session, args.base_url, item["input_ids"],
                                          item["request_id"], group)

            start = time.perf_counter()
            rows = await asyncio.gather(*(limited(item) for item in inputs))
            elapsed = time.perf_counter() - start
            write_jsonl(folder / "responses.jsonl", rows)
            summary = summarize(group, rows, elapsed)
            write_json(folder / "summary.json", summary)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if len(rows) != COUNT or any(not row["success"] for row in rows):
                raise RuntimeError(f"{group}: one or more measurements failed; see responses.jsonl")
    write_json(results / "comparison.json", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--results", default=str(Path(__file__).resolve().parents[2] / "results" / "target1"))
    asyncio.run(main(parser.parse_args()))
