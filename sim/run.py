#!/usr/bin/env python3
"""Run simulated matches and report the mean of (16 - place).

    python3 sim/run.py --matches 60 --workers 3
    python3 sim/run.py --bot example_bot.py --matches 60
    python3 sim/run.py --set retreat_hp=0.5 --matches 200
"""
import argparse
import ast
import importlib.util
import math
import multiprocessing
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

DEFAULT_STREAM = ROOT.parent / "kit" / "mechfight-kit" / "check" / "stream.jsonl.gz"
MAX_WORKERS = 3


def load_bot_class(path):
    """Load a Bot class from a file without starting its stdin loop."""
    source = Path(path).read_text().replace("mf.run(Bot)", "pass")
    spec = importlib.util.spec_from_loader("candidate_bot", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.Bot


def play(job):
    from engine import Sim, load_config

    seed, bot_path, overrides, stream = job
    bot_class = load_bot_class(bot_path)
    cfg = load_config(stream)

    def make_bot(hello):
        if overrides:
            from config import Params
            return bot_class(hello, Params(**overrides))
        return bot_class(hello)

    return Sim(cfg, seed, make_bot).run()


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        key, value = item.split("=", 1)
        overrides[key] = ast.literal_eval(value)
    return overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=str(ROOT / "bot.py"))
    ap.add_argument("--matches", type=int, default=30)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--stream", default=str(DEFAULT_STREAM))
    ap.add_argument("--set", action="append", dest="overrides")
    args = ap.parse_args()
    workers = max(1, min(args.workers, MAX_WORKERS))
    overrides = parse_overrides(args.overrides)
    os.nice(10)
    jobs = [(args.seed + index, args.bot, overrides, args.stream) for index in range(args.matches)]
    started = time.time()
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(play, jobs, chunksize=1)
    scores = [r["score"] for r in results]
    deviation = statistics.stdev(scores) if len(scores) > 1 else 0.0
    error = deviation / math.sqrt(len(scores))
    print(f"bot={Path(args.bot).name} overrides={overrides or '-'} matches={len(scores)} time={time.time() - started:.0f}s")
    print(f"score {statistics.mean(scores):.2f} +/- {1.96 * error:.2f}  stdev {deviation:.2f}  median {statistics.median(scores):.1f}  wins {sum(s == 15 for s in scores)}")
    print(f"kills {statistics.mean(r['kills'] for r in results):.2f}  death_t {statistics.mean(r['death_t'] for r in results):.0f}s  cubes {statistics.mean(r['cubes'] for r in results):.1f}  survived_to_clock {sum(r['survived'] for r in results)}")
    print("killers:", Counter("zone" if r["killer"] is None else "squad" for r in results if not r["survived"]).most_common())


if __name__ == "__main__":
    main()
