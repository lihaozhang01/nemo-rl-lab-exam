#!/usr/bin/env python
"""Poll NeMo Lab validation accuracy for a submitted job."""
from __future__ import annotations

import argparse
import time

from nemo_rl_lab.cli_login import job_overview_via_server


def _print_validations(job_id: str) -> int:
    overview = job_overview_via_server(job_id)
    status = overview.get("status") or overview.get("state") or "?"
    vals = overview.get("validations") or []
    print(f"job={job_id} status={status} validations={len(vals)}")
    if not vals:
        print("no validation points yet")
        return 0
    print("step\taccuracy\tavg_reward\tsamples")
    for item in vals:
        step = item.get("step", "?")
        accuracy = item.get("accuracy")
        avg_reward = item.get("avg_reward")
        samples = item.get("sample_count", "?")
        acc_text = "?" if accuracy is None else f"{float(accuracy):.4f}"
        reward_text = "?" if avg_reward is None else f"{float(avg_reward):.4f}"
        print(f"{step}\t{acc_text}\t{reward_text}\t{samples}")
    return len(vals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--interval", type=float, default=60.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    seen = -1
    while True:
        count = _print_validations(args.job_id)
        if args.once:
            return 0
        if count == seen:
            print(f"waiting {args.interval:g}s for new validation point...")
        seen = count
        print("", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
