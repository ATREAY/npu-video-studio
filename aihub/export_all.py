"""Compile + profile every pipeline model on a real Snapdragon X device via AI Hub.

For each model in aihub/config.MODELS this runs:
    python -m qai_hub_models.models.<name>.export --device "<DEVICE>" --target-runtime onnx

which uploads the model, compiles it for the device, runs an on-device profiling job,
and downloads the compiled artifact. Raw logs -> benchmarks/raw/<role>.log.
A summary table is appended to benchmarks/results.md.

Run:  python aihub/export_all.py --device "Snapdragon X Elite CRD"
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import re
import subprocess
import sys

# qai_hub_models prompts interactively before cloning a model's source git repo
# (asset_loaders._query_yes_no). QAIHM_CI=1 makes it auto-accept. PYTHONUNBUFFERED
# makes the child stream its stdout line-by-line instead of block-buffering to the pipe.
CHILD_ENV = {**os.environ, "QAIHM_CI": "1", "PYTHONUNBUFFERED": "1"}

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
RAW = PROJ / "benchmarks" / "raw"
RESULTS = PROJ / "benchmarks" / "results.md"

sys.path.insert(0, str(HERE))
import config  # noqa: E402

# "Estimated inference time (ms)   : 0.4"   (one per model component)
_LAT = re.compile(r"inference time \(ms\)\s*:\s*([\d.]+)", re.I)
# "Compute Unit(s) : npu (145 ops) gpu (0 ops) cpu (0 ops)"
_CU = re.compile(r"npu \((\d+) ops\) gpu \((\d+) ops\) cpu \((\d+) ops\)", re.I)
_JOB = re.compile(r"(https://\S*aihub\S*\.qualcomm\.com/\S+)", re.I)


def parse_metrics(out: str) -> dict:
    """Sum component latencies; aggregate NPU op fraction across components."""
    lats = [float(x) for x in _LAT.findall(out)]
    cus = [(int(n), int(g), int(c)) for n, g, c in _CU.findall(out)]
    total_lat = round(sum(lats), 2) if lats else None
    if cus:
        npu = sum(n for n, _, _ in cus)
        allops = sum(n + g + c for n, g, c in cus)
        npu_frac = round(100.0 * npu / allops, 1) if allops else None
        fallback = sum(c for _, _, c in cus)
    else:
        npu_frac, fallback = None, None
    return {
        "latency_ms": total_lat,
        "components": len(lats),
        "npu_pct": npu_frac,
        "cpu_ops": fallback,
        "job_urls": list(dict.fromkeys(_JOB.findall(out))),
    }


def run_one(role: str, module: str, device: str, runtime: str,
            precision: str = "float") -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    log_path = RAW / f"{role}.{device.replace(' ', '_')}.{precision}.log"
    cmd = [
        sys.executable, "-u", "-m", f"qai_hub_models.models.{module}.export",
        "--device", device,
        "--target-runtime", runtime,       # onnx -> ONNX Runtime + QNN EP on Windows-on-ARM
    ]
    if precision == "w8a8":
        cmd += ["--quantize", "w8a8"]
    else:
        cmd += ["--precision", "float"]
    print(f"\n=== {role}: {' '.join(cmd)}", flush=True)
    # stream live to the terminal AND tee to the log; stdin=DEVNULL so any
    # unexpected prompt fails fast instead of hanging forever.
    lines: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, text=True, bufsize=1,
                          env=CHILD_ENV) as p, log_path.open("w") as lf:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(f"    [{role}] {line}")
            sys.stdout.flush()
            lf.write(line)
            lf.flush()
            lines.append(line)
        rc = p.wait()
    out = "".join(lines)

    m = parse_metrics(out)
    row = {
        "role": role,
        "module": module,
        "ok": rc == 0 and m["latency_ms"] is not None,
        "latency_ms": m["latency_ms"],
        "components": m["components"],
        "npu_pct": m["npu_pct"],
        "cpu_ops": m["cpu_ops"],
        "job_urls": m["job_urls"],
        "log": str(log_path.relative_to(PROJ)),
    }
    status = "OK" if row["ok"] else f"FAILED (rc={rc})"
    print(f"    {status}  latency={row['latency_ms']}ms ({m['components']} comp)  "
          f"npu={row['npu_pct']}%  cpu_ops={row['cpu_ops']}  -> {row['log']}")
    return row


def write_results(device: str, rows: list[dict], precision: str) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    oks = [r["latency_ms"] for r in rows if r["ok"] and r["latency_ms"] is not None]
    total = sum(oks) if oks else None
    complete = len(oks) == len(rows)

    lines = [
        f"\n## Run {stamp} — `{device}` — runtime `{config.TARGET_RUNTIME}` — precision `{precision}`\n",
        "| Stage | Model | Latency (ms) | Comp | NPU % | CPU ops | Status |",
        "|-------|-------|-------------:|-----:|------:|--------:|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['role']} | `{r['module']}` | {r['latency_ms']} | {r['components']} | "
            f"{r['npu_pct']} | {r['cpu_ops']} | {'ok' if r['ok'] else 'FAIL'} |"
        )
    budget = config.FRAME_BUDGET_MS
    if total is None:
        verdict = f"n/a — {len(oks)}/{len(rows)} stages OK"
        total_str = "n/a"
    else:
        headroom = budget - total
        verdict = (f"**{total:.2f} ms** vs {budget:.0f} ms budget → "
                   f"{'PASS' if total <= budget else 'OVER BUDGET'} "
                   f"({headroom:+.1f} ms headroom)")
        if not complete:
            verdict += f"  [partial: {len(oks)}/{len(rows)} stages]"
        total_str = f"{total:.2f} ms"
    lines.append(f"\nSum of stage latencies: {verdict}\n")

    # job URLs for the write-up
    for r in rows:
        if r["job_urls"]:
            lines.append(f"- {r['role']} jobs: " + " ".join(r["job_urls"]))
    with RESULTS.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote summary ({total_str}) -> {RESULTS.relative_to(PROJ)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--runtime", default=config.TARGET_RUNTIME)
    ap.add_argument("--precision", default="float", choices=["float", "w8a8"],
                    help="float baseline, or w8a8 INT8 quantized")
    ap.add_argument("--only", nargs="*", help="subset of roles, e.g. --only superres segment")
    args = ap.parse_args()

    items = list(config.MODELS.items())
    if args.only:
        items = [(k, v) for k, v in items if k in args.only]

    print(f"device    : {args.device}")
    print(f"runtime   : {args.runtime}")
    print(f"precision : {args.precision}")
    print(f"models    : {[k for k, _ in items]}")

    rows = [run_one(role, module, args.device, args.runtime, args.precision)
            for role, module in items]
    write_results(args.device, rows, args.precision)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
