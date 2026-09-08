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
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
RAW = PROJ / "benchmarks" / "raw"
RESULTS = PROJ / "benchmarks" / "results.md"

sys.path.insert(0, str(HERE))
import config  # noqa: E402

# lines like "Estimated inference time (ms): 4.2" / "NPU (Compute Units): 100 %"
_LAT = re.compile(r"inference time.*?:\s*([\d.]+)", re.I)
_NPU = re.compile(r"NPU.*?:\s*([\d.]+)", re.I)
_JOB = re.compile(r"(https://\S*aihub\.qualcomm\.com/\S+)", re.I)


def run_one(role: str, module: str, device: str, runtime: str,
            precision: str = "float") -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    tag = precision.replace("a", "a").replace("w8a8", "w8a8")
    log_path = RAW / f"{role}.{device.replace(' ', '_')}.{precision}.log"
    cmd = [
        sys.executable, "-m", f"qai_hub_models.models.{module}.export",
        "--device", device,
        "--target-runtime", runtime,       # onnx -> ONNX Runtime + QNN EP on Windows-on-ARM
    ]
    if precision == "w8a8":
        cmd += ["--quantize", "w8a8"]
    else:
        cmd += ["--precision", "float"]
    print(f"\n=== {role}: {' '.join(cmd)}", flush=True)
    # stream live to the terminal AND tee to the log (AI Hub jobs take minutes;
    # a silent capture looks hung)
    lines: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p, log_path.open("w") as lf:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(f"    [{role}] {line}")
            sys.stdout.flush()
            lf.write(line)
            lines.append(line)
        rc = p.wait()
    out = "".join(lines)

    class _P:  # keep the rest of the function unchanged
        returncode = rc
    proc = _P()
    lat = _LAT.search(out)
    npu = _NPU.search(out)
    job = _JOB.search(out)
    row = {
        "role": role,
        "module": module,
        "ok": proc.returncode == 0,
        "latency_ms": lat.group(1) if lat else "?",
        "npu_pct": npu.group(1) if npu else "?",
        "job_url": job.group(1) if job else "",
        "log": str(log_path.relative_to(PROJ)),
    }
    status = "OK" if row["ok"] else f"FAILED (rc={proc.returncode})"
    print(f"    {status}  latency={row['latency_ms']}ms  npu={row['npu_pct']}%  -> {row['log']}")
    return row


def write_results(device: str, rows: list[dict]) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    total = 0.0
    for r in rows:
        try:
            total += float(r["latency_ms"])
        except ValueError:
            total = float("nan")
            break
    lines = [
        f"\n## Run {stamp} — device: `{device}` — runtime: `{config.TARGET_RUNTIME}`\n",
        "| Stage | Model | On-device latency (ms) | NPU util (%) | Status | Log |",
        "|-------|-------|------------------------|--------------|--------|-----|",
    ]
    for r in rows:
        lines.append(
            f"| {r['role']} | `{r['module']}` | {r['latency_ms']} | {r['npu_pct']} | "
            f"{'ok' if r['ok'] else 'FAIL'} | {r['log']} |"
        )
    budget = config.FRAME_BUDGET_MS
    verdict = "PASS" if (total == total and total <= budget) else "OVER BUDGET"
    lines.append(
        f"\n**Sum of stage latencies: {total:.1f} ms** vs {budget:.0f} ms budget "
        f"(30 fps) -> **{verdict}**\n"
    )
    with RESULTS.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote summary -> {RESULTS.relative_to(PROJ)}")


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
    write_results(args.device, rows)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
