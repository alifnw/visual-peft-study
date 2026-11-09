"""Sample one training process tree without mixing its memory with other users."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("process_tree_rss_mib", "process_tree_vram_mib", "device_gpu_utilization_percent",
           "device_vram_used_mib", "host_available_ram_mib")


def process_tree(pid):
    records = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            records[int(path.name)] = (int(fields[1]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue
    selected = {pid} if pid in records else set()
    while True:
        expanded = selected | {child for child, (parent, _) in records.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    rss = sum(records[child][1] for child in selected) * os.sysconf("SC_PAGE_SIZE") / 1024**2
    return selected, rss if selected else None


def query_gpu(gpu, fields, compute=False):
    kind = "--query-compute-apps=" if compute else "--query-gpu="
    output = subprocess.check_output(["nvidia-smi", "--id=" + gpu, kind + fields,
                                      "--format=csv,noheader,nounits"],
                                     text=True, timeout=10)
    return list(csv.reader(output.splitlines()))


def sample(pid, gpu, started):
    pids, rss = process_tree(pid)
    row = {"elapsed_seconds": time.monotonic() - started,
           "utc": datetime.now(timezone.utc).isoformat(), "process_tree_rss_mib": rss}
    try:
        device = query_gpu(gpu, "utilization.gpu,memory.used")[0]
        row.update(device_gpu_utilization_percent=float(device[0]), device_vram_used_mib=float(device[1]))
        contexts = query_gpu(gpu, "pid,used_gpu_memory", compute=True)
        row["process_tree_vram_mib"] = sum(float(memory) for child, memory in contexts if int(child) in pids)
    except (subprocess.SubprocessError, ValueError, IndexError, OSError) as error:
        row["sampling_error"] = type(error).__name__
    meminfo = {line.split(":", 1)[0]: int(line.split()[1]) / 1024
               for line in Path("/proc/meminfo").read_text().splitlines()
               if line.startswith("MemAvailable:")}
    row["host_available_ram_mib"] = meminfo["MemAvailable"]
    return row


def summarize(samples, elapsed):
    result = {}
    for metric in METRICS:
        weighted, duration, valid = 0.0, 0.0, []
        for index, row in enumerate(samples):
            value = row.get(metric)
            if value is None:
                continue
            end = samples[index + 1]["elapsed_seconds"] if index + 1 < len(samples) else elapsed
            weight = max(0.0, end - row["elapsed_seconds"])
            weighted += value * weight
            duration += weight
            valid.append(value)
        result[metric] = {"time_weighted_mean": weighted / duration if duration else None,
                          "observed_peak": max(valid) if valid else None,
                          "observed_minimum": min(valid) if valid else None,
                          "covered_seconds": duration, "valid_samples": len(valid)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval < 1:
        parser.error("Provide a training command and a sampling interval of at least one second")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    samples = []
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with (output / "training.log").open("w") as log, (output / "resource_samples.csv").open("w") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        def terminate(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        writer = csv.DictWriter(stream, fieldnames=("elapsed_seconds", "utc", *METRICS, "sampling_error"))
        writer.writeheader()
        (output / "profile_status.json").write_text(json.dumps({"phase": "running", "pid": child.pid,
                                                               "gpu": args.gpu, "started_utc": started_utc}))
        print(json.dumps({"phase": "running", "child_pid": child.pid, "profile": str(output)}), flush=True)
        deadline = started
        while child.poll() is None:
            row = sample(child.pid, args.gpu, started)
            samples.append(row)
            writer.writerow(row)
            stream.flush()
            deadline += args.interval
            while child.poll() is None and time.monotonic() < deadline:
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        returncode = child.wait()
    elapsed = time.monotonic() - started
    result = {"phase": "complete" if returncode == 0 else "failed", "returncode": returncode,
              "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
              "wall_elapsed_seconds": elapsed, "gpu": args.gpu, "sample_interval_seconds": args.interval,
              "sample_count": len(samples), "command": command,
              "metrics": summarize(samples, elapsed),
              "measurement_scope": "Child process startup through exit, including loading, validation and checkpoint writes. Setup and queue wait are excluded.",
              "mean_definition": "Time-weighted left-hold average between samples. Missing samples are excluded and coverage is reported.",
              "memory_definition": "RSS sums this process tree and can count shared pages more than once. VRAM sums NVIDIA contexts for this process tree.",
              "device_utilization_scope": "Whole selected GPU, not utilization attributable only to this training process.",
              "peak_definition": "Sampled peaks may miss short spikes. Framework and OS high-water marks remain in result.json."}
    if args.result and (ROOT / args.result).exists():
        training = json.loads((ROOT / args.result).read_text())
        if training.get("total_epoch_seconds") is not None:
            total = training["total_epoch_seconds"]
            result["observed_total_epoch_minutes"] = total / 60
            result["observed_mean_epoch_seconds"] = total / training["epochs"] if training["epochs"] else None
        result["training_result"] = str(args.result)
        result["training_phase"] = training.get("phase")
        for key in ("peak_process_rss_mib", "peak_gpu_allocated_mib", "peak_gpu_reserved_mib"):
            result[key] = training.get(key)
    (output / "resource_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "profile_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
