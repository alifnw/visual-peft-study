"""Bounded cache-aware cgroup estimates, never a memory reservation or guarantee."""

from pathlib import Path

GIB = 1024 ** 3
ACTIVE_CREDIT_FRACTION = 0.25
ACTIVE_CREDIT_CAP = 512 * 1024 ** 2


def estimate(limit, used, stats, events, full_pressure_avg10=0.0):
    clean_file = max(0, stats.get("file", 0) - sum(stats.get(k, 0) for k in
                     ("shmem", "file_dirty", "file_writeback", "file_mapped")))
    inactive = min(max(0, stats.get("inactive_file", 0)), clean_file)
    active_clean = min(max(0, stats.get("active_file", 0)), max(0, clean_file - inactive))
    active_credit = min(ACTIVE_CREDIT_CAP, int(active_clean * ACTIVE_CREDIT_FRACTION))
    conservative = limit - used + inactive
    resident_ceiling = limit - sum(stats.get(k, 0) for k in ("anon", "kernel", "shmem"))
    return {
        "limit_bytes": limit, "used_bytes": used,
        "headroom_gib": conservative / GIB,
        "runtime_headroom_gib": (conservative + active_credit) / GIB,
        "resident_headroom_gib": resident_ceiling / GIB,
        "active_clean_file_bytes": active_clean,
        "inactive_file_bytes": stats.get("inactive_file", 0),
        "active_cache_credit_bytes": active_credit,
        "cache_credit_fraction": ACTIVE_CREDIT_FRACTION,
        "cache_credit_cap_bytes": ACTIVE_CREDIT_CAP,
        "full_pressure_avg10_percent": full_pressure_avg10,
        "oom_kill_count": events.get("oom_kill", 0),
        "oom_count": events.get("oom", 0),
        "definition": "free + clean inactive file + 25% clean unmapped active file, active credit capped at 512 MiB",
    }


def _pressure(path):
    full_pressure = 0.0
    if not path.exists():
        return full_pressure
    for line in path.read_text().splitlines():
        if line.startswith("full "):
            values = dict(word.split("=") for word in line.split()[1:])
            full_pressure = float(values["avg10"])
    return full_pressure


def snapshot(cgroup_root=Path("/sys/fs/cgroup"), proc_root=Path("/proc")):
    cgroup_root, proc_root = Path(cgroup_root), Path(proc_root)
    meminfo = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
               for line in (proc_root / "meminfo").read_text().splitlines()
               if line.split(":", 1)[0] in {"MemTotal", "MemAvailable"}}
    total, available = meminfo["MemTotal"], meminfo["MemAvailable"]
    host = estimate(total, total - available, {"anon": total - available}, {},
                    _pressure(proc_root / "pressure/memory"))
    host.update(scope="host_memavailable", definition="Host MemAvailable from /proc/meminfo")
    states, directories = [host], {cgroup_root}
    for line in (proc_root / "self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            relative = Path(line[3:].lstrip("/"))
            if ".." not in relative.parts:
                current = cgroup_root / relative
                while current.is_relative_to(cgroup_root):
                    directories.add(current)
                    if current == cgroup_root:
                        break
                    current = current.parent
    for directory in sorted(directories):
        limit_path = directory / "memory.max"
        if not limit_path.exists() or limit_path.read_text().strip() == "max":
            continue
        stats = {k: int(v) for k, v in
                 (line.split() for line in (directory / "memory.stat").read_text().splitlines())}
        events = {k: int(v) for k, v in
                  (line.split() for line in (directory / "memory.events").read_text().splitlines())}
        state = estimate(int(limit_path.read_text()),
                         int((directory / "memory.current").read_text()), stats, events,
                         _pressure(directory / "memory.pressure"))
        state["scope"] = str(directory)
        states.append(state)
    result = dict(min(states, key=lambda state: state["runtime_headroom_gib"]))
    for key in ["headroom_gib", "runtime_headroom_gib", "resident_headroom_gib"]:
        result[key] = min(state[key] for state in states)
    for key in ["oom_kill_count", "oom_count"]:
        result[key] = sum(state[key] for state in states)
    result["full_pressure_avg10_percent"] = max(state["full_pressure_avg10_percent"] for state in states)
    result["host_memavailable_bytes"] = available
    result["limits_checked"] = [{key: state[key] for key in
                                 ("scope", "limit_bytes", "used_bytes", "headroom_gib")}
                                for state in states]
    return result


def stop_reason(state, initial_oom_kills, initial_oom_count=None):
    if state["oom_kill_count"] != initial_oom_kills:
        return "New OOM kill recorded in the shared cgroup"
    if initial_oom_count is not None and state["oom_count"] != initial_oom_count:
        return "New OOM allocation event recorded in the shared cgroup"
    if state["runtime_headroom_gib"] < 0.35:
        return "Cache-budgeted RAM headroom below 0.35 GiB"
    if state["resident_headroom_gib"] < 0.75:
        return "Anonymous, shared-memory and kernel allocations leave less than 0.75 GiB"
    if state["full_pressure_avg10_percent"] > 10.0:
        return "Memory full-stall pressure exceeded 10% over the recent 10-second window"
    return None
