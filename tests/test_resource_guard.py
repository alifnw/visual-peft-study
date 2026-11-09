from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resource_guard import GIB, estimate, snapshot, stop_reason


class ResourceGuardTests(unittest.TestCase):
    def test_cache_credit_is_fractional_and_capped(self):
        state = estimate(16 * GIB, 16 * GIB, {"file": 8 * GIB, "active_file": 8 * GIB}, {})
        self.assertEqual(state["active_cache_credit_bytes"], 512 * 1024 ** 2)
        self.assertEqual(state["headroom_gib"], 0)

    def test_dirty_shared_and_mapped_pages_are_not_credited(self):
        stats = {"file": GIB, "active_file": GIB, "file_dirty": GIB // 4,
                 "file_writeback": GIB // 4, "shmem": GIB // 4, "file_mapped": GIB // 4}
        self.assertEqual(estimate(16 * GIB, 16 * GIB, stats, {})["active_cache_credit_bytes"], 0)

    def test_inactive_and_active_are_not_double_counted(self):
        stats = {"file": GIB, "active_file": GIB, "inactive_file": GIB}
        state = estimate(16 * GIB, 16 * GIB, stats, {})
        self.assertEqual(state["active_cache_credit_bytes"], 0)
        self.assertEqual(state["runtime_headroom_gib"], 1)

    def test_nonreclaimable_usage_still_stops_the_job(self):
        stats = {"anon": 16 * GIB, "file": 2 * GIB, "active_file": 2 * GIB}
        state = estimate(16 * GIB, 16 * GIB, stats, {})
        self.assertIn("Anonymous", stop_reason(state, 0))

    def test_memory_pressure_and_oom_still_stop_the_job(self):
        state = estimate(16 * GIB, 10 * GIB, {}, {"oom_kill": 1, "oom": 2}, 11)
        self.assertIn("OOM kill", stop_reason(state, 0))
        self.assertIn("allocation", stop_reason(state, 1, 1))
        self.assertIn("pressure", stop_reason(state, 1, 2))

    def test_critical_headroom_is_not_allowed(self):
        state = estimate(16 * GIB, 16 * GIB, {}, {})
        self.assertIn("0.35", stop_reason(state, 0))

    def fixture(self, directory):
        proc, cgroup = Path(directory) / "proc", Path(directory) / "cgroup"
        (proc / "self").mkdir(parents=True)
        cgroup.mkdir()
        (proc / "meminfo").write_text(f"MemTotal: {64 * 1024**2} kB\nMemAvailable: {40 * 1024**2} kB\n")
        (proc / "self/cgroup").write_text("0::/user.slice/session.scope\n")
        return proc, cgroup

    def write_limit(self, directory, limit, used):
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in {"memory.max": str(limit), "memory.current": str(used),
                           "memory.stat": f"anon {used}\n", "memory.events": "oom 0\noom_kill 0\n"}.items():
            (directory / name).write_text(text)

    def test_bare_metal_without_root_controller_uses_host_available(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, cgroup = self.fixture(directory)
            self.write_limit(cgroup / "user.slice/session.scope", "max", GIB)
            state = snapshot(cgroup, proc)
            self.assertEqual(state["scope"], "host_memavailable")
            self.assertEqual(state["headroom_gib"], 40)

    def test_parent_limit_applies_to_an_unlimited_session(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, cgroup = self.fixture(directory)
            self.write_limit(cgroup / "user.slice/session.scope", "max", GIB)
            self.write_limit(cgroup / "user.slice", 16 * GIB, 15 * GIB)
            state = snapshot(cgroup, proc)
            self.assertEqual(state["headroom_gib"], 1)
            self.assertEqual(state["resident_headroom_gib"], 1)

    def test_container_root_limit_remains_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, cgroup = self.fixture(directory)
            self.write_limit(cgroup, 16 * GIB, 14 * GIB)
            state = snapshot(cgroup, proc)
            self.assertEqual(state["headroom_gib"], 2)


if __name__ == "__main__":
    unittest.main()
