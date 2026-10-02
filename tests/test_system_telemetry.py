"""Tests for Settings-tab system telemetry helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from web import app as web_app


class SystemTelemetryHelpersTests(unittest.TestCase):
    def test_level_from_pct(self) -> None:
        self.assertEqual(web_app._level_from_pct(50, warn=70, critical=90), "ok")
        self.assertEqual(web_app._level_from_pct(75, warn=70, critical=90), "warn")
        self.assertEqual(web_app._level_from_pct(95, warn=70, critical=90), "critical")
        self.assertEqual(web_app._level_from_pct(None, warn=70, critical=90), "unknown")

    def test_level_from_temp(self) -> None:
        self.assertEqual(web_app._level_from_temp_c(55), "ok")
        self.assertEqual(web_app._level_from_temp_c(72), "warn")
        self.assertEqual(web_app._level_from_temp_c(82), "critical")
        self.assertEqual(web_app._level_from_temp_c(None), "unknown")

    @patch("web.app.shutil.which", return_value="/usr/bin/vcgencmd")
    @patch("web.app.subprocess.run")
    def test_throttle_status_parses_flags(self, run: MagicMock, _which: MagicMock) -> None:
        run.return_value = MagicMock(stdout="throttled=0x50005\n", returncode=0)
        status = web_app._pi_throttle_status()
        assert status is not None
        self.assertTrue(status["now"]["under_voltage"])
        self.assertTrue(status["now"]["throttled"])
        self.assertTrue(status["since_boot"]["under_voltage"])
        self.assertTrue(status["since_boot"]["throttled"])
        self.assertEqual(status["level"], "critical")

    @patch("web.app.psutil.cpu_percent", return_value=42.0)
    @patch("web.app.psutil.cpu_count", return_value=4)
    @patch("web.app.psutil.virtual_memory")
    @patch("web.app.psutil.boot_time", return_value=1_700_000_000)
    @patch("web.app.time.time", return_value=1_700_003_600)
    @patch("web.app._read_cpu_temp_c", return_value=61.5)
    @patch("web.app._pi_throttle_status", return_value=None)
    @patch("web.app._disk_usage_summary")
    @patch("web.app.hostname", return_value="pi-test")
    def test_system_telemetry_snapshot(
        self,
        _host: MagicMock,
        disk: MagicMock,
        _thr: MagicMock,
        _temp: MagicMock,
        _time: MagicMock,
        _boot: MagicMock,
        mem: MagicMock,
        _count: MagicMock,
        cpu: MagicMock,
    ) -> None:
        def _cpu_percent(*_a, **kw):
            if kw.get("percpu"):
                return [10.0, 20.0, 30.0, 40.0]
            return 42.0

        cpu.side_effect = _cpu_percent
        mem.return_value = MagicMock(
            percent=55.5,
            used=2 * 1024**3,
            total=4 * 1024**3,
            available=1.5 * 1024**3,
        )
        disk.return_value = {
            "ok": True,
            "free_pct": 40.0,
            "level": "ok",
            "line": "10 GB free of 25 GB (40%)",
            "free_bytes": 10,
            "total_bytes": 25,
            "used_bytes": 15,
            "free_label": "10 GB",
            "total_label": "25 GB",
        }
        with patch("web.app.os.getloadavg", return_value=(0.5, 0.6, 0.7), create=True):
            snap = web_app._system_telemetry()
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["hostname"], "pi-test")
        self.assertEqual(snap["cpu_percent"], 42.0)
        self.assertEqual(snap["memory_percent"], 55.5)
        self.assertEqual(snap["temp_c"], 61.5)
        self.assertEqual(snap["temp_level"], "ok")
        self.assertEqual(snap["uptime_s"], 3600)
        self.assertEqual(snap["overall_level"], "ok")


if __name__ == "__main__":
    unittest.main()
