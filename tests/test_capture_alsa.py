"""Tests for ALSA capture helpers."""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from core.capture_alsa import (
    CaptureDeviceError,
    _arecord_lists_capture_cards,
    check_capture_device_available,
)


class ArecordListingTests(unittest.TestCase):
    def test_detects_capture_cards(self) -> None:
        listing = (
            "**** List of CAPTURE Hardware Devices ****\n"
            "card 3: sndrpigooglevoi [snd_rpi_googlevoicehat_card], device 0: ..."
        )
        self.assertTrue(_arecord_lists_capture_cards(listing))

    def test_empty_listing(self) -> None:
        self.assertFalse(_arecord_lists_capture_cards("**** List of CAPTURE Hardware Devices ****\n"))


class CheckCaptureDeviceTests(unittest.TestCase):
    @patch("core.capture_alsa.shutil.which", return_value=None)
    def test_requires_arecord(self, _which: MagicMock) -> None:
        with self.assertRaises(CaptureDeviceError) as ctx:
            check_capture_device_available("plughw:0,0")
        self.assertIn("arecord not found", str(ctx.exception))

    @patch("core.capture_alsa.shutil.which", return_value="/usr/bin/arecord")
    @patch("core.capture_alsa.subprocess.run")
    def test_no_hardware(self, run: MagicMock, _which: MagicMock) -> None:
        run.return_value = MagicMock(
            returncode=0,
            stdout="**** List of CAPTURE Hardware Devices ****\n",
            stderr="",
        )
        with self.assertRaises(CaptureDeviceError) as ctx:
            check_capture_device_available("plughw:0,0")
        self.assertIn("No sound capture device", str(ctx.exception))

    @patch("core.capture_alsa.shutil.which", return_value="/usr/bin/arecord")
    @patch("core.capture_alsa._probe_arecord_device", return_value="No such file or directory")
    @patch("core.capture_alsa.subprocess.run")
    def test_device_wont_open(
        self,
        run: MagicMock,
        _probe: MagicMock,
        _which: MagicMock,
    ) -> None:
        run.return_value = MagicMock(returncode=0, stdout="card 1: test\n", stderr="")
        with self.assertRaises(CaptureDeviceError) as ctx:
            check_capture_device_available("plughw:99,0")
        self.assertIn("Cannot open ALSA capture device", str(ctx.exception))

    @patch("core.capture_alsa.shutil.which", return_value="/usr/bin/arecord")
    @patch("core.capture_alsa._probe_arecord_device", return_value=None)
    @patch("core.capture_alsa.subprocess.run")
    def test_ok(
        self,
        run: MagicMock,
        _probe: MagicMock,
        _which: MagicMock,
    ) -> None:
        run.return_value = MagicMock(returncode=0, stdout="card 1: test\n", stderr="")
        check_capture_device_available("plughw:1,0")


class ProbeArecordDeviceTests(unittest.TestCase):
    @patch("core.capture_alsa.subprocess.Popen")
    def test_timeout_means_open(self, popen: MagicMock) -> None:
        from core.capture_alsa import _probe_arecord_device

        proc = MagicMock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="arecord", timeout=2)
        popen.return_value = proc
        self.assertIsNone(_probe_arecord_device("plughw:1,0"))


if __name__ == "__main__":
    unittest.main()
