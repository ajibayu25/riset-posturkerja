"""Serial helper to bridge Arduino-based glare detector into the ROSA UI."""

from __future__ import annotations

import json
import threading
import time
from typing import Dict, Optional

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover - pyserial optional in CI
    serial = None  # type: ignore


class GlareSerialClient(threading.Thread):
    """Background thread that listens to glare detector readings via serial."""

    def __init__(
        self,
        port: Optional[str],
        baudrate: int = 115200,
        timeout: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Dict[str, object] = {
            "connected": False,
            "glare": None,
            "lux_screen": None,
            "lux_room": None,
            "ratio": None,
            "updated_at": 0.0,
            "message": "Awaiting sensor data",
        }
        self._serial: Optional["serial.Serial"] = None  # type: ignore

    def run(self) -> None:  # pragma: no cover - background loop
        if self.port is None:
            self._update_status(False, "Port belum dikonfigurasi (GLARE_SERIAL_PORT).")
            return

        if serial is None:
            self._update_status(False, "Modul pyserial belum terpasang.")
            return

        while not self._stop_event.is_set():
            if self._serial is None:
                try:
                    self._serial = serial.Serial(
                        self.port,
                        self.baudrate,
                        timeout=self.timeout,
                    )
                    self._update_status(True, "Sensor terhubung.")
                except Exception as exc:  # pragma: no cover - hardware specific
                    self._update_status(False, f"Gagal koneksi: {exc}")
                    time.sleep(2.0)
                    continue

            try:
                line = self._serial.readline()
            except Exception as exc:  # pragma: no cover - hardware specific
                self._update_status(False, f"Koneksi terputus: {exc}")
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
                time.sleep(1.5)
                continue

            if not line:
                continue
            try:
                payload = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not payload:
                continue
            parsed = self._parse_payload(payload)
            if parsed is None:
                continue
            parsed["connected"] = True
            parsed["updated_at"] = time.time()
            parsed.setdefault("message", "Data diterima.")
            self._store(parsed)

    def stop(self) -> None:
        """Signal background loop to stop and close serial connection."""
        self._stop_event.set()
        if self._serial is not None:  # pragma: no cover - hardware specific
            try:
                self._serial.close()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, object]:
        """Return a shallow copy of the latest glare reading."""
        with self._lock:
            return dict(self._latest)

    def _store(self, payload: Dict[str, object]) -> None:
        with self._lock:
            self._latest.update(payload)

    def _update_status(self, connected: bool, message: str) -> None:
        with self._lock:
            self._latest["connected"] = connected
            self._latest["message"] = message
            if not connected:
                self._latest["glare"] = None

    @staticmethod
    def _parse_payload(text: str) -> Optional[Dict[str, object]]:
        """Parse JSON or key=value payload emitted by the Arduino sketch."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {
                    "glare": bool(data.get("glare")),
                    "lux_screen": GlareSerialClient._safe_float(data.get("lux_screen")),
                    "lux_room": GlareSerialClient._safe_float(data.get("lux_room")),
                    "ratio": GlareSerialClient._safe_float(data.get("ratio")),
                    "message": data.get("message", "Data diterima."),
                }
        except json.JSONDecodeError:
            pass

        if "=" not in text:
            return None

        result: Dict[str, object] = {}
        for token in text.replace(",", " ").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"glare", "flag"}:
                result["glare"] = value in {"1", "true", "True", "GLARE"}
            elif key in {"ratio", "lux_screen", "lux_room"}:
                result[key] = GlareSerialClient._safe_float(value)
        if not result:
            return None
        return result

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = ["GlareSerialClient"]
