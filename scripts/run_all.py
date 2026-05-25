"""
Start MedRev + echobox services.

Processes:
  - MedRev Flask    (port 8080)
  - echobox app     (port 8000)
  - echobox ML      (port 9090, optional)
  - echobox Vite    (port 5173)

Ctrl+C stops all services.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ECHOBOX = ROOT / "echobox"


def main():
    procs: list[subprocess.Popen] = []

    def stop_all():
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()

    def on_sig(signum, frame):
        print("\nShutting down...")
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    env = os.environ.copy()

    # 1. echobox FastAPI backend
    print("[1/3] Starting echobox app (port 8000)...")
    uv_cmd = "uv.exe" if sys.platform == "win32" else "uv"
    app_proc = subprocess.Popen(
        [
            uv_cmd, "run", "--package", "echobox-app",
            "uvicorn", "echobox_app.main:create_app",
            "--factory", "--host", "127.0.0.1", "--port", "8000",
        ],
        cwd=str(ECHOBOX),
        env=env,
    )
    procs.append(app_proc)
    time.sleep(2)

    # 2. echobox Vite dev server
    print("[2/3] Starting echobox frontend (port 5173)...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    web_proc = subprocess.Popen(
        [npm, "--prefix", str(ECHOBOX / "frontend"), "run", "dev"],
        cwd=str(ECHOBOX),
        env=env,
    )
    procs.append(web_proc)
    time.sleep(2)

    # 3. MedRev Flask
    print("[3/3] Starting MedRev (port 8080)...")
    medrev_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "backend" / "app.py")],
        cwd=str(ROOT),
        env=env,
    )
    procs.append(medrev_proc)

    print("\nAll services started:")
    print("  MedRev:         http://127.0.0.1:8080")
    print("  echobox API:    http://127.0.0.1:8000")
    print("  echobox UI:     http://127.0.0.1:5173")
    print("\nPress Ctrl+C to stop all.\n")

    try:
        while any(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()


if __name__ == "__main__":
    main()
