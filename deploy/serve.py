"""Serve Streamlit and the optional D-Tale explorer through one Cloud Run port."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    config = Path("deploy/nginx.conf").read_text().replace("__PORT__", str(port))
    Path("/tmp/nginx.conf").write_text(config)
    processes = []

    def stop(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        processes.append(subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address=127.0.0.1", "--server.port=8501",
            "--server.headless=true", "--server.enableXsrfProtection=true",
        ]))
        processes.append(subprocess.Popen(["nginx", "-c", "/tmp/nginx.conf", "-g", "daemon off;"]))
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise SystemExit(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
