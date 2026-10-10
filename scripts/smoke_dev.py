"""Verify documented one-command development startup and clean shutdown."""
import subprocess
import time
import urllib.request
from pathlib import Path
root = Path(__file__).resolve().parents[1]
process = subprocess.Popen(["bash", "scripts/dev.sh"], cwd=root)
try:
    for url in ["http://127.0.0.1:8000/health", "http://127.0.0.1:3000"]:
        for attempt in range(100):
            if process.poll() is not None:
                raise RuntimeError("Development servers exited before readiness")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    assert response.status == 200
                    print(f"Verified HTTP 200: {url}",flush=True)
                break
            except OSError:
                time.sleep(.2)
        else:
            raise RuntimeError(f"Startup timed out: {url}")
finally:
    process.terminate()
    process.wait(timeout=10)
    time.sleep(.5)
for url in ["http://127.0.0.1:8000/health", "http://127.0.0.1:3000"]:
    try:
        urllib.request.urlopen(url,timeout=1)
    except OSError:
        continue
    raise RuntimeError(f"Owned server remained running: {url}")
print("Verified both development servers stopped")