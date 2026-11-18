#!/usr/bin/env python3
"""One-click launcher: sets up everything, trains (first run only), opens the dashboard.

    python run.py                 normal start
    python run.py --retrain       force re-training
    python run.py --quick         faster, smaller hyper-parameter search
    python run.py --no-browser    don't open the browser
    python run.py --no-venv       use the current Python instead of ./.venv
"""
import importlib.util
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
CORE_MODULES = ["numpy", "pandas", "sklearn", "matplotlib", "flask", "joblib"]
OPTIONAL = [("xgboost", "xgboost"), ("shap", "shap")]   # (module, pip name)
TRIED = ROOT / ".optional_tried"
ARGS = sys.argv[1:]


def say(msg=""):
    print(msg, flush=True)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_target_venv():
    try:
        return Path(sys.prefix).resolve() == VENV.resolve()
    except OSError:
        return False


def pip(*args):
    return subprocess.call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args])


def have(mod):
    return importlib.util.find_spec(mod) is not None


def ensure_venv():
    """Re-launch inside ./.venv (created on first run)."""
    if "--no-venv" in ARGS or in_target_venv() or os.environ.get("DD_INSIDE_VENV"):
        return
    try:
        if not venv_python().exists():
            say("Creating virtual environment (.venv) ...")
            import venv
            venv.EnvBuilder(with_pip=True, clear=False).create(VENV)
        env = dict(os.environ, DD_INSIDE_VENV="1")
        sys.exit(subprocess.call([str(venv_python()), str(Path(__file__).resolve()), *ARGS], env=env))
    except SystemExit:
        raise
    except Exception as e:
        say(f"(could not use a virtual environment: {e}; continuing with the current Python)")


def ensure_packages():
    if not all(have(m) for m in CORE_MODULES):
        say("Installing required packages (first run only, ~1-2 min) ...")
        if pip("-r", str(ROOT / "requirements.txt")) != 0:
            sys.exit("\nCould not install the required packages. Check your internet connection and retry.")
    if not TRIED.exists():
        for mod, name in OPTIONAL:
            if not have(mod):
                say(f"Installing optional package '{name}' ...")
                if pip(name) != 0:
                    say(f"  '{name}' could not be installed - the built-in fallback will be used instead.")
        TRIED.write_text("done")


def ensure_data():
    from src.common import DATA_PATH, download_data
    if not DATA_PATH.exists():
        say("Downloading the Pima Indians Diabetes dataset ...")
        try:
            download_data()
        except Exception as e:
            sys.exit(f"\n{e}")


def ensure_model():
    from src.common import MODEL_PATH, RESULTS_PATH
    if "--retrain" in ARGS or not (MODEL_PATH.exists() and RESULTS_PATH.exists()):
        say("\nTraining models (first run only, a few minutes) ...\n")
        cmd = [sys.executable, "-m", "src.train"] + (["--quick"] if "--quick" in ARGS else [])
        if subprocess.call(cmd, cwd=str(ROOT)) != 0:
            sys.exit("\nTraining failed - see the messages above.")


def free_port(start=5000):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main():
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 or newer is required (https://www.python.org/downloads/).")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    ensure_venv()
    ensure_packages()
    ensure_data()
    ensure_model()
    port = int(ARGS[ARGS.index("--port") + 1]) if "--port" in ARGS else free_port()
    url = f"http://127.0.0.1:{port}/"
    say(f"\nDashboard running at {url}   (press Ctrl+C to stop)\n")
    if "--no-browser" not in ARGS:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    from src.app import run
    run(port=port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nStopped.")
