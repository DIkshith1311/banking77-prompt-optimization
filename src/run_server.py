"""Run the benchmark against a locally managed llama.cpp server."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests

from run import METHODS

MODEL = "qwen2.5-3b-instruct-q4_k_m"
SOURCE_DIR = Path(__file__).resolve().parent


def wait_for_server(process, origin):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {process.returncode}")
        try:
            response = requests.get(f"{origin}/health", timeout=2)
            if response.ok and response.json().get("status") == "ok":
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-exe", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True, help="GGUF file or first shard")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18976)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--device", default="none", help="Device name from llama-server --list-devices")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--phase", choices=("all", "fit", "test"), default="all")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()

    origin = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.server_exe.resolve()), "-m", str(args.gguf.resolve()), "--host", "127.0.0.1",
        "--port", str(args.port), "--alias", args.model, "-t", str(args.threads),
        "-tb", str(args.threads), "--device", args.device, "-ngl", str(args.gpu_layers), "-fit", "off",
        "-c", "12288", "-np", "4", "-b", "512", "-ub", "128", "-fa", "on",
    ]
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    with args.server_log.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            wait_for_server(process, origin)
            metadata = requests.get(f"{origin}/v1/models", timeout=10)
            metadata.raise_for_status()
            if args.model not in {model["id"] for model in metadata.json()["data"]}:
                raise RuntimeError("The endpoint advertised a different model")
            print(f"Ready: {args.model}", flush=True)
            subprocess.run([
                sys.executable, "-u", str(SOURCE_DIR / "run.py"), "--url", f"{origin}/v1",
                "--model", args.model, "--out", str(args.out.resolve()), args.phase,
                "--workers", "4", "--methods", *args.methods,
            ], check=True)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
