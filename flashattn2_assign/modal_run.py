import modal
import subprocess

app = modal.App("flashattention2-assignment")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential")
    .add_local_dir(".", remote_path="/root/project", copy=False)
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 20,
)
def build_and_run(N=128, D=64, Br=64, Bc=64, dtype="f16", causal=False):
    proj = "/root/project"

    build_cmd = f"""
    set -euo pipefail
    cd "{proj}"
    echo "=== nvcc version ==="
    nvcc --version
    echo "=== Build ==="
    nvcc -O2 -std=c++17 -Isrc \
      src/main.cu \
      src/flashattn_cuda.cu \
      src/flashattn_cpu.c \
      src/naive_attention.c \
      -o flashattn
    """

    causal_flag = "--causal" if causal else ""
    run_cmd = f"""
    set -e
    cd "{proj}"
    echo "=== Run ==="
    ./flashattn --N {N} --D {D} --Br {Br} --Bc {Bc} --dtype {dtype} {causal_flag}
    """

    subprocess.check_call(["bash", "-lc", build_cmd])

    # ✅ run and stream output; if it fails, show return code but don’t throw away logs
    rc = subprocess.call(["bash", "-lc", run_cmd])
    print(f"Program exit code: {rc}")
    if rc != 0:
        raise RuntimeError(f"flashattn returned non-zero exit code {rc}")

@app.local_entrypoint()
def main():
    build_and_run.remote(N=128, D=64, Br=64, Bc=64, dtype="f16", causal=False)
    build_and_run.remote(N=512, D=64, Br=128, Bc=64, dtype="f16", causal=False)
    build_and_run.remote(N=256, D=64, Br=64,  Bc=64, dtype="f16", causal=True)