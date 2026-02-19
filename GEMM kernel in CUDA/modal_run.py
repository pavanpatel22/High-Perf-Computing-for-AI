import modal
import subprocess
from pathlib import Path

app = modal.App("naive-gemm-runner")

# CUDA "devel" image so nvcc exists + install build tools
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("build-essential", "cmake")
    # Modal 1.0+: add local project files via Image (Mount is deprecated/removed)
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/naive_gemm",
    )
)

@app.function(gpu="T4", image=image, timeout=60 * 20)
def build_and_run():
    subprocess.run(
        ["cmake", "-S", "/root/naive_gemm", "-B", "/root/naive_gemm/build"],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", "/root/naive_gemm/build", "-j"],
        check=True,
    )
    subprocess.run(
        ["/root/naive_gemm/build/naive_gemm"],
        check=True,
    )

@app.local_entrypoint()
def main():
    build_and_run.remote()
