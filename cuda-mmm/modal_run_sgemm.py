import subprocess
import modal

app = modal.App("cuda-mmm-sgemm")

# CUDA "devel" image so nvcc exists in the container
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("cmake", "ninja-build", "build-essential")
    # Copy your local project into the container image, but ignore build artifacts
    .add_local_dir(
        ".",
        remote_path="/root/cuda-mmm",
        ignore=[
            "build",
            "build/**",
            ".git",
            ".git/**",
            ".vscode",
            ".vscode/**",
            "__pycache__",
            "**/__pycache__/**",
            "**/*.exe",
            "**/*.obj",
            "**/*.pdb",
            "**/*.exp",
            "**/*.lib",
            "**/*.o",
            "**/*.a",
            "**/*.so",
        ],
    )
)

@app.function(
    gpu="H100",
    image=image,
    timeout=60 * 45,  # 45 minutes
)
def build_and_run():
    print("\n===== GPU INFO =====\n")
    subprocess.check_call("nvidia-smi", shell=True)

    # Build OUTSIDE the source tree to avoid Windows CMakeCache conflicts
    build_dir = "/tmp/cuda-mmm-build"
    subprocess.check_call(f"rm -rf {build_dir}", shell=True)

    print("\n===== CONFIGURE =====\n")
    subprocess.check_call(
        f"cmake -S /root/cuda-mmm -B {build_dir} -G Ninja -DCMAKE_CUDA_ARCHITECTURES=90",
        shell=True,
    )

    print("\n===== BUILD =====\n")
    subprocess.check_call(f"cmake --build {build_dir} -j", shell=True)

    exe = f"{build_dir}/sgemm_bench"

    # Quick small smoke test (should be fast)
    print("\n===== SMOKE TEST (512) =====\n")
    for a in [0, 6]:
        print("\n------------------------------")
        print(f"SMOKE algo={a}  M=N=K=512")
        print("------------------------------\n")
        subprocess.run(
            f"{exe} --m=512 --n=512 --k=512 --algo={a} --warmup=10 --iters=10 --alpha=1 --beta=0",
            shell=True,
            check=False,
        )

    # Main tests: run all algorithms on 4096, do not crash on FAIL
    print("\n===== MAIN RUNS (4096) =====\n")
    for a in [0, 1, 2, 3, 4, 5, 6]:
        print("\n==============================")
        print(f"RUN algo={a}  M=N=K=4096")
        print("==============================\n")
        subprocess.run(
            f"{exe} --m=4096 --n=4096 --k=4096 --algo={a} --warmup=10 --iters=20 --alpha=1 --beta=0",
            shell=True,
            check=False,
        )

    print("\nDONE. Copy the output above into your report.\n")

@app.local_entrypoint()
def main():
    build_and_run.remote()
