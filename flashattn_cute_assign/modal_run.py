import subprocess
import modal

app = modal.App("flashattn-cute-assignment")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential", "git")
    .add_local_dir(".", remote_path="/root/project", copy=False)
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 20,
)
def build_and_run(N=128, D=64, Br=64, Bc=64, dtype="f16", causal=False):
    proj = "/root/project"
    cutlass_dir = f"{proj}/third_party/cutlass"
    cutlass_inc = f"{cutlass_dir}/include"

    build_cmd = f"""
    set -euo pipefail
    cd "{proj}"

    echo "=== nvcc version ==="
    nvcc --version

    echo "=== Ensure CUTLASS (CuTe) headers ==="
    mkdir -p third_party
    if [ ! -d "{cutlass_dir}" ]; then
      git clone --recursive https://github.com/NVIDIA/cutlass.git "{cutlass_dir}"
    else
      echo "CUTLASS already present."
    fi

    test -f "{cutlass_inc}/cute/tensor.hpp"
    test -f "{cutlass_inc}/cute/layout.hpp"
    echo "CuTe headers found."

    echo "=== src/ files ==="
    ls -la src

    echo "=== Build CuTe FlashAttention ==="
    # NOTE: main_cute.cu should include <cstring> for strcmp on Linux.
    # If you haven't added it, this compile can fail. Best fix is to add:
    #   #include <cstring>
    # at top of main_cute.cu.
    nvcc -O2 -std=c++17 \
      -Isrc -I"{cutlass_inc}" \
      src/main_cute.cu src/flashattn_cuda_cute.cu \
      -o flashattn_cute

    echo "=== Built binary ==="
    ls -la ./flashattn_cute
    """

    causal_flag = "--causal" if causal else ""
    run_cmd = f"""
    set -e
    cd "{proj}"
    echo "=== Run ==="
    ./flashattn_cute --N {N} --D {D} --Br {Br} --Bc {Bc} --dtype {dtype} {causal_flag}
    """

    subprocess.check_call(["bash", "-lc", build_cmd])

    rc = subprocess.call(["bash", "-lc", run_cmd])
    print(f"Program exit code: {rc}")
    if rc != 0:
        raise RuntimeError(f"flashattn_cute returned non-zero exit code {rc}")


@app.local_entrypoint()
def main():
    build_and_run.remote(N=128, D=64, Br=64, Bc=64, dtype="f16", causal=False)
    build_and_run.remote(N=512, D=64, Br=128, Bc=64, dtype="f16", causal=False)
    build_and_run.remote(N=256, D=64, Br=64,  Bc=64, dtype="f16", causal=True)