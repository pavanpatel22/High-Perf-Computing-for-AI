"""
Test runner: checks all generated test cases against multi-rank EP MoE.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from moe_ep_distributed import run_case

TESTS_DIR  = "tests"
WORLD_SIZE = 2   # simulated ranks
TOLERANCE  = 1e-5

def main():
    with open(os.path.join(TESTS_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    print(f"Running {len(manifest)} test cases  (world_size={WORLD_SIZE})\n")
    print(f"{'Case':<12} {'B':>4} {'S':>4} {'Max Err':>14}  Result")
    print("-" * 45)

    all_pass   = True
    global_max = 0.0

    for entry in manifest:
        name     = entry["name"]
        test_dir = os.path.join(TESTS_DIR, name)
        err      = run_case(test_dir, world_size=WORLD_SIZE)
        ok       = err < TOLERANCE
        if err > global_max:
            global_max = err
        if not ok:
            all_pass = False
        print(f"{name:<12} {entry['batch']:>4} {entry['seq']:>4} "
              f"{err:>14.9f}  {'PASS' if ok else 'FAIL'}")

    print("-" * 45)
    print(f"Global max error: {global_max:.9f}")
    print(f"\nAll tests: {'PASSED ✓' if all_pass else 'FAILED ✗'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())