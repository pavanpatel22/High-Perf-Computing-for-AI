# nano-sglang

A minimal LLM inference engine. Model: Qwen3-0.6B.

## Run tests

```bash
pytest tests/test_kv_cache.py -v          # local, no GPU
modal run modal_run.py::test              # all tests on GPU
```

## What to implement

| File | Function |
|------|----------|
| `kv_cache.py` | `update()`, `get()` |
| `engine.py` | `prefill()`, `generate()` |
| `scheduler.py` | `_decode_running()`, `step()`, `run_to_completion()` |
| `block_manager.py` | `allocate()`, `free()` (stretch) |

## Reference

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [nano-vllm walkthrough](https://neutree.ai/blog/nano-vllm-part-1)
