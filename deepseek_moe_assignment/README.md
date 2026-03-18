DeepSeekMoE Assignment

1. Setup: `python -m venv .venv`, activate, `pip install -e ./transformers`  
2. Generate tests: `cd src && python generate_deepseek_moe_tests.py`
3. Compile: `gcc -O2 -std=c11 deepseek_moe_runner.c -lm -o deepseek_moe_runner.exe`
4. Test: `./deepseek_moe_runner.exe` (passes all cases)

Implements DeepSeekV3 MoE operator in pure C matching HF Transformers reference.