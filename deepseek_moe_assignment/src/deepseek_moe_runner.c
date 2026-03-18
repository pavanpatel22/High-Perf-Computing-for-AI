#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_CASES 16
#define MAX_PATH 256
#define MAX_EXPERTS 8
#define MAX_SHARED 2
#define MAX(a,b) ((a) > (b) ? (a) : (b))

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float* load_f32(const char* path, size_t expected_elems) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }
    float* buf = (float*)malloc(expected_elems * sizeof(float));
    if (!buf) {
        fprintf(stderr, "OOM loading %s\n", path);
        exit(1);
    }
    size_t n = fread(buf, sizeof(float), expected_elems, f);
    fclose(f);
    if (n != expected_elems) {
        fprintf(stderr, "Expected %zu floats in %s, got %zu\n", expected_elems, path, n);
        exit(1);
    }
    return buf;
}

static int* load_i32(const char* path, size_t expected_elems) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(1);
    }
    int* buf = (int*)malloc(expected_elems * sizeof(int));
    if (!buf) {
        fprintf(stderr, "OOM loading %s\n", path);
        exit(1);
    }
    size_t n = fread(buf, sizeof(int), expected_elems, f);
    fclose(f);
    if (n != expected_elems) {
        fprintf(stderr, "Expected %zu ints in %s, got %zu\n", expected_elems, path, n);
        exit(1);
    }
    return buf;
}

/* Very small parser for meta.json */
typedef struct {
    int hidden_size;
    int intermediate_size;
    int n_routed_experts;
    int n_shared_experts;
    int top_k;
    float routed_scaling_factor;
    int batch_size;
    int seq_len;
} Meta;

static void parse_meta(const char* path, Meta* m) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open meta %s\n", path);
        exit(1);
    }
    char buf[4096];  // larger buffer for safety
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    char* p;

    p = strstr(buf, "\"hidden_size\"");
    if (p) sscanf(p, "\"hidden_size\"%*[^0-9]%d", &m->hidden_size);

    p = strstr(buf, "\"intermediate_size\"");
    if (p) sscanf(p, "\"intermediate_size\"%*[^0-9]%d", &m->intermediate_size);

    p = strstr(buf, "\"n_routed_experts\"");
    if (p) sscanf(p, "\"n_routed_experts\"%*[^0-9]%d", &m->n_routed_experts);

    p = strstr(buf, "\"n_shared_experts\"");
    if (p) sscanf(p, "\"n_shared_experts\"%*[^0-9]%d", &m->n_shared_experts);

    p = strstr(buf, "\"top_k\"");
    if (p) sscanf(p, "\"top_k\"%*[^0-9]%d", &m->top_k);

    p = strstr(buf, "\"routed_scaling_factor\"");
    if (p) sscanf(p, "\"routed_scaling_factor\"%*[^0-9]%f", &m->routed_scaling_factor);

    p = strstr(buf, "\"batch_size\"");
    if (p) sscanf(p, "\"batch_size\"%*[^0-9]%d", &m->batch_size);

    p = strstr(buf, "\"seq_len\"");
    if (p) sscanf(p, "\"seq_len\"%*[^0-9]%d", &m->seq_len);

    /* Validate we parsed everything */
    if (m->hidden_size <= 0 || m->intermediate_size <= 0 || m->batch_size <= 0 || m->seq_len <= 0) {
        fprintf(stderr, "Failed to parse valid config from %s\n", path);
        exit(1);
    }
}

/* Linear: out[N, out_dim] = in[N, in_dim] @ W[out_dim, in_dim]^T */
static void linear_forward(const float* input, const float* weight, float* output,
                          int N, int in_dim, int out_dim) {
    for (int n = 0; n < N; ++n) {
        const float* x = input + n * in_dim;
        float* y = output + n * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float* w = weight + o * in_dim;
            float sum = 0.0f;
            for (int i = 0; i < in_dim; ++i)
                sum += x[i] * w[i];
            y[o] = sum;
        }
    }
}

/* TinyMLP with SwiGLU */
static void tiny_mlp_forward(const float* input, const float* w_gate, const float* w_up, 
                            const float* w_down, float* output, int N, int H, int I) {
    float* gate = (float*)malloc(N * I * sizeof(float));
    float* up   = (float*)malloc(N * I * sizeof(float));
    float* hid  = (float*)malloc(N * I * sizeof(float));
    if (!gate || !up || !hid) {
        fprintf(stderr, "OOM in tiny_mlp_forward\n");
        exit(1);
    }

    linear_forward(input, w_gate, gate, N, H, I);
    linear_forward(input, w_up, up, N, H, I);

    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < I; ++i) {
            float g = gate[n * I + i];
            float u = up[n * I + i];
            hid[n * I + i] = sigmoid(g) * u;
        }
    }

    linear_forward(hid, w_down, output, N, I, H);

    free(gate);
    free(up);
    free(hid);
}

/* Run one case directory: returns max_abs_diff */
static double run_case(const char* base_dir, const char* case_name) {
    char path[MAX_PATH];
    Meta meta;

    /* meta.json */
    snprintf(path, sizeof(path), "%s/%s/meta.json", base_dir, case_name);
    parse_meta(path, &meta);

    int B = meta.batch_size;
    int S = meta.seq_len;
    int H = meta.hidden_size;
    int I = meta.intermediate_size;
    int E = meta.n_routed_experts;
    int N_SHARED = meta.n_shared_experts;
    int TOP_K = meta.top_k;

    int N = B * S;

    if (E > MAX_EXPERTS || N_SHARED > MAX_SHARED) {
        fprintf(stderr, "Too many experts: E=%d, N_SHARED=%d\n", E, N_SHARED);
        exit(1);
    }

    /* Load inputs/outputs/router data */
    snprintf(path, sizeof(path), "%s/%s/inputs.bin", base_dir, case_name);
    float* inputs = load_f32(path, (size_t)B * S * H);
    snprintf(path, sizeof(path), "%s/%s/outputs.bin", base_dir, case_name);
    float* expected = load_f32(path, (size_t)B * S * H);

    snprintf(path, sizeof(path), "%s/%s/topk_indices.bin", base_dir, case_name);
    int* topk_idx = load_i32(path, (size_t)B * S * TOP_K);
    snprintf(path, sizeof(path), "%s/%s/topk_weights.bin", base_dir, case_name);
    float* topk_w = load_f32(path, (size_t)B * S * TOP_K);

    /* router weight (unused in forward) */
    snprintf(path, sizeof(path), "%s/%s/router_weight.bin", base_dir, case_name);
    float* router_weight = load_f32(path, (size_t)E * H);
    (void)router_weight;

    /* shared experts */
    float* shared_gate[MAX_SHARED];
    float* shared_up[MAX_SHARED];
    float* shared_down[MAX_SHARED];

    for (int s = 0; s < N_SHARED; ++s) {
        snprintf(path, sizeof(path), "%s/%s/shared_%d_gate_proj_weight.bin", base_dir, case_name, s);
        shared_gate[s] = load_f32(path, (size_t)I * H);
        snprintf(path, sizeof(path), "%s/%s/shared_%d_up_proj_weight.bin", base_dir, case_name, s);
        shared_up[s] = load_f32(path, (size_t)I * H);
        snprintf(path, sizeof(path), "%s/%s/shared_%d_down_proj_weight.bin", base_dir, case_name, s);
        shared_down[s] = load_f32(path, (size_t)H * I);
    }

    /* routed experts - FIXED: using fixed-size arrays */
    float* expert_gate[MAX_EXPERTS];
    float* expert_up[MAX_EXPERTS];
    float* expert_down[MAX_EXPERTS];

    for (int e = 0; e < E; ++e) {
        snprintf(path, sizeof(path), "%s/%s/expert_%d_gate_proj_weight.bin", base_dir, case_name, e);
        expert_gate[e] = load_f32(path, (size_t)I * H);
        snprintf(path, sizeof(path), "%s/%s/expert_%d_up_proj_weight.bin", base_dir, case_name, e);
        expert_up[e] = load_f32(path, (size_t)I * H);
        snprintf(path, sizeof(path), "%s/%s/expert_%d_down_proj_weight.bin", base_dir, case_name, e);
        expert_down[e] = load_f32(path, (size_t)H * I);
    }

    /* shared_out */
    float* shared_out = (float*)calloc((size_t)N * H, sizeof(float));
    float* tmp = (float*)malloc((size_t)N * H * sizeof(float));
    if (!shared_out || !tmp) {
        fprintf(stderr, "OOM shared/tmp\n");
        exit(1);
    }

    for (int s = 0; s < N_SHARED; ++s) {
        tiny_mlp_forward(inputs, shared_gate[s], shared_up[s], shared_down[s], tmp, N, H, I);
        for (int i = 0; i < N * H; ++i)
            shared_out[i] += tmp[i];
    }

    /* all expert outputs on all tokens */
    float* expert_out = (float*)malloc((size_t)N * E * H * sizeof(float));
    if (!expert_out) {
        fprintf(stderr, "OOM expert_out\n");
        exit(1);
    }
    for (int e = 0; e < E; ++e) {
        tiny_mlp_forward(inputs, expert_gate[e], expert_up[e], expert_down[e], tmp, N, H, I);
        for (int n = 0; n < N; ++n) {
            float* dst = expert_out + (n * E * H + e * H);
            float* src = tmp + n * H;
            memcpy(dst, src, (size_t)H * sizeof(float));
        }
    }

    /* combine routed */
    float* routed_out = (float*)calloc((size_t)N * H, sizeof(float));
    if (!routed_out) {
        fprintf(stderr, "OOM routed_out\n");
        exit(1);
    }

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < TOP_K; ++k) {
            int e_idx = topk_idx[n * TOP_K + k];
            float w = topk_w[n * TOP_K + k];
            const float* y = expert_out + (n * E * H + e_idx * H);
            float* out = routed_out + n * H;
            for (int h = 0; h < H; ++h)
                out[h] += w * y[h];
        }
    }

    float* final_out = (float*)malloc((size_t)N * H * sizeof(float));
    if (!final_out) {
        fprintf(stderr, "OOM final_out\n");
        exit(1);
    }
    for (int i = 0; i < N * H; ++i) {
        final_out[i] = inputs[i] + shared_out[i] + routed_out[i];
    }

    double max_abs_diff = 0.0;
    for (int i = 0; i < N * H; ++i) {
        double diff = fabs((double)final_out[i] - (double)expected[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
    }

    /* cleanup */
    free(inputs);
    free(expected);
    free(topk_idx);
    free(topk_w);
    free(router_weight);
    for (int s = 0; s < N_SHARED; ++s) {
        free(shared_gate[s]);
        free(shared_up[s]);
        free(shared_down[s]);
    }
    for (int e = 0; e < E; ++e) {
        free(expert_gate[e]);
        free(expert_up[e]);
        free(expert_down[e]);
    }
    free(shared_out);
    free(tmp);
    free(expert_out);
    free(routed_out);
    free(final_out);

    return max_abs_diff;
}

int main() {
    const char* base_dir = "deepseek_moe_tests_multi";

    /* Keep in sync with Python test_specs */
    const char* cases[] = { "case_01", "case_02", "case_03" };
    const int num_cases = 3;

    double global_max = 0.0;
    int all_ok = 1;
    const double tol = 1e-5;

    for (int i = 0; i < num_cases; ++i) {
        const char* name = cases[i];
        double d = run_case(base_dir, name);
        printf("Case %s: max abs diff = %.9f\n", name, d);
        if (d > global_max) global_max = d;
        if (d > tol) all_ok = 0;
    }

    printf("Global max abs diff = %.9f\n", global_max);
    if (all_ok) {
        printf("All MoE tests PASSED.\n");
        return 0;
    } else {
        printf("Some MoE tests FAILED.\n");
        return 1;
    }
}