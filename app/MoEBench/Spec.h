#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// greedy generation with MTP speculative decoding (llama.cpp common/speculative, as in examples/speculative-simple)
struct moe_spec_args {
    const char * model_path;
    const char * prompt;       // already formatted with the chat template
    int n_predict;
    int n_draft;               // max draft tokens per step (0 = no speculation)
    int n_ctx;
    int n_threads;
    int cpu_moe;               // experts in CPU memory (mmap)
    int moe_cache_mib;         // upstream MoE cache size of the target context
};

struct moe_spec_result {
    int    ok;
    int    n_prompt;
    int    n_gen;
    int    n_drafted;
    int    n_accepted;
    double t_prompt_s;
    double t_gen_s;
    char * text;               // malloc'd, caller frees
};

struct moe_spec_result moe_spec_run(const struct moe_spec_args * args);

#ifdef __cplusplus
}
#endif
