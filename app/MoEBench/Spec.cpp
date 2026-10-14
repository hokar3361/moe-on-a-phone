// Port of llama.cpp examples/speculative-simple to a function the app can call.
#include "Spec.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" struct moe_spec_result moe_spec_run(const struct moe_spec_args * a) {
    moe_spec_result res = {};

    common_params params;
    params.model.path        = a->model_path;
    params.prompt            = a->prompt;
    params.n_predict         = a->n_predict;
    params.n_ctx             = a->n_ctx;
    params.cpuparams.n_threads       = a->n_threads;
    params.cpuparams_batch.n_threads = a->n_threads;
    params.sampling.temp     = 0.0f;
    params.sampling.top_k    = 1;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K }; // greedy
    params.sampling.user_sampling_config = ~0u;                  // don't let the GGUF sampling defaults override these
    params.moe_cache_size    = (size_t) a->moe_cache_mib*1024*1024;
    if (a->cpu_moe) {
        params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
    }
    params.tensor_buft_overrides.push_back({ nullptr, nullptr });
    params.speculative.draft.tensor_buft_overrides.push_back({ nullptr, nullptr });
    if (a->n_draft > 0) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = a->n_draft;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max         = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        return res;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    common_speculative_init_result_ptr spec_init;
    llama_context * ctx_dft = nullptr;
    if (a->n_draft > 0) {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
        ctx_dft = params.speculative.draft.ctx_dft;
    }

    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = ctx_dft != nullptr && common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    res.n_prompt = (int) inp.size();

    const llama_seq_id seq_id = 0;
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    common_speculative * spec = a->n_draft > 0 ? common_speculative_init(params.speculative, 1) : nullptr;

    const auto t_enc_start = ggml_time_us();
    {
        common_batch batch_prompt(ctx_tgt);
        for (size_t i = 0; i < inp.size() - 1; ++i) {
            batch_prompt.add(inp[i], i, seq_id, false);
        }
        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());
        if (spec && !common_speculative_process(spec, batch_prompt)) {
            return res;
        }
    }
    const auto t_enc_end = ggml_time_us();

    llama_token id_last = inp.back();
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
    int n_past = (int) inp.size() - 1;
    if (spec) {
        common_speculative_begin(spec, seq_id, prompt_tgt);
    }

    common_batch batch_tgt(ctx_tgt);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;
    std::string text;
    int n_gen = 0;
    bool done = false;

    const auto t_dec_start = ggml_time_us();
    while (!done) {
        if (spec && draft.empty()) {
            ckpt.update_pos(prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));
            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            int n_draft_max = std::max(0, std::min((int) llama_n_ctx(ctx_tgt) - n_past - 2, a->n_predict - n_gen - 1));
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ n_draft_max,
                /* .pos0       = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft,
            };
            common_speculative_draft(spec);
            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
        }

        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + i, seq_id, true);
        }
        llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        if (spec && !common_speculative_process(spec, batch_tgt)) {
            break;
        }

        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        const size_t n_draft = draft.size();
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
            // partial acceptance on a model without partial sequence removal: roll back and redo
            draft = std::move(ids);
            ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            if (ctx_dft) {
                ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
            }
            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        if (spec) {
            common_speculative_accept(spec, seq_id, (uint16_t) (ids.size() - 1));
        }
        n_past         += (int) ids.size() - 1;
        res.n_drafted  += (int) n_draft;
        res.n_accepted += (int) ids.size() - 1;

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];
            if (llama_vocab_is_eog(vocab, id_last) || n_gen >= a->n_predict) {
                done = true;
                break;
            }
            text += common_token_to_piece(ctx_tgt, id_last);
            n_gen++;
        }
        draft.clear();
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
        }
        if (n_gen >= a->n_predict) {
            done = true;
        }
    }
    const auto t_dec_end = ggml_time_us();

    if (spec) {
        common_speculative_print_stats(spec);
        common_speculative_free(spec);
    }

    res.ok         = 1;
    res.n_gen      = n_gen;
    res.t_prompt_s = (t_enc_end - t_enc_start)/1e6;
    res.t_gen_s    = (t_dec_end - t_dec_start)/1e6;
    res.text       = strdup(text.c_str());
    return res;
}
