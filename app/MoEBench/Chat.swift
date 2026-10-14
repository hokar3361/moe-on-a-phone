import Foundation
import llama

// Interactive generation for the demo screen: the model stays loaded, each prompt streams greedy tokens.
final class ChatEngine {
    struct Stats {
        var nGen = 0
        var tps = 0.0
        var promptSeconds = 0.0
    }

    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var smpl: UnsafeMutablePointer<llama_sampler>?
    private var pattern: UnsafeMutablePointer<CChar>?
    private(set) var loadedName = ""
    private(set) var loadedCacheMiB = 0
    private var stopRequested = false

    deinit { unload() }

    func unload() {
        if let smpl { llama_sampler_free(smpl) }
        if let ctx { llama_free(ctx) }
        if let model { llama_model_free(model) }
        if let pattern { free(pattern) }
        smpl = nil; ctx = nil; model = nil; pattern = nil
        loadedName = ""
    }

    func stop() { stopRequested = true }

    // best configuration found so far: experts in CPU memory, 3 GB GPU expert cache filled with parallel F_NOCACHE pread
    func load(file: String, cacheMiB: Int, log: (String) -> Void) -> Bool {
        unload()
        setenv("LLAMA_MOE_CACHE_PREAD", "1", 1)
        setenv("LLAMA_MOE_CACHE_PREDICT", "0", 1)
        setenv("LLAMA_MOE_PREFETCH", "0", 1)
        llama_backend_init()

        var mp = llama_model_default_params()
        mp.load_mode = LLAMA_LOAD_MODE_MMAP
        pattern = strdup("\\.ffn_(up|down|gate|gate_up)_exps")
        var overrides = [
            llama_model_tensor_buft_override(pattern: pattern,
                buft: ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))),
            llama_model_tensor_buft_override(pattern: nil, buft: nil),
        ]
        let path = documentsURL().appendingPathComponent(file).path
        let t0 = Date()
        model = overrides.withUnsafeMutableBufferPointer { ov in
            mp.tensor_buft_overrides = UnsafePointer(ov.baseAddress)
            return llama_model_load_from_file(path, mp)
        }
        guard let model else { log("failed to load \(file)"); return false }

        var cp = llama_context_default_params()
        cp.n_ctx = 4096
        cp.n_batch = 512
        cp.n_threads = 4
        cp.n_threads_batch = 4
        cp.moe_cache_size = cacheMiB * 1024 * 1024
        ctx = llama_init_from_model(model, cp)
        guard ctx != nil else { log("failed to create the context (try a smaller cache)"); unload(); return false }

        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        loadedName = file
        loadedCacheMiB = cacheMiB
        log(String(format: "loaded %@ in %.1f s, cache %d MiB", file, Date().timeIntervalSince(t0), cacheMiB))
        return true
    }

    func generate(prompt: String, maxTokens: Int, onText: (String, Stats) -> Void) -> Stats {
        var st = Stats()
        guard let model, let ctx, let smpl else { return st }
        stopRequested = false
        let vocab = llama_model_get_vocab(model)
        llama_memory_clear(llama_get_memory(ctx), true)
        llama_sampler_reset(smpl)

        let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let n = Int32(text.utf8.count + 8)
        var toks = [llama_token](repeating: 0, count: Int(n))
        let k = llama_tokenize(vocab, text, Int32(text.utf8.count), &toks, n, true, true)
        toks = Array(toks.prefix(Int(max(0, k))))

        let t0 = Date()
        if llama_decode(ctx, llama_batch_get_one(&toks, Int32(toks.count))) != 0 { return st }
        st.promptSeconds = Date().timeIntervalSince(t0)

        var pending: [UInt8] = []
        var nDecode = 0
        let tGen0 = Date()
        while st.nGen < maxTokens && !stopRequested {
            var tok = llama_sampler_sample(smpl, ctx, -1)
            if llama_vocab_is_eog(vocab, tok) { break }
            var buf = [CChar](repeating: 0, count: 256)
            let m = llama_token_to_piece(vocab, tok, &buf, Int32(buf.count), 0, false)
            pending += buf.prefix(Int(max(0, m))).map { UInt8(bitPattern: $0) }
            st.nGen += 1
            if llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0 { break }
            nDecode += 1
            st.tps = Double(nDecode) / Date().timeIntervalSince(tGen0)
            if let s = String(validating: pending, as: UTF8.self) {
                pending.removeAll()
                onText(s, st)
            }
        }
        return st
    }
}
