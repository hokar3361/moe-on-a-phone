import Foundation
import os
import llama

// Fixed measurement (same as the Mac Studio runs): no thinking, greedy, 400 tokens, decode tok/s.
let benchPrompts: [(String, String)] = [
    ("code", "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."),
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
]

struct BenchConfig {
    var model = ""            // file name in Documents (first shard for split models)
    var prompts = ["code", "story"]
    var nGen = 400
    var nCtx = 2048
    var nGpuLayers: Int32 = -1
    var loadMode = LLAMA_LOAD_MODE_MMAP
    var stopAtEOG = true
    var moeCacheMiB = 0       // upstream GPU-side LRU of CPU-resident experts (llama.cpp --moe-cache-mib)
    var pread = false         // our patch: fill the MoE cache with parallel F_NOCACHE pread (LLAMA_MOE_CACHE_PREAD)
    var nUbatch = 512         // prompt micro-batch; small values keep every batch inside the MoE cache
    var repack = true         // CPU repacked weight layouts (they read the data at load time)
    var unmappedExperts = false // LLAMA_UNMAPPED_EXPERT_FILES: don't map expert-only files (needs --pread)
    var timing = false        // GGML_METAL_TIMING: average GPU span / wall time per graph every 64 graphs
    var bellSpin = false      // GGML_METAL_BELL_SPIN: the bell worker spins instead of using the event listener
    var bell = false          // LLAMA_MOE_CACHE_BELL: MoE cache lookups wait for the CPU inside the command buffer
    var spin = false          // GGML_METAL_SPIN_WAIT: poll the GPU instead of blocking
    var mtp = -1              // >= 0: generate with llama.cpp common speculative decoding, N MTP draft tokens (0 = none)
    var cachePredict = 0      // our patch: read the top-k predicted experts of the next layer into the MoE cache
    var prefetch = 0          // our patch: warm the page cache with the selected experts (1 pread, 2 madvise, 3 touch)
    var predict = 0           // our patch: also read the top-k predicted experts of the next layer
    var cpuMoE = false        // keep expert tensors on CPU (mmap), rest on GPU; like llama.cpp --cpu-moe

    // --model X --prompts code,story --n 400 --ctx 2048 --ngl -1 --load mmap|none|direct --no-eog-stop --cpu-moe --moe-cache-mib N --pread --prefetch M --predict K --mtp N
    static func fromArguments(_ args: [String]) -> BenchConfig? {
        var c = BenchConfig()
        var i = 1
        func next() -> String? { i += 1; return i < args.count ? args[i] : nil }
        while i < args.count {
            switch args[i] {
            case "--model":   c.model = next() ?? ""
            case "--prompts": c.prompts = (next() ?? "").split(separator: ",").map(String.init)
            case "--n":       c.nGen = Int(next() ?? "") ?? c.nGen
            case "--ctx":     c.nCtx = Int(next() ?? "") ?? c.nCtx
            case "--ngl":     c.nGpuLayers = Int32(next() ?? "") ?? c.nGpuLayers
            case "--load":    c.loadMode = llama_load_mode_from_str(next() ?? "mmap")
            case "--no-eog-stop": c.stopAtEOG = false
            case "--cpu-moe": c.cpuMoE = true
            case "--pread": c.pread = true
            case "--ubatch": c.nUbatch = Int(next() ?? "") ?? c.nUbatch
            case "--no-repack": c.repack = false
            case "--unmapped-experts": c.unmappedExperts = true
            case "--bell": c.bell = true
            case "--bell-spin": c.bellSpin = true
            case "--timing": c.timing = true
            case "--spin": c.spin = true
            case "--mtp": c.mtp = Int(next() ?? "") ?? 0
            case "--cache-predict": c.cachePredict = Int(next() ?? "") ?? 0
            case "--prefetch": c.prefetch = Int(next() ?? "") ?? 0
            case "--predict": c.predict = Int(next() ?? "") ?? 0
            case "--moe-cache-mib": c.moeCacheMiB = Int(next() ?? "") ?? 0
            default: break
            }
            i += 1
        }
        return c.model.isEmpty ? nil : c
    }
}

func documentsURL() -> URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
}

struct MemInfo {
    var footprint: UInt64 = 0
    var footprintPeak: UInt64 = 0
    var available: UInt64 = 0

    static func now() -> MemInfo {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        var m = MemInfo()
        if kr == KERN_SUCCESS {
            m.footprint = info.phys_footprint
            m.footprintPeak = UInt64(info.ledger_phys_footprint_peak)
        }
        m.available = UInt64(os_proc_available_memory())
        return m
    }

    var text: String {
        let g = { (v: UInt64) in String(format: "%.2f", Double(v) / 1073741824.0) }
        return "footprint \(g(footprint)) GiB (peak \(g(footprintPeak))), available \(g(available)) GiB"
    }
}

final class BenchRunner {
    let log: (String) -> Void
    init(log: @escaping (String) -> Void) { self.log = log }

    func run(_ cfg: BenchConfig) {
        let path = documentsURL().appendingPathComponent(cfg.model).path
        log("model: \(cfg.model)")
        log("mem before load: \(MemInfo.now().text)")

        setenv("GGML_METAL_SPIN_WAIT", cfg.spin ? "1" : "0", 1)
        setenv("LLAMA_MOE_CACHE_BELL", cfg.bell ? "1" : "0", 1)
        setenv("GGML_METAL_TIMING", cfg.timing ? "1" : "0", 1)
        setenv("GGML_METAL_BELL_SPIN", cfg.bellSpin ? "1" : "0", 1)
        setenv("LLAMA_UNMAPPED_EXPERT_FILES", cfg.unmappedExperts ? "1" : "0", 1)
        log("spin wait \(cfg.spin), bell \(cfg.bell), unmapped experts \(cfg.unmappedExperts), ubatch \(cfg.nUbatch)")
        if cfg.mtp >= 0 {
            runSpec(cfg, path: path)
            return
        }
        llama_backend_init()
        var mp = llama_model_default_params()
        mp.n_gpu_layers = cfg.nGpuLayers
        mp.load_mode = cfg.loadMode
        mp.use_extra_bufts = cfg.repack && !cfg.unmappedExperts
        let pattern = strdup("\\.ffn_(up|down|gate)_exps")
        defer { free(pattern) }
        var overrides = [
            llama_model_tensor_buft_override(pattern: pattern,
                buft: ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))),
            llama_model_tensor_buft_override(pattern: nil, buft: nil),
        ]
        log("ngl \(cfg.nGpuLayers), cpu-moe \(cfg.cpuMoE), load \(String(cString: llama_load_mode_name(cfg.loadMode)))")
        let tLoad0 = Date()
        let loaded = overrides.withUnsafeMutableBufferPointer { ov -> OpaquePointer? in
            if cfg.cpuMoE { mp.tensor_buft_overrides = UnsafePointer(ov.baseAddress) }
            return llama_model_load_from_file(path, mp)
        }
        guard let model = loaded else {
            log("ERROR: failed to load model")
            return
        }
        let tLoad = Date().timeIntervalSince(tLoad0)
        defer { llama_model_free(model) }

        var desc = [CChar](repeating: 0, count: 256)
        llama_model_desc(model, &desc, desc.count)
        log(String(format: "loaded %@ (%.2f GiB, %.2f B params) in %.1f s",
                   String(cString: desc), Double(llama_model_size(model)) / 1073741824.0,
                   Double(llama_model_n_params(model)) / 1e9, tLoad))
        log("mem after load: \(MemInfo.now().text)")

        var cp = llama_context_default_params()
        cp.n_ctx = UInt32(cfg.nCtx)
        cp.n_batch = 512
        cp.n_ubatch = UInt32(min(512, cfg.nUbatch))
        let nThreads = Int32(max(1, min(6, ProcessInfo.processInfo.activeProcessorCount - 2)))
        cp.n_threads = nThreads
        cp.n_threads_batch = nThreads
        cp.moe_cache_size = cfg.moeCacheMiB * 1024 * 1024
        setenv("LLAMA_MOE_CACHE_PREAD", cfg.pread ? "1" : "0", 1)
        setenv("LLAMA_MOE_CACHE_PREDICT", String(cfg.cachePredict), 1)
        setenv("LLAMA_MOE_PREFETCH", String(cfg.prefetch), 1)
        setenv("LLAMA_MOE_PREDICT", String(cfg.predict), 1)
        log("moe cache \(cfg.moeCacheMiB) MiB, pread \(cfg.pread), cache-predict \(cfg.cachePredict), prefetch \(cfg.prefetch), predict \(cfg.predict)")
        guard let ctx = llama_init_from_model(model, cp) else {
            log("ERROR: failed to create context")
            return
        }
        defer { llama_free(ctx) }
        log("mem after context: \(MemInfo.now().text)")

        let vocab = llama_model_get_vocab(model)
        let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())!
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
        defer { llama_sampler_free(smpl) }

        for name in cfg.prompts {
            guard let p = benchPrompts.first(where: { $0.0 == name }) else { continue }
            if !waitForCool() { log("ERROR: still hot after 10 min, stopping"); return }
            let thermal0 = thermalName()
            llama_memory_clear(llama_get_memory(ctx), true)
            llama_sampler_reset(smpl)
            llama_perf_context_reset(ctx)

            // ChatML with thinking disabled (empty think block), as on the Mac Studio.
            let text = "<|im_start|>user\n\(p.1)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            var toks = tokenize(vocab, text)

            let t0 = Date()
            if llama_decode(ctx, llama_batch_get_one(&toks, Int32(toks.count))) != 0 {
                log("ERROR: prompt decode failed"); return
            }
            let tPrompt = Date().timeIntervalSince(t0)

            var out = ""
            var nOut = 0
            var nDecode = 0
            let tGen0 = Date()
            while nOut < cfg.nGen {
                var tok = llama_sampler_sample(smpl, ctx, -1)
                if cfg.stopAtEOG && llama_vocab_is_eog(vocab, tok) { break }
                out += piece(vocab, tok)
                nOut += 1
                if nOut == cfg.nGen { break }
                if llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0 {
                    log("ERROR: decode failed at \(nOut)"); break
                }
                nDecode += 1
            }
            let tGen = Date().timeIntervalSince(tGen0)
            // decode tok/s over the single-token decode calls (excludes prompt processing)
            let tps = nDecode > 0 ? Double(nDecode) / tGen : 0
            let mem = MemInfo.now()
            log(String(format: "[%@] prompt %d tok in %.2f s (%.1f tok/s) | gen %d tok in %.2f s -> decode %.2f tok/s",
                       name, toks.count, tPrompt, Double(toks.count) / tPrompt, nOut, tGen, tps))
            log("mem: \(mem.text) | thermal \(thermal0) -> \(thermalName())")
            log("--- output [\(name)] ---\n\(out)\n--- end ---")

            let rec: [String: Any] = [
                "date": ISO8601DateFormatter().string(from: Date()),
                "model": cfg.model, "prompt": name, "n_prompt": toks.count,
                "t_prompt_s": tPrompt, "n_gen": nOut, "t_gen_s": tGen, "decode_tps": tps,
                "footprint_gib": Double(mem.footprint) / 1073741824.0,
                "footprint_peak_gib": Double(mem.footprintPeak) / 1073741824.0,
                "available_gib": Double(mem.available) / 1073741824.0,
                "load_mode": String(cString: llama_load_mode_name(cfg.loadMode)),
                "n_ctx": cfg.nCtx, "ngl": cfg.nGpuLayers, "n_threads": nThreads, "cpu_moe": cfg.cpuMoE, "bell": cfg.bell, "unmapped_experts": cfg.unmappedExperts, "n_ubatch": cfg.nUbatch, "cache_predict": cfg.cachePredict, "prefetch": cfg.prefetch, "predict": cfg.predict,
                "thermal_start": thermal0, "thermal_end": thermalName(), "moe_cache_mib": cfg.moeCacheMiB, "pread": cfg.pread,
                "output": out,
            ]
            appendResult(rec)
        }
        log("DONE")
    }

    // generation through common/speculative (Spec.cpp): same prompts, greedy, optional MTP drafts
    private func runSpec(_ cfg: BenchConfig, path: String) {
        setenv("LLAMA_MOE_CACHE_PREAD", cfg.pread ? "1" : "0", 1)
        setenv("LLAMA_MOE_CACHE_PREDICT", String(cfg.cachePredict), 1)
        log("spec: mtp \(cfg.mtp), moe cache \(cfg.moeCacheMiB) MiB, cpu-moe \(cfg.cpuMoE), pread \(cfg.pread)")
        for name in cfg.prompts {
            guard let p = benchPrompts.first(where: { $0.0 == name }) else { continue }
            if !waitForCool() { log("ERROR: still hot after 10 min, stopping"); return }
            let thermal0 = thermalName()
            let text = "<|im_start|>user\n\(p.1)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            let r: moe_spec_result = path.withCString { mpath in
                text.withCString { ptext in
                    var a = moe_spec_args(model_path: mpath, prompt: ptext, n_predict: Int32(cfg.nGen), n_draft: Int32(cfg.mtp),
                                          n_ctx: Int32(cfg.nCtx), n_threads: 4, cpu_moe: cfg.cpuMoE ? 1 : 0,
                                          moe_cache_mib: Int32(cfg.moeCacheMiB))
                    return moe_spec_run(&a)
                }
            }
            guard r.ok != 0 else { log("ERROR: spec run failed"); return }
            let out = r.text != nil ? String(cString: r.text) : ""
            free(r.text)
            let tps = Double(r.n_gen) / r.t_gen_s
            let mem = MemInfo.now()
            log(String(format: "[%@] prompt %d tok in %.2f s | gen %d tok in %.2f s -> decode %.2f tok/s | drafted %d accepted %d",
                       name, r.n_prompt, r.t_prompt_s, r.n_gen, r.t_gen_s, tps, r.n_drafted, r.n_accepted))
            log("mem: \(mem.text) | thermal \(thermal0) -> \(thermalName())")
            log("--- output [\(name)] ---\n\(out)\n--- end ---")
            appendResult([
                "date": ISO8601DateFormatter().string(from: Date()), "model": cfg.model, "prompt": name,
                "mode": "spec", "mtp": cfg.mtp, "n_gen": Int(r.n_gen), "t_gen_s": r.t_gen_s, "decode_tps": tps,
                "n_drafted": Int(r.n_drafted), "n_accepted": Int(r.n_accepted), "moe_cache_mib": cfg.moeCacheMiB,
                "pread": cfg.pread, "spin": cfg.spin, "unmapped_experts": cfg.unmappedExperts, "bell": cfg.bell, "n_ubatch": cfg.nUbatch, "thermal_start": thermal0, "thermal_end": thermalName(),
                "footprint_gib": Double(mem.footprint) / 1073741824.0, "output": out,
            ])
        }
        log("DONE")
    }

    private func thermalName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    // wait until the phone is back to nominal or fair (at most 10 min); false if it stays hot
    private func waitForCool() -> Bool {
        let t0 = Date()
        while ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
            if Date().timeIntervalSince(t0) > 600 { return false }
            log("thermal \(thermalName()), cooling down...")
            Thread.sleep(forTimeInterval: 30)
        }
        return true
    }

    private func tokenize(_ vocab: OpaquePointer?, _ text: String) -> [llama_token] {
        let n = Int32(text.utf8.count + 8)
        var buf = [llama_token](repeating: 0, count: Int(n))
        let k = llama_tokenize(vocab, text, Int32(text.utf8.count), &buf, n, true, true)
        return Array(buf.prefix(Int(max(0, k))))
    }

    private var pending: [CChar] = []
    private func piece(_ vocab: OpaquePointer?, _ tok: llama_token) -> String {
        var buf = [CChar](repeating: 0, count: 256)
        let k = llama_token_to_piece(vocab, tok, &buf, Int32(buf.count), 0, false)
        pending += buf.prefix(Int(max(0, k)))
        // hold incomplete UTF-8 sequences until the next token completes them
        if let s = String(validating: pending.map { UInt8(bitPattern: $0) }, as: UTF8.self) {
            pending.removeAll()
            return s
        }
        return ""
    }

    private func appendResult(_ rec: [String: Any]) {
        let url = documentsURL().appendingPathComponent("results.jsonl")
        guard var data = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]) else { return }
        data.append(0x0A)
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(data); try? h.close()
        } else {
            try? data.write(to: url)
        }
    }
}
