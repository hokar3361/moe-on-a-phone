import SwiftUI

@main
struct MoEBenchApp: App {
    var body: some Scene {
        WindowGroup {
            // launched with benchmark arguments (devicectl): the log screen; otherwise the chat demo
            if ProcessInfo.processInfo.arguments.contains(where: { $0 == "--model" || $0 == "--iotest" || $0 == "--vatest" }) {
                ContentView()
            } else {
                ChatView()
            }
        }
    }
}

@MainActor
final class BenchState: ObservableObject {
    @Published var log = ""
    @Published var models: [String] = []
    @Published var running = false

    init() {
        signal(SIGPIPE, SIG_IGN) // console may detach; don't die writing to stderr
        refresh()
        let free = (try? documentsURL().resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        let freeLine = String(format: "storage free: %.1f GB", Double(free) / 1e9)
        fputs(freeLine + "\n", stderr); fflush(stderr)
        log += freeLine + "\n"
        // Launched with arguments (e.g. from `devicectl device process launch`): run right away.
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "--iotest"), i + 1 < args.count {
            let file = args[i + 1]
            running = true
            UIApplication.shared.isIdleTimerDisabled = true
            Thread.detachNewThread {
                runIOTest(file: file) { line in
                    fputs(line + "\n", stderr); fflush(stderr)
                    Task { @MainActor in self.log += line + "\n" }
                }
                Task { @MainActor in self.running = false }
            }
            return
        }
        if let i = args.firstIndex(of: "--vatest") {
            let files = Array(args[(i + 1)...])
            Thread.detachNewThread {
                runVATest(files: files) { line in
                    fputs(line + "\n", stderr); fflush(stderr)
                    Task { @MainActor in self.log += line + "\n" }
                }
            }
            return
        }
        if let cfg = BenchConfig.fromArguments(ProcessInfo.processInfo.arguments) {
            start(cfg)
        }
    }

    func refresh() {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: documentsURL().path)) ?? []
        // split models: list only the first shard
        models = files.filter { $0.hasSuffix(".gguf") && !$0.contains("-0000") || $0.contains("-00001-of-") }.sorted()
    }

    func start(_ cfg: BenchConfig) {
        guard !running else { return }
        running = true
        UIApplication.shared.isIdleTimerDisabled = true
        Thread.detachNewThread {
            let runner = BenchRunner { line in
                fputs(line + "\n", stderr)
                fflush(stderr)
                Task { @MainActor in self.log += line + "\n" }
            }
            runner.run(cfg)
            Task { @MainActor in
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }
}

struct ContentView: View {
    @StateObject var state = BenchState()

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading) {
                List(state.models, id: \.self) { m in
                    Button(m) {
                        var c = BenchConfig()
                        c.model = m
                        state.start(c)
                    }
                    .disabled(state.running)
                }
                .frame(maxHeight: 200)
                ScrollView {
                    Text(state.log)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(8)
                }
            }
            .navigationTitle(state.running ? "Running..." : "MoEBench")
            .toolbar {
                Button("Refresh") { state.refresh() }
                Button("Copy") { UIPasteboard.general.string = state.log }
            }
        }
    }
}

@MainActor
final class ChatState: ObservableObject {
    @Published var models: [String] = []
    @Published var selected = ""
    @Published var output = ""
    @Published var status = "Ready"
    @Published var busy = false
    @Published var cacheMiB = 3000
    @Published var thermal = ""

    private let engine = ChatEngine()
    private let queue = DispatchQueue(label: "chat", qos: .userInitiated)
    private var timer: Timer?

    init() {
        signal(SIGPIPE, SIG_IGN)
        refresh()
        updateThermal()
        timer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.updateThermal() }
        }
    }

    func refresh() {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: documentsURL().path)) ?? []
        models = files.filter { $0.hasSuffix(".gguf") && !$0.contains("-0000") || $0.contains("-00001-of-") }.sorted()
        if selected.isEmpty, let first = models.first(where: { $0.contains("Qwen3.6") }) ?? models.first {
            selected = first
        }
    }

    func updateThermal() {
        let names = ["nominal", "fair", "serious", "critical"]
        let t = ProcessInfo.processInfo.thermalState.rawValue
        let mem = MemInfo.now()
        thermal = String(format: "thermal %@ · app %.1f GB · free %.1f GB", t < names.count ? names[t] : "?",
                         Double(mem.footprint) / 1073741824.0, Double(mem.available) / 1073741824.0)
    }

    func send(_ prompt: String) {
        guard !busy, !selected.isEmpty, !prompt.isEmpty else { return }
        busy = true
        output = ""
        UIApplication.shared.isIdleTimerDisabled = true
        let file = selected, cache = cacheMiB
        queue.async {
            if self.engine.loadedName != file || self.engine.loadedCacheMiB != cache {
                Task { @MainActor in self.status = "Loading \(file)..." }
                let ok = self.engine.load(file: file, cacheMiB: cache) { line in
                    Task { @MainActor in self.status = line }
                }
                if !ok {
                    Task { @MainActor in self.busy = false; UIApplication.shared.isIdleTimerDisabled = false }
                    return
                }
            }
            Task { @MainActor in self.status = "Reading the prompt..." }
            var lastUI = Date.distantPast
            var chunk = ""
            let st = self.engine.generate(prompt: prompt, maxTokens: 400) { piece, st in
                chunk += piece
                if Date().timeIntervalSince(lastUI) > 0.05 {
                    lastUI = Date()
                    let c = chunk; chunk = ""
                    Task { @MainActor in
                        self.output += c
                        self.status = String(format: "%.1f tok/s · %d tokens", st.tps, st.nGen)
                    }
                }
            }
            let c = chunk
            Task { @MainActor in
                self.output += c
                self.status = String(format: "%.2f tok/s · %d tokens · prompt %.1f s", st.tps, st.nGen, st.promptSeconds)
                self.busy = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    func stop() { engine.stop() }
}

struct ChatView: View {
    @StateObject var state = ChatState()
    @State private var prompt = ""

    private let presets = [
        ("Code", "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."),
        ("Story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ]

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 8) {
                Picker("Model", selection: $state.selected) {
                    ForEach(state.models, id: \.self) { Text($0).lineLimit(1).truncationMode(.middle) }
                }
                .pickerStyle(.menu)
                .disabled(state.busy)

                HStack {
                    Text(state.status).font(.system(.headline, design: .monospaced))
                    Spacer()
                }
                Text(state.thermal).font(.caption).foregroundStyle(.secondary)

                ScrollViewReader { proxy in
                    ScrollView {
                        Text(state.output)
                            .font(.system(size: 14))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .padding(8)
                        Color.clear.frame(height: 1).id("end")
                    }
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .onChange(of: state.output) { proxy.scrollTo("end", anchor: .bottom) }
                }

                HStack {
                    ForEach(presets, id: \.0) { p in
                        Button(p.0) { prompt = p.1 }.buttonStyle(.bordered).disabled(state.busy)
                    }
                    Spacer()
                    Stepper("cache \(state.cacheMiB) MiB", value: $state.cacheMiB, in: 1000...3800, step: 200)
                        .font(.caption).disabled(state.busy)
                }
                HStack {
                    TextField("Prompt", text: $prompt, axis: .vertical)
                        .lineLimit(1...4)
                        .textFieldStyle(.roundedBorder)
                    if state.busy {
                        Button("Stop") { state.stop() }.buttonStyle(.borderedProminent).tint(.red)
                    } else {
                        Button("Send") { state.send(prompt) }.buttonStyle(.borderedProminent)
                    }
                }
            }
            .padding()
            .navigationTitle("MoE on a phone")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { Button("Refresh") { state.refresh() } }
        }
    }
}
