import Foundation

// Random-read throughput of the model file with F_NOCACHE pread (bypassing the page cache),
// for block sizes near one expert slice and several thread counts.
func runIOTest(file: String, seconds: Double = 4, log: (String) -> Void) {
    let path = documentsURL().appendingPathComponent(file).path
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { log("ERROR: open \(path)"); return }
    defer { close(fd) }
    _ = fcntl(fd, F_NOCACHE, 1)
    var st = stat()
    fstat(fd, &st)
    let size = Int64(st.st_size)
    log(String(format: "iotest: %@ (%.2f GB), F_NOCACHE pread, %.0f s per case", file, Double(size) / 1e9, seconds))

    for block in [256 << 10, 1 << 20, 4 << 20] {
        for threads in [1, 2, 4, 8, 16] {
            let lock = NSLock()
            var total = 0
            let deadline = Date().addingTimeInterval(seconds)
            let t0 = Date()
            DispatchQueue.concurrentPerform(iterations: threads) { ti in
                let buf = UnsafeMutableRawPointer.allocate(byteCount: block, alignment: 16384)
                defer { buf.deallocate() }
                var rng = SystemRandomNumberGenerator()
                var n = 0
                while Date() < deadline {
                    let off = Int64.random(in: 0..<(size - Int64(block)) / 16384, using: &rng) * 16384
                    let r = pread(fd, buf, block, off)
                    if r <= 0 { break }
                    n += r
                }
                lock.lock(); total += n; lock.unlock()
            }
            let dt = Date().timeIntervalSince(t0)
            log(String(format: "  block %5d KiB  threads %2d  %7.0f MB/s", block >> 10, threads, Double(total) / dt / 1e6))
        }
    }
    log("DONE")
}

// How much address space the app can map: anonymous PROT_NONE reservations, then the given files.
func runVATest(files: [String], log: (String) -> Void) {
    var maxGiB = 0
    for gib in stride(from: 4, through: 256, by: 4) {
        let p = mmap(nil, gib << 30, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0)
        if p == MAP_FAILED { break }
        munmap(p, gib << 30)
        maxGiB = gib
    }
    log("vatest: largest anonymous reservation \(maxGiB) GiB")
    var maps: [(UnsafeMutableRawPointer, Int)] = []
    var total = 0
    for f in files {
        let fd = open(documentsURL().appendingPathComponent(f).path, O_RDONLY)
        var st = stat(); fstat(fd, &st)
        let size = Int(st.st_size)
        let p = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
        close(fd)
        if p == MAP_FAILED {
            log(String(format: "vatest: mmap %@ (%.1f GiB) failed after %.1f GiB mapped: %s", f, Double(size) / 1073741824, Double(total) / 1073741824, strerror(errno)))
            continue
        }
        maps.append((p!, size)); total += size
        log(String(format: "vatest: mapped %@ (%.1f GiB), total %.1f GiB", f, Double(size) / 1073741824, Double(total) / 1073741824))
    }
    for (p, n) in maps { munmap(p, n) }
    log("DONE")
}
