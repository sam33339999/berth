// Coop autotune without ATen. Algorithm matches quant/coop_autotune.cu
// (version 4, disk format coop_autotune_v1.bin). The L2 thrash buffer is a
// cudaMalloc instead of at::empty; the fork used the caching allocator only
// so a raw alloc would not fail while Torch held the pool.

#include "check.cuh"
#include "coop_autotune.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

#include "util.h"

namespace {

constexpr uint64_t kVersion = 4;
constexpr char kMagic[8] = {'E', 'X', '3', 'A', 'T', 'U', 'N', 'E'};
constexpr uint32_t kFormat = 1;

struct Expanded {
    void* kernel;
    int block_dim;
    int num_sms;
    int concurrency;
    int tag;
    float latency;
    std::vector<float> samples;
};

struct DiskHeader {
    char magic[8];
    uint32_t format;
    uint32_t record_size;
};

struct DiskRecord {
    uint64_t hash;
    int32_t tag;
    int32_t block_dim;
    int32_t num_sms;
    int32_t concurrency;
    uint32_t reserved0;
    uint32_t reserved1;
};

std::map<uint64_t, CoopAutotuneLaunch> g_launch_cache;
std::set<std::tuple<int, void*, size_t>> g_attr;
std::mutex g_disk_mu;
bool g_disk_loaded = false;
std::map<uint64_t, DiskRecord> g_disk;

uint64_t salt_hash(uint64_t hash)
{
    hash ^= kVersion;
    hash *= 1099511628211ull;
    return hash;
}

std::filesystem::path disk_cache_path()
{
    const char* override_path = std::getenv("EXLLAMAV3_TUNE_CACHE");
    if (override_path && override_path[0])
    {
        std::filesystem::path path(override_path);
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec))
            return path / "coop_autotune_v1.bin";
        return path;
    }
    const char* xdg = std::getenv("XDG_CACHE_HOME");
    std::filesystem::path base;
    if (xdg && xdg[0]) base = xdg;
    else
    {
        const char* home = std::getenv("HOME");
        if (!home || !home[0]) return {};
        base = std::filesystem::path(home) / ".cache";
    }
    return base / "exllamav3" / "autotune" / "coop_autotune_v1.bin";
}

bool read_header(std::ifstream& in)
{
    DiskHeader header{};
    in.read((char*) &header, sizeof(header));
    if (!in) return false;
    if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0) return false;
    if (header.format != kFormat) return false;
    if (header.record_size < sizeof(DiskRecord)) return false;
    return true;
}

void load_disk_locked()
{
    if (g_disk_loaded) return;
    g_disk_loaded = true;
    std::filesystem::path path = disk_cache_path();
    if (path.empty() || !std::filesystem::exists(path)) return;
    std::ifstream in(path, std::ios::binary);
    if (!in || !read_header(in)) return;

    DiskHeader header{};
    in.seekg(0, std::ios::beg);
    in.read((char*) &header, sizeof(header));
    if (!in) return;

    std::vector<char> bytes(header.record_size);
    while (in.read(bytes.data(), header.record_size))
    {
        if (header.record_size >= sizeof(DiskHeader) &&
            std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) == 0)
        {
            in.seekg(-(std::streamoff)(header.record_size - sizeof(DiskHeader)), std::ios::cur);
            continue;
        }
        DiskRecord record{};
        std::memcpy(&record, bytes.data(), sizeof(record));
        g_disk[record.hash] = record;
    }
}

void append_disk_locked(const DiskRecord& record)
{
    std::filesystem::path path = disk_cache_path();
    if (path.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return;
    bool write_header = !std::filesystem::exists(path) || std::filesystem::file_size(path, ec) == 0;
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) return;
    if (write_header)
    {
        DiskHeader header{};
        std::memcpy(header.magic, kMagic, sizeof(kMagic));
        header.format = kFormat;
        header.record_size = sizeof(DiskRecord);
        char buf[sizeof(header) + sizeof(record)];
        std::memcpy(buf, &header, sizeof(header));
        std::memcpy(buf + sizeof(header), &record, sizeof(record));
        out.write(buf, sizeof(buf));
    }
    else out.write((const char*) &record, sizeof(record));
}

bool launch_from_disk(
    uint64_t hash,
    const std::vector<CoopAutotuneCandidate>& candidates,
    CoopAutotuneLaunch* out)
{
    std::lock_guard<std::mutex> lock(g_disk_mu);
    load_disk_locked();
    auto lookup = g_disk.find(hash);
    if (lookup == g_disk.end()) return false;
    const DiskRecord& record = lookup->second;
    for (const CoopAutotuneCandidate& candidate : candidates)
    {
        if (candidate.tag != record.tag) continue;
        if (candidate.block_dim != record.block_dim) continue;
        if (record.num_sms < 1 || record.num_sms > candidate.max_num_sms) continue;
        int max_concurrency = MAX(candidate.max_concurrency, 1);
        int total_sms = candidate.total_sms > 0 ? candidate.total_sms : candidate.max_num_sms;
        int expected = MAX(MIN(total_sms / record.num_sms, max_concurrency), 1);
        if (record.concurrency != expected) continue;
        *out = {candidate.kernel, record.block_dim, record.num_sms, record.concurrency, record.tag};
        return true;
    }
    return false;
}

void store_disk(uint64_t hash, const CoopAutotuneLaunch& launch)
{
    DiskRecord record{hash, launch.tag, launch.block_dim, launch.num_sms, launch.concurrency, 0, 0};
    std::lock_guard<std::mutex> lock(g_disk_mu);
    load_disk_locked();
    g_disk[hash] = record;
    append_disk_locked(record);
}

void set_attr_once(void* kernel, size_t smem)
{
    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    auto key = std::make_tuple(device, kernel, smem);
    if (g_attr.find(key) != g_attr.end()) return;
    berth_cuda_ok(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem),
        "gemm max dynamic smem");
    g_attr.insert(key);
}

struct Thrash {
    void* ptr = nullptr;
    size_t size = 0;
    Thrash() = default;
    Thrash(const Thrash&) = delete;
    Thrash& operator=(const Thrash&) = delete;
    Thrash(Thrash&& other) noexcept : ptr(other.ptr), size(other.size)
    {
        other.ptr = nullptr;
        other.size = 0;
    }
    Thrash& operator=(Thrash&& other) noexcept
    {
        if (this != &other)
        {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }
    ~Thrash() { if (ptr) cudaFree(ptr); }
};

Thrash alloc_thrash()
{
    int device = 0;
    berth_cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
    int l2 = 0;
    berth_cuda_ok(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device), "L2 cache size");
    size_t base = (size_t) (l2 > 0 ? l2 : 64 * 1024 * 1024);
    for (size_t size : {base * 2, base})
    {
        Thrash buf;
        if (cudaMalloc(&buf.ptr, size) == cudaSuccess)
        {
            buf.size = size;
            return buf;
        }
        buf.ptr = nullptr;
    }
    return {};
}

void thrash_l2(const Thrash& buf, cudaStream_t stream)
{
    if (!buf.ptr) return;
    berth_cuda_ok(cudaMemsetAsync(buf.ptr, 0, buf.size, stream), "L2 thrash memset");
}

float trimmed_mean(std::vector<float>& samples)
{
    if (samples.empty()) throw std::runtime_error("CoopKernelAutotuner: no timing samples");
    std::sort(samples.begin(), samples.end());
    size_t trim = samples.size() / 4;
    size_t begin = trim;
    size_t end = samples.size() - trim;
    if (begin >= end)
    {
        begin = 0;
        end = samples.size();
    }
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += samples[i];
    return (float) (sum / (double) (end - begin));
}

void measure_sample(
    Expanded& candidate,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    int repeats,
    cudaEvent_t start,
    cudaEvent_t end,
    const Thrash& thrash)
{
    thrash_l2(thrash, stream);
    berth_cuda_ok(cudaEventRecord(start, stream), "tune event start");
    for (int i = 0; i < repeats; ++i)
    {
        berth_cuda_ok(
            cudaLaunchCooperativeKernel(
                candidate.kernel,
                dim3(candidate.num_sms, 1, candidate.concurrency),
                candidate.block_dim,
                kernel_args,
                smem,
                stream),
            "tune cooperative launch");
    }
    berth_cuda_ok(cudaEventRecord(end, stream), "tune event end");
    berth_cuda_ok(cudaEventSynchronize(end), "tune event sync");
    float ms = 0.0f;
    berth_cuda_ok(cudaEventElapsedTime(&ms, start, end), "tune elapsed");
    candidate.samples.push_back(ms / (float) repeats);
}

void keep_best(std::vector<Expanded>& candidates, size_t limit)
{
    for (Expanded& candidate : candidates) candidate.latency = trimmed_mean(candidate.samples);
    std::sort(candidates.begin(), candidates.end(), [](const Expanded& a, const Expanded& b) {
        return a.latency < b.latency;
    });
    if (candidates.size() > limit) candidates.resize(limit);
}

void measure_stage(
    std::vector<Expanded>& candidates,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    int rounds,
    int repeats,
    size_t keep,
    cudaEvent_t start,
    cudaEvent_t end,
    const Thrash& thrash)
{
    if (candidates.empty()) throw std::runtime_error("CoopKernelAutotuner: no candidates in stage");
    for (Expanded& candidate : candidates)
    {
        candidate.samples.clear();
        candidate.samples.reserve(rounds);
        set_attr_once(candidate.kernel, smem);
        berth_cuda_ok(
            cudaLaunchCooperativeKernel(
                candidate.kernel,
                dim3(candidate.num_sms, 1, candidate.concurrency),
                candidate.block_dim,
                kernel_args,
                smem,
                stream),
            "tune warmup launch");
    }
    berth_cuda_ok(cudaStreamSynchronize(stream), "tune warmup sync");

    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    for (int round = 0; round < rounds; ++round)
    {
        size_t n = order.size();
        size_t step = 2 * (size_t) round + 1;
        while (std::gcd(step, n) != 1) step += 2;
        size_t pos = ((uint64_t) round * 1103515245ull + 12345ull) % n;
        for (size_t i = 0; i < n; ++i)
        {
            measure_sample(candidates[order[pos]], kernel_args, smem, stream, repeats, start, end, thrash);
            pos = (pos + step) % n;
        }
    }
    keep_best(candidates, keep);
}

CoopAutotuneLaunch tune(
    const std::vector<CoopAutotuneCandidate>& base_candidates,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    size_t numel_b)
{
    std::vector<Expanded> candidates;
    for (const CoopAutotuneCandidate& base : base_candidates)
    {
        int total_tiles = (int) (numel_b / base.block_dim / 16);
        int min_sms = MAX(2, MIN(total_tiles / 32, base.max_num_sms) / base.max_concurrency / 2 * 2);
        if (!base.kernel) throw std::runtime_error("CoopKernelAutotuner: null kernel candidate");
        if (base.block_dim <= 0) throw std::runtime_error("CoopKernelAutotuner: invalid block_dim");
        if (base.max_num_sms <= 0) throw std::runtime_error("CoopKernelAutotuner: invalid max_num_sms");
        int max_concurrency = MAX(base.max_concurrency, 1);
        int total_sms = base.total_sms > 0 ? base.total_sms : base.max_num_sms;
        if (max_concurrency > 1 || base.max_num_sms == 1)
        {
            int concurrency = MAX(MIN(total_sms, max_concurrency), 1);
            candidates.push_back({base.kernel, base.block_dim, 1, concurrency, base.tag, 0.0f, {}});
        }
        for (int num_sms = min_sms; num_sms <= base.max_num_sms * 85 / 100; num_sms += 2)
        {
            int concurrency = MAX(MIN(total_sms / num_sms, max_concurrency), 1);
            candidates.push_back({base.kernel, base.block_dim, num_sms, concurrency, base.tag, 0.0f, {}});
        }
        if (base.max_num_sms > 1)
        {
            int concurrency = MAX(MIN(total_sms / base.max_num_sms, max_concurrency), 1);
            candidates.push_back({base.kernel, base.block_dim, base.max_num_sms, concurrency, base.tag, 0.0f, {}});
        }
    }
    if (candidates.empty()) throw std::runtime_error("CoopKernelAutotuner: no candidates");

    Thrash thrash = alloc_thrash();
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    berth_cuda_ok(cudaEventCreate(&start), "tune event create");
    berth_cuda_ok(cudaEventCreate(&end), "tune event create");

    int repeats = 8;
    if (numel_b > 1e6) repeats = 5;
    if (numel_b > 1e7) repeats = 4;
    if (numel_b > 1e8) repeats = 3;
    if (numel_b > 2e8) repeats = 2;
    int max_rounds = 32;
    if (numel_b > 1e6) max_rounds = 12;
    if (numel_b > 1e7) max_rounds = 8;
    if (numel_b > 1e8) max_rounds = 5;
    if (numel_b > 2e8) max_rounds = 2;
    int max_cands = 7;
    if (numel_b > 1e6) max_cands = 5;
    if (numel_b > 1e7) max_cands = 4;
    if (numel_b > 1e8) max_cands = 3;
    if (numel_b > 2e8) max_cands = 2;

    if (candidates.size() > 1 && max_cands > 4)
        measure_stage(candidates, kernel_args, smem, stream, MIN(2, max_rounds), repeats, MIN(6, max_cands), start, end, thrash);
    if (candidates.size() > 1 && max_cands > 1)
        measure_stage(candidates, kernel_args, smem, stream, MIN(8, max_rounds), repeats, MIN(3, max_cands), start, end, thrash);
    if (candidates.size() > 1)
        measure_stage(candidates, kernel_args, smem, stream, MIN(14, max_rounds), repeats, 1, start, end, thrash);

    berth_cuda_ok(cudaEventDestroy(start), "tune event destroy");
    berth_cuda_ok(cudaEventDestroy(end), "tune event destroy");
    const Expanded& best = candidates[0];
    return {best.kernel, best.block_dim, best.num_sms, best.concurrency, best.tag};
}

}  // namespace

extern "C" uint64_t berth_exl3_salt_hash(uint64_t hash)
{
    return salt_hash(hash);
}

bool CoopKernelAutotuner::launch_locked(
    uint64_t hash,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    CoopAutotuneLaunch* out)
{
    hash = salt_hash(hash);
    CoopAutotuneLaunch launch{};
    auto lookup = g_launch_cache.find(hash);
    if (lookup == g_launch_cache.end()) return false;
    launch = lookup->second;
    set_attr_once(launch.kernel, smem);
    berth_cuda_ok(
        cudaLaunchCooperativeKernel(
            launch.kernel,
            dim3(launch.num_sms, 1, launch.concurrency),
            launch.block_dim,
            kernel_args,
            smem,
            stream),
        "gemm cooperative launch");
    if (out) *out = launch;
    return true;
}

CoopAutotuneLaunch CoopKernelAutotuner::launch(
    uint64_t hash,
    const std::vector<CoopAutotuneCandidate>& candidates,
    void** kernel_args,
    size_t smem,
    cudaStream_t stream,
    size_t numel_b)
{
    CoopAutotuneLaunch launch{};
    uint64_t salted = salt_hash(hash);
    if (launch_locked(hash, kernel_args, smem, stream, &launch)) return launch;
    if (launch_from_disk(salted, candidates, &launch))
    {
        g_launch_cache[salted] = launch;
        set_attr_once(launch.kernel, smem);
        berth_cuda_ok(
            cudaLaunchCooperativeKernel(
                launch.kernel,
                dim3(launch.num_sms, 1, launch.concurrency),
                launch.block_dim,
                kernel_args,
                smem,
                stream),
            "gemm cooperative launch");
        return launch;
    }
    launch = tune(candidates, kernel_args, smem, stream, numel_b);
    auto inserted = g_launch_cache.emplace(salted, launch);
    launch = inserted.first->second;
    store_disk(salted, launch);
    set_attr_once(launch.kernel, smem);
    berth_cuda_ok(
        cudaLaunchCooperativeKernel(
            launch.kernel,
            dim3(launch.num_sms, 1, launch.concurrency),
            launch.block_dim,
            kernel_args,
            smem,
            stream),
        "gemm cooperative launch");
    return launch;
}
