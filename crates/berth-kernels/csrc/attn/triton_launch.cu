// cuLaunchKernel for a Triton cubin whose dynamic shared memory exceeds the
// Ampere default of 48 KiB.
//
// berth-cuda Function::launch does not call
// cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES). The
// fork's TritonKernel does, when shared_bytes > 48 KiB. This shim loads the
// same image, opts the function in, and launches with the caller's kernelParams
// array (the same pointer-to-argument layout Function::launch uses).
//
// libcuda is resolved with dlsym. build.rs does not pass -lcuda.

#include "attn/error.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>

namespace {

struct Drv {
    CUresult (*init)(unsigned int);
    CUresult (*module_load_data)(CUmodule*, const void*);
    CUresult (*module_get_function)(CUfunction*, CUmodule, const char*);
    CUresult (*func_set_attribute)(CUfunction, CUfunction_attribute, int);
    CUresult (*launch_kernel)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int,
        CUstream,
        void**,
        void**);
    CUresult (*get_error_string)(CUresult, const char**);
};

Drv load_drv()
{
    Drv d{};
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(std::string("cuLaunch: dlopen libcuda.so.1: ") + dlerror());
    auto sym = [&] (const char* name) -> void* {
        dlerror();
        void* ptr = dlsym(lib, name);
        const char* err = dlerror();
        if (err) throw std::runtime_error(std::string("cuLaunch: dlsym ") + name + ": " + err);
        return ptr;
    };
    d.init = (decltype(d.init)) sym("cuInit");
    d.module_load_data = (decltype(d.module_load_data)) sym("cuModuleLoadData");
    d.module_get_function = (decltype(d.module_get_function)) sym("cuModuleGetFunction");
    d.func_set_attribute = (decltype(d.func_set_attribute)) sym("cuFuncSetAttribute");
    d.launch_kernel = (decltype(d.launch_kernel)) sym("cuLaunchKernel");
    d.get_error_string = (decltype(d.get_error_string)) sym("cuGetErrorString");
    return d;
}

Drv& drv()
{
    static Drv d = load_drv();
    return d;
}

void drv_ok(CUresult rc, const char* what)
{
    if (rc == CUDA_SUCCESS) return;
    const char* text = nullptr;
    if (drv().get_error_string) drv().get_error_string(rc, &text);
    std::string msg = std::string(what) + ": " + (text ? text : "CUDA driver error");
    throw std::runtime_error(msg);
}

struct Loaded {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

std::mutex g_mu;
std::map<std::string, Loaded> g_loaded;
bool g_init = false;

uint64_t fnv(const void* data, size_t n)
{
    const auto* bytes = (const unsigned char*) data;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    return h;
}

Loaded module_for(const void* image, size_t image_bytes, const char* name)
{
    std::string key = std::string(name) + ":" + std::to_string(image_bytes) + ":" + std::to_string(fnv(image, image_bytes));
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_loaded.find(key);
    if (it != g_loaded.end()) return it->second;

    if (!g_init) {
        drv_ok(drv().init(0), "cuInit");
        g_init = true;
    }
    // Bind the runtime to the primary context the same way TritonKernel does
    // before the driver API. berth-cuda has already retained that context.
    cudaFree(nullptr);

    CUmodule mod = nullptr;
    drv_ok(drv().module_load_data(&mod, image), "cuModuleLoadData");
    CUfunction fn = nullptr;
    drv_ok(drv().module_get_function(&fn, mod, name), "cuModuleGetFunction");
    Loaded loaded{mod, fn};
    g_loaded.emplace(key, loaded);
    return loaded;
}

}  // namespace

extern "C" int berth_attn_cubin_launch(
    const void* image,
    size_t image_bytes,
    const char* name,
    unsigned grid_x,
    unsigned grid_y,
    unsigned grid_z,
    unsigned block_x,
    unsigned block_y,
    unsigned block_z,
    unsigned shared_bytes,
    void** kernel_params,
    void* stream)
{
    try {
        berth_attn_clear_error();
        berth_attn_check(image && image_bytes > 0 && name && kernel_params, "cubin launch: null image, name, or params");
        berth_attn_check(grid_x > 0 && grid_y > 0 && grid_z > 0, "cubin launch: grid must be positive");
        berth_attn_check(block_x > 0 && block_y > 0 && block_z > 0, "cubin launch: block must be positive");

        Loaded loaded = module_for(image, image_bytes, name);
        if (shared_bytes > 48 * 1024) {
            drv_ok(
                drv().func_set_attribute(
                    loaded.function,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    (int) shared_bytes),
                "cuFuncSetAttribute");
        }
        drv_ok(
            drv().launch_kernel(
                loaded.function,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                shared_bytes,
                (CUstream) stream,
                kernel_params,
                nullptr),
            "cuLaunchKernel");
        return 0;
    } catch (const std::exception& err) {
        berth_attn_set_error(err.what());
        return -1;
    }
}
