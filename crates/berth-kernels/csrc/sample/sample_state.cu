#include "sample/sample_state.h"

namespace {

thread_local std::string g_err;
thread_local std::string g_launch;

}  // namespace

void berth_sample_clear()
{
    g_err.clear();
    g_launch.clear();
}

void berth_sample_set_error(const std::string& msg)
{
    g_err = msg;
    g_launch.clear();
}

void berth_sample_set_launch(const std::string& msg)
{
    g_launch = msg;
    g_err.clear();
}

extern "C" const char* berth_sample_last_error(void)
{
    return g_err.c_str();
}

extern "C" const char* berth_sample_last_launch(void)
{
    return g_launch.c_str();
}
