#include "error.h"

namespace {

thread_local std::string g_err;

}

void berth_attn_clear_error()
{
    g_err.clear();
}

void berth_attn_set_error(const std::string& msg)
{
    g_err = msg;
}

extern "C" const char* berth_attn_last_error(void)
{
    return g_err.c_str();
}
