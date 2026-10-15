#pragma once

#include <stdexcept>
#include <string>

// One error string for every attention translation unit. A thread_local in a
// header would be a different object in each TU, and last_error() would miss
// the message the launcher just set.

void berth_attn_clear_error();
void berth_attn_set_error(const std::string& msg);

inline int ceil_div_int(int n, int d)
{
    return (n + d - 1) / d;
}

inline void berth_attn_check(bool ok, const char* msg)
{
    if (!ok) throw std::invalid_argument(msg);
}
