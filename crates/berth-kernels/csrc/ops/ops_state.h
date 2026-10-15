#pragma once

#include <string>

// Thread-local launch status for every ops translation unit. The strings live
// in ops_state.cu so a norm launch and an mgemm launch share one buffer.

void berth_ops_clear();
void berth_ops_set_error(const std::string& msg);
void berth_ops_set_launch(const std::string& msg);
