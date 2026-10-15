#pragma once

#include <string>

// Thread-local launch status for the sampler translation units. Separate from
// the ops buffer so a norm launch and a penalty launch do not share a string.

void berth_sample_clear();
void berth_sample_set_error(const std::string& msg);
void berth_sample_set_launch(const std::string& msg);
