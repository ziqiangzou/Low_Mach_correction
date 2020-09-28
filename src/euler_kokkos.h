#pragma once

#include <string>

namespace euler_kokkos
{

void initialize(int& argc, char**& argv);
void finalize();
void print_kokkos_configuration();
void abort(const std::string& msg);

}
