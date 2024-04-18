#include <nanobind/nanobind.h>

int add(int a, int b) { return a + b; }

NB_MODULE(tvb_kernels, m) {
    m.def("add", &add);
}
