#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "delays.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(tvb_kernels, m) {
    m.def("delays2", [](
      int t,
      nb::ndarray<float, nb::device::cpu, nb::c_contig> out1,
      nb::ndarray<float, nb::device::cpu, nb::c_contig> out2,
      nb::ndarray<float, nb::device::cpu, nb::c_contig> buf,
      nb::ndarray<float, nb::device::cpu, nb::c_contig> weights,
      nb::ndarray<int, nb::device::cpu, nb::c_contig> idelays,
      nb::ndarray<int, nb::device::cpu, nb::c_contig> indices,
      nb::ndarray<int, nb::device::cpu, nb::c_contig> indptr
    ) {
      size_t nv = buf.shape(0), nh = buf.shape(1);
      delays2(nv, nh, t, out1.data(), out2.data(), buf.data(), weights.data(),
              idelays.data(), indices.data(), indptr.data());
    });
}
