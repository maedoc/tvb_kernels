#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "tvb_connectivity.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(tvb_kernels, m) {

  typedef nb::ndarray<float, nb::device::cpu, nb::c_contig> fvec;
  typedef nb::ndarray<double, nb::device::cpu, nb::c_contig> dvec;
  typedef nb::ndarray<int, nb::device::cpu, nb::c_contig> ivec;

  m.def("delays2f", [](
    int t,
    fvec out1, fvec out2, fvec buf, fvec weights, 
    ivec idelays, ivec indices, ivec indptr
  ) {
    size_t nv = buf.shape(0), nh = buf.shape(1);
    delays2<>(nv, nh, t, out1.data(), out2.data(), buf.data(), weights.data(),
            idelays.data(), indices.data(), indptr.data());
  });
}
