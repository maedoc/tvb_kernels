#include "heun.hpp"

extern "C" {
  void step_bistable_8(float dt, float *nx, float *x) { step_bistable<8>(dt, nx, x); }
  void step_bistable_32(float dt, float *nx, float *x) { step_bistable<32>(dt, nx, x); }
  void step_bistable_128(float dt, float *nx, float *x) { step_bistable<128>(dt, nx, x); }
  void step_bistable_256(float dt, float *nx, float *x) { step_bistable<256>(dt, nx, x); }
  void step_bistable_512(float dt, float *nx, float *x) { step_bistable<512>(dt, nx, x); }
}
