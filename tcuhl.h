#pragma once

#include "stdio.h"

#define err(ans)                                                               \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

template <typename T>
__device__ T modp2(T a, T b) { return a & (b - 1); }

inline
__host__ __device__
void handle_oob(void *buf, int i, int len) {
  if (i % 100 == 0)
  printf("[tcuhl] oob %p + %d, len %d\n", buf, i, len);  
}

// base vector with checked access
template <typename T, typename S> struct bvec {
  T *buf;
  unsigned len, nbytes;

  __host__ __device__ 
  bvec(T *buf_, int len_) : buf(buf_), len(len_), nbytes(len_*sizeof(T)) { }

  __host__ __device__ 
  T& operator[](int i) {
#ifdef TCUHL_BOUNDSCHECK
    if (i < 0 || i >= len)
      handle_oob((void*) buf, i, len);
#endif
    return buf[i];
  }

  __host__ __device__
  S operator+(unsigned i) {
#ifdef TCUHL_BOUNDSCHECK
    if (i >= len)
      handle_oob((void*) buf, i, len);
#endif
    return S(buf + i, len - i);
  }

  __host__ __device__
  S slice(unsigned i, unsigned new_len) {
#ifdef TCUHL_BOUNDSCHECK
    if ((i + new_len) > len)
      handle_oob((void*) buf, i + new_len, len);
#endif
    return S(buf + i, new_len);
  }
};

// fwd declaration, see impl below
template <typename T> struct dvec;

// host-storage vector
template <typename T> struct hvec : bvec<T, hvec<T>> {

  __host__
  hvec(T *buf_, int len_) : bvec<T, hvec<T>>(buf_, len_) { }

#ifdef TCUHL_HVEC_PY_ARRAY_CTOR_PLZ
  template <int U=pybind11::array::c_style | pybind11::array::forcecast> __host__
  hvec(pybind11::array_t<T,U> np) : bvec<T, hvec<T>>(np.mutable_data(), np.size()) { }
#endif

  __host__
  hvec(int len_) : bvec<T, hvec<T>>(nullptr, len_) {
#ifdef TCUHL_DONT_PIN
    this->buf = (T*) malloc(this->nbytes);
#else
    cudaMallocHost(&this->buf, this->nbytes);
#endif
    memset(this->buf, 0, this->nbytes);
    // printf("hvec malloc %p %d B\n", this->buf, this->nbytes);
  }

  void zero() { memset((void*) this->buf, 0, this->nbytes); }

  void from(dvec<T> src) { err(cudaMemcpy(this->buf, src.buf, min(this->nbytes, src.nbytes), cudaMemcpyDeviceToHost)); }
  
  void from(hvec<T> src) { err(cudaMemcpy(this->buf, src.buf, min(this->nbytes, src.nbytes), cudaMemcpyHostToHost)); }
};

// device-storage vector
template <typename T> struct dvec : bvec<T, dvec<T>> {
  __host__ __device__
  dvec(T *buf, int len) : bvec<T, dvec<T>>(buf, len) { }

  __host__
  dvec(int len) : bvec<T, dvec<T>>(nullptr, len) {
    err(cudaMalloc(&this->buf, this->nbytes));
    err(cudaMemset(this->buf, 0, this->nbytes));
    // printf("dvec malloc %p %d B\n", this->buf, this->nbytes);
  }

  __host__
  dvec(hvec<T> &v) : bvec<T, dvec<T>>(nullptr, v.len) {
    err(cudaMalloc(&this->buf, this->nbytes));
    err(cudaMemcpy(this->buf, v.buf, this->nbytes, cudaMemcpyHostToDevice));
  }

  void zero() { err(cudaMemset((void*) this->buf, 0, this->nbytes)); }


  void from(dvec<T> src) {
    err(cudaMemcpy(this->buf, src.buf, min(this->nbytes, src.nbytes), cudaMemcpyDeviceToDevice));
  }
  
  void from(hvec<T> src) {
    err(cudaMemcpy(this->buf, src.buf, min(this->nbytes, src.nbytes), cudaMemcpyHostToDevice)); }
};

// pinned (host-device) vector
// TODO implement pinning
template <typename T>
struct pvec {
  unsigned len;
  hvec<T> h;
  dvec<T> d;

  pvec(int len_) : h(len_), d(len_), len(len_) { }
  
  pvec(hvec<T> h_) : h(h_), d(h.len), len(h.len) { to_device(); }

  pvec(T* buf, int len_) : h(buf,len_), d(len_), len(len_) { to_device(); }

  pvec(hvec<T> h_, dvec<T> d_) : h(h_), d(d_), len(h_.len) { }

  hvec<T> to_host()   { err(cudaMemcpy(h.buf, d.buf, d.nbytes, cudaMemcpyDeviceToHost)); return h; }

  dvec<T> to_device() { err(cudaMemcpy(d.buf, h.buf, h.nbytes, cudaMemcpyHostToDevice)); return d; }

  dvec<T> from_host(hvec<T> h_) { return pvec<T>(h_, d).to_device(); }

  void to_host(hvec<T> h_) { pvec<T>(h_, d).to_host(); }

  void zero() { h.zero(); d.zero(); }

};
