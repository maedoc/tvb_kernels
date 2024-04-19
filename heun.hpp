// generates fused, dense simd
// tbd how well it works for batching

template <int N>
struct Buffer { 
    int n = N; float * __restrict__ x;
    Buffer() {}
    Buffer(float *in_x) : x(in_x) {}
    float& operator[](int i) { return x[i]; }
};

template <typename Dfun, typename Buf> struct Heun {
    float dt;
    Dfun dfun;
    typedef Buf B;

    Buf f1, f2, xi;
    void operator()(Buf nx, Buf x) {
        dfun(f1, x);
        stage1(xi, x, f1);
        dfun(f2, xi);
        stage2(nx, x);
    }
    void stage1(Buf &xi, Buf &x, Buf &f1) {
        for (int i=0; i<xi.n; i++)
            xi[i] = x[i] + dt*f1[i];
    }
    void stage2(Buf &nx, Buf &x) {
        for (int i=0; i<nx.n; i++)
            nx[i] = x[i] + dt/2*(f1[i] + f2[i]);
    }
};

template <typename Buf>
struct Bistable {
    typedef Buf B;
    void operator()(Buf &f, Buf &x) {
        for (int i=0; i<f.n; i++)
            f[i] = x[i] - x[i]*x[i]*x[i];
    }
};

template <int N>
void step_bistable(float dt, float *nx, float *x) {
    typedef Buffer<N> B;
    Heun<Bistable<B>, B> step;
    step.dt = dt;
    step(nx, x);
}

