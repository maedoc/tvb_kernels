// https://gitlab.ebrains.eu/woodman/popcorn/-/blob/delays2-batch/delays2.c

template <typename real>
void delays2(int nv, int nh, int t,
             real *out1, real *out2,
             real *buf, real *weights, int *idelays, int *indices, int *indptr)
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
        // compute coupling terms for both Heun stages
        real acc1 = 0.0f, acc2 = 0.0f;

#pragma omp simd reduction(+:acc1,acc2)
        for (int j=indptr[i]; j<indptr[i+1]; j++) {
            real *b = buf + indices[j]*nh;
            real w = weights[j];
            int roll_t = nh + t - idelays[j];
            acc1 += w * b[(roll_t+0) & nhm];
            acc2 += w * b[(roll_t+1) & nhm];
        }
        out1[i] = acc1;
        out2[i] = acc2;
    }
}

// batch variant
void delays2_batch(int bs, int nv, int nh, int t,
             float *out1, float *out2,
             float *buf, float *weights, int *idelays, int *indices, int *indptr,
             float *x
             )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
    // update buffer
            /*
#pragma omp simd
        for (int l=0; l<bs; l++)
            buf[(i*nh + ((nh + t) & nhm))*bs + l] = x[i*bs + l];
            */

#pragma omp simd
        for (int l=0; l<bs; l++)
            out1[bs*i+l] = out2[bs*i+l] = 0.0f;
        
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            float *b = buf + indices[j]*nh*bs;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            float *b1 = b + ((roll_t+0) & nhm)*bs;
            float *b2 = b + ((roll_t+1) & nhm)*bs;

            
#pragma omp simd
            for (int l=0; l<bs; l++)
            {
                out1[bs*i + l] += w * b1[l];
                out2[bs*i + l] += w * b2[l];
            }
        }
    }
}
