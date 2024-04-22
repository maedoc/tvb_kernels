import numpy as np
import scipy.sparse
import pytest

import tvb_kernels.tvb_kernels as tvb



def np_delays2(buf,nh,t,idelays,indices,weights,indptr,c):
    xij = buf[indices, (nh + t + np.c_[0,1].T - idelays) & (nh-1)] # (2, nnz, 8)
    np.add.reduceat(xij*weights.reshape(-1,1), indptr[:-1], axis=1, out=c)
    c[:, np.argwhere(np.diff(indptr)==0)] = 0


def test_delays2():
    nv = 256
    glwk = np.random.randn(4, nv, nv).astype('f')
    m = glwk[0] > 0
    glwk[:,m] = 0
    g, l, w, k = [scipy.sparse.csr_array(_) for _ in glwk]
    assert g.nnz < (nv**2//4*3)

    dt = 0.1
    local_velocity = 1.0
    v2v_velocity = 10.0
    il = (l.data / v2v_velocity / dt).astype('i')
    ig = (g.data / local_velocity / dt).astype('i')
    nh = 2**int(np.ceil(np.log2(il.max() + 1)))
    bs = 1
    buf = np.zeros((nv, nh, bs), 'f')
    buf[:] = np.random.randn(*buf.shape)
    c = np.zeros((2, nv, bs), 'f')

    x = np.random.randn(*buf.shape).astype('f')
    x = x*(x > 0.5)

    c_c = np.zeros_like(c)
    c_np = np.zeros_like(c)

    tvb.delays2f(15, c_c[0], c_c[1], buf, w.data, il, w.indices, w.indptr)
    np_delays2(buf, nh, 15, il, w.indices, w.data, w.indptr, c_np)

    np.testing.assert_allclose(c_c, c_np, 1e-4, 1e-4)
