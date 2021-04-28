#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

__kernel void mandelbrot(
    __global const cdouble_t *c_gpu,
    __global double *result_g,
    const int MAX_ITER,
    const int THRESHOLD
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int width = get_global_size(0);

    cdouble_t z = cdouble_new(0, 0);
    cdouble_t c = c_gpu[gidx * width + gidy];
    double n = 0;

    while (cdouble_abs(z) <= THRESHOLD && n < MAX_ITER){
        z = cdouble_add(cdouble_mul(z, z), c);
        n = n + 1 ;
    }
    result_g[gidx * width + gidy] = n/MAX_ITER;
}
