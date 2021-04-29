import unittest
import numpy as np
import mandelbrot_functions as mf


class TestMandelbrotMethods(unittest.TestCase):

    def test_vector(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_naive(c, T, I),
                mf.mandelbrot_vector([c, T, I])
            )
        )

    def test_numba(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_naive(c, T, I),
                mf.mandelbrot_numba(c, T, I)
            )
        )

    def test_gpu(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_naive(c, T, I),
                mf.mandelbrot_gpu(c, T, I)
            )
        )

    def test_cython_naive(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_naive(c, T, I),
                mf.mandelbrot_naive_cython(c, T, I)
            )
        )

    def test_cython_vector(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_naive(c, T, I),
                mf.mandelbrot_vector_cython([c, T, I])
            )
        )

    def test_parallel(self):
        c = mf.create_mesh(100, 100)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_naive(c, T, I),
                mf.mandelbrot_parallel_vector(c, T, I, 12, 20, 5)
            )
        )

    def test_distributed(self):
        c = mf.create_mesh(4096, 4096)
        T = 2
        I = 100
        self.assertTrue(
            np.allclose(
                mf.mandelbrot_gpu(c, T, I),
                mf.mandelbrot_distributed(c, T, I, 12, 512, 8)
            )
        )


if __name__ == '__main__':
    unittest.main()
