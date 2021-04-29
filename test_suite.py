import unittest
import numpy as np
import mandelbrot_functions as mf


class TestMandelbrotMethods(unittest.TestCase):

    def test_vector(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(np.allclose(mf.mandelbrot_naive(c, T, I), mf.mandelbrot_vector([c, T, I])))

    def test_numba(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(np.allclose(mf.mandelbrot_naive(c, T, I), mf.mandelbrot_numba(c, T, I)))

    def test_gpu(self):
        c = mf.create_mesh(50, 50)
        T = 2
        I = 100
        self.assertTrue(np.allclose(mf.mandelbrot_naive(c, T, I), mf.mandelbrot_gpu(c, T, I)))


if __name__ == '__main__':
    unittest.main()
