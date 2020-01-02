import nnetsauce.simulation.nodesimulation as nsim
import unittest as ut


# Basic tests


class TestSimul(ut.TestCase):
    def test_sobol(self):
        res = nsim.generate_sobol2(n_dims=3, n_points=4)
        self.assertAlmostEqual(res[2, 2], 0.25)

    def test_hammersley(self):
        res = nsim.generate_hammersley(n_dims=4, n_points=5)
        self.assertAlmostEqual(res[2, 2], 0.1111111111111111)

    def test_halton(self):
        res = nsim.generate_halton(n_dims=5, n_points=6)
        self.assertAlmostEqual(res[2, 2], 0.59999999999999998)

    def test_uniform(self):
        res = nsim.generate_uniform(n_dims=3, n_points=10, seed=123)
        self.assertAlmostEqual(res[2, 2], 0.72445532486063524)


if __name__ == "__main__":
    ut.main()
