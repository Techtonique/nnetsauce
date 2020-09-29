import nnetsauce as ns
from time import time 
from tqdm import tqdm

fit_obj = ns.HypTan(0)

print("tanh(0)")
print(fit_obj.calculate())
print("\n")

fit_obj2 = ns.HypTan(5)

print("tanh(5)")
print(fit_obj2.calculate())
print("\n")

fit_obj3 = ns.HypTan(10)

print("tanh(10)")
print(fit_obj3.calculate())
print("\n")

fit_obj4 = ns.HypTan(0.5)

print("tanh(0.5)")
print(fit_obj4.calculate())
print("\n")

fit_obj5 = ns.HypTan(-0.5)

print("tanh(-0.5)")
print(fit_obj5.calculate())
print("\n")

n_dims = 6
n_points = 8

start = time()
print(ns.Simulator(n_points=n_points, n_dims=n_dims, skip=1).draw())
#ns.Simulator(n_points=n_points, n_dims=n_dims, skip=1).draw()
print(f"timing ns.Simulator {time() - start}")

start = time()
print(ns.simulation.generate_sobol_randtoolbox(n_dims=n_dims, n_points=n_points))
#ns.simulation.generate_sobol_randtoolbox(n_dims=n_dims, n_points=n_points)
print(f"timing generate_sobol_randtoolbox {time() - start}")


n_dims = 100
n_points = 100
n_repeats = 100


start = time()
[ns.Simulator(n_points=n_points, n_dims=n_dims, skip=1).draw() for _ in tqdm(range(n_repeats))]
print(time() - start)


start = time()
[ns.simulation.generate_sobol_randtoolbox(n_dims=n_dims, n_points=n_points) for _ in tqdm(range(n_repeats))]
print(time() - start)