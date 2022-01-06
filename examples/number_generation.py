import nnetsauce as ns
from time import time 
from tqdm import tqdm


print("\n")
print(f"1 - Sobol ----------")
print("\n")


n_dims = 5
n_points = 10

start = time()
print(ns.Simulator(n_points=n_points, n_dims=n_dims).draw())
print(f"timing ns.Simulator {time() - start}")

# start = time()
# print(ns.simulation.generate_sobol_randtoolbox(n_dims=n_dims, n_points=n_points))
# print(f"timing generate_sobol_randtoolbox {time() - start}")


print("\n")
n_dims = 100
n_points = 10000
n_repeats = 100


start = time()
[ns.Simulator(n_points=n_points, n_dims=n_dims).draw() for _ in tqdm(range(n_repeats))]
print(f"timing ns.Simulator (large matrix) {time() - start}")


# start = time()
# [ns.simulation.generate_sobol_randtoolbox(n_dims=n_dims, n_points=n_points) for _ in tqdm(range(n_repeats))]
# print(f"timing generate_sobol_randtoolbox (large matrix) {time() - start}")

start = time()
[ns.simulation.generate_sobol2(n_dims=n_dims, n_points=n_points) for _ in tqdm(range(n_repeats))]
print(f"timing generate_sobol2 (large matrix) {time() - start}")


print("\n")
print(f"2 - Halton ----------")
print("\n")

n_dims = 4
n_points = 5

start = time()
print(ns.Simulator(n_points=n_points, n_dims=n_dims, type_sim="halton").draw())
print(f"timing ns.Simulator {time() - start}")
print("\n")
# start = time()
# print(ns.simulation.generate_halton_randtoolbox(n_dims=n_dims, n_points=n_points))
# print(f"timing generate_halton_randtoolbox {time() - start}")
# print("\n")
start = time()
print(ns.simulation.generate_halton(n_dims=n_dims, n_points=n_points))
print(f"timing generate_halton{time() - start}")
print("\n")

print("\n")
n_dims = 100
n_points = 10000
n_repeats = 100


start = time()
[ns.Simulator(n_points=n_points, n_dims=n_dims, type_sim="halton").draw() for _ in tqdm(range(n_repeats))]
print(f"timing ns.Simulator (large matrix) {time() - start}")


start = time()
[ns.simulation.generate_halton(n_dims=n_dims, n_points=n_points) for _ in tqdm(range(n_repeats))]
print(f"timing generate_halton (large matrix) {time() - start}")


# start = time()
# [ns.simulation.generate_halton_randtoolbox(n_dims=n_dims, n_points=n_points) for _ in tqdm(range(n_repeats))]
# print(f"timing generate_halton_randtoolbox(large matrix) {time() - start}")


print("\n")
print(f"3 - Hammersley ----------")
print("\n")

n_dims = 5
n_points = 6

start = time()
print(ns.Simulator(n_points=n_points, n_dims=n_dims, type_sim="hammersley").draw())
print(f"timing ns.Simulator {time() - start}")
print("\n")
start = time()
print(ns.simulation.generate_hammersley(n_dims=n_dims, n_points=n_points))
print(f"timing generate_hammersley {time() - start}")
print("\n")


print("\n")
n_dims = 100
n_points = 10000
n_repeats = 100


start = time()
[ns.Simulator(n_points=n_points, n_dims=n_dims, type_sim="hammersley").draw() for _ in tqdm(range(n_repeats))]
print(time() - start)
print(f"timing ns.Simulator (large matrix) {time() - start}")

start = time()
[ns.simulation.generate_hammersley(n_dims=n_dims, n_points=n_points) for _ in tqdm(range(n_repeats))]
print(time() - start)
print(f"timing ns.simulation.generate_hammersley (large matrix) {time() - start}")



