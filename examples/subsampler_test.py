
import os 
import matplotlib.pyplot as plt
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_digits, load_diabetes
from time import time


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# dataset no. 1 ---------- 

print("\n\n dataset no. 1 (classification) ---------- \n\n")

dataset = load_digits()
t = dataset.target

print(" \n sequential ----- \n")

sub1 = ns.SubSampler(y=t, row_sample=0.8, seed=123, n_jobs=None)
start = time()
x1 = sub1.subsample()
print(f"elapsed time: {time() - start}")

print(x1)

_ = plt.hist(x1, bins='auto')
plt.show()

print(" \n parallel ----- \n")

sub2 = ns.SubSampler(y=t, row_sample=0.8, seed=123, n_jobs=2)
start = time()
x2 = sub2.subsample()
print(f"elapsed time: {time() - start}")

print(x2)

print(f" \n check: {np.allclose(x1, x2)} \n")

_ = plt.hist(x2, bins='auto')
plt.show()

# dataset no. 2 ---------- 

print("\n\n dataset no. 2 (regression) ---------- \n\n")

dataset = load_diabetes()
t = dataset.target

print(" \n sequential ----- \n")

sub1 = ns.SubSampler(y=t, row_sample=0.8, seed=123, n_jobs=None)
start = time()
y1 = sub1.subsample()
print(f"elapsed time: {time() - start}")

print(y1)

_ = plt.hist(y1, bins='auto')
plt.show()

print(" \n parallel ----- \n")

sub2 = ns.SubSampler(y=t, row_sample=0.8, seed=123, n_jobs=2)
start = time()
y2 = sub2.subsample()
print(f"elapsed time: {time() - start}")

print(y2)

_ = plt.hist(y2, bins='auto')
plt.show()

print(f" \n check: {np.allclose(y1, y2)} \n")




