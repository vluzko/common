"""Is it faster to arrayize once, or in batches"""
import numpy as np
import timeit
import torch


def np_all(vals):
    stack = np.stack(vals)
    for batch in stack[::32]:
        pass
    return [x for x in stack]


def np_batch(vals):
    l = []
    for i in range(0, len(vals), 32):
        b = np.stack(vals[i: i + 32])
        l.extend(x for x in b)
    return l


def run(n, k):
    vals = [np.arange(k) for _ in range(n)]
    t1 = timeit.timeit(lambda : np_all(vals), number=100)
    t2 = timeit.timeit(lambda : np_batch(vals), number=100)
    print(f'List sweep @{n}:\n\t{t1}\n\t{t2}')

# Sweep across list lengths
for n in (100, 1000, 10000, 100000):
    run(n, 5)

# Sweep across array sizes
for k in (5, 50, 500, 5000):
    run(1000, k)