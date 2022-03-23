import timeit
import torch
from sys import argv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sizes = tuple(64 * x for x in range(90, 100))

order = argv[1]

if order == 'mat_vec':

    tensors = tuple(
        (torch.arange(x*x, dtype=torch.float32).view(x, x).to(DEVICE), torch.arange(x, dtype=torch.float32).view(x, 1).to(DEVICE)) for x in sizes
    )
else:
    tensors = tuple(
        (torch.arange(x, dtype=torch.float32).view(1, x).to(DEVICE), torch.arange(x*x, dtype=torch.float32).view(x, x).to(DEVICE)) for x in sizes
    )


for i, (a, b) in enumerate(tensors):
    x = timeit.timeit(lambda: torch.matmul(a, b), number=50)
    print(x)


# for i, (a, b) in enumerate(tensors):
#     x = timeit.timeit(lambda: torch.matmul(b, a), number=50)
#     print(x)