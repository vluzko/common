import timeit
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sizes = (4 * x for x in range(1, 100))

tensors = (
    (torch.arange(x*x, dtype=torch.float32).view(x, x).to(DEVICE), torch.arange(x*x, dtype=torch.float32).view(x, x).to(DEVICE)) for x in sizes
)


for i, (a, b) in enumerate(tensors):
    x = timeit.timeit(lambda: torch.matmul(a, b), number=50)
    print(x)