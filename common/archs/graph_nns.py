import torch

from torch_geometric.data import Data


edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)


class EdgeConv(MessagePassing):
    """A message passing network.

    Attributes:
        nodes:      The number of nodes in the graph.
        node_dim:   The dimension of a single node.
    """

    nodes: int
    node_dim: int


    def __init__(self, nodes: int, node_dim: int):
        super().__init__(aggr='max')
        self.nodes = nodes
        self.node_dim = node_dim
        self.size = num_objects
        self.mlp = nn.Sequential(
            nn.Linear(self.size, self.obj_dim),
            ReLU(),
            nn.Linear(self.size, self.obj_dim)
        )

    def forward(self, x, edge_index):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)  # shape [N, F_out]

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        import pdb
        pdb.set_trace()
        return self.mlp(edge_features)  # shape [E, F_out]