import torch
from torch import nn
from torch import Tensor
import networkx as nx
from itertools import combinations

parents = {
            'Ab': 'Hip',
            'Chest': 'Ab',
            'Head': 'Neck',
            'Hip': 'Hip',
            'LFArm': 'LUArm',
            'LFoot': 'LShin',
            'LHand': 'LFArm',
            'LShin': 'LThigh',
            'LShoulder': 'Chest',
            'LThigh': 'Hip',
            'LToe': 'LFoot',
            'LUArm': 'LShoulder',
            'Neck': 'Chest',
            'RFArm': 'RUArm',
            'RFoot': 'RShin',
            'RHand': 'RFArm',
            'RShin': 'RThigh',
            'RShoulder': 'Chest',
            'RThigh': 'Hip',
            'RToe': 'RFoot',
            'RUArm': 'RShoulder'
        }

parents_index = [list(parents.keys()).index(parents[x]) for x in parents.keys()]

class CompositionalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
    
        graph_index = nx.Graph()
        graph_index.add_edges_from([(x, parents_index[x]) for x in range(len(parents_index))])
        self.all_shortest_paths_index = dict(nx.all_pairs_shortest_path(graph_index))

        self.P = list(combinations(range(len(parents)), r=2))
        self.avg_reduction = reduction == 'mean'

    def sgn(self, path, m: int):
        return 1 if parents_index[path[m]] == path[m + 1] else -1

    @staticmethod
    def get_ith_bone_idx(i: int):
        return [start_index := i * 3, start_index + 1 , start_index + 2]

    def get_long_range_relative_position(self, input: Tensor, joint_u: int, joint_v: int) -> Tensor:
        path = self.all_shortest_paths_index[joint_u][joint_v]
        idx = [self.get_ith_bone_idx(i) for i in path[:-1]]
        sign = torch.tensor([[self.sgn(path, m)] * 3 for m in range(len(path) - 1)], requires_grad=False, device=input.device, dtype=input.dtype)
        return (input[:, idx] * sign).sum(dim=1)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        delta_batch = []

        # Todo: consider pre-computing the index and sign for each pair of joints as the pairs are already determined on init
        for u, v in self.P:
            delta_j = self.get_long_range_relative_position(input, int(u), int(v))
            delta_j_gt = target[:, self.get_ith_bone_idx(u)] - target[:, self.get_ith_bone_idx(v)]
            delta_batch.append(delta_j - delta_j_gt)

        # stacked shape: 210, 2000, 3
        stacked = torch.stack(delta_batch, dim=0)
        # norm shape: 210, 2000
        delta_norms = torch.linalg.norm(stacked, ord=1, dim=2)

        if self.avg_reduction:
            # mean shape: 210; sum shape: 1
            return delta_norms.mean(dim=1).sum()
        return delta_norms.sum()
