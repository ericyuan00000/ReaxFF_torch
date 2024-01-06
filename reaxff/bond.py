import torch
from torch import nn


class ReaxffBondEnergy(nn.Module):
    '''
    ReaxFF bond energy calculation

    Parameters:
        rbondparam: Bond parameters
    '''
    def __init__(self, rbondparam: dict[tuple[int, int], list]):
        super(ReaxffBondEnergy, self).__init__()
        self.rbondparam = rbondparam

