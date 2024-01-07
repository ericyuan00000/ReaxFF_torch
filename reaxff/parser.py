import ase
import torch


class ReaxffParameters():
    '''
    Parameters:
        filename: path to the ReaxFF parameter file
    '''
    def __init__(self, filename: str):
        with open(filename) as f:
            lines = f.readlines()

        # description
        lines.pop(0)

        # genral parameters
        param_info = lines.pop(0).split()
        self.vpar = torch.full((41+1,), torch.nan)
        for i in range(int(param_info[0])):
            line = lines.pop(0).split()
            self.vpar[i+1] = float(line[0])

        # atom parameters
        param_info = lines.pop(0).split()
        lines.pop(0)
        lines.pop(0)
        lines.pop(0)
        self.atomtype = torch.full((int(param_info[0])+1,), torch.nan)
        self.ratomparam = torch.full((int(param_info[0])+1, 32+1), torch.nan)
        for block in range(int(param_info[0])):
            line = lines.pop(0).split()
            self.atomtype[block+1] = ase.data.atomic_numbers[line.pop(0)]
            self.ratomparam[block+1, 1:9] = torch.tensor([float(x) for x in line])
            line = lines.pop(0).split()
            self.ratomparam[block+1, 9:17] = torch.tensor([float(x) for x in line])
            line = lines.pop(0).split()
            self.ratomparam[block+1, 17:25] = torch.tensor([float(x) for x in line])
            line = lines.pop(0).split()
            self.ratomparam[block+1, 25:33] = torch.tensor([float(x) for x in line])

        # bond parameters
        param_info = lines.pop(0).split()
        lines.pop(0)
        self.rbondparam = torch.full((len(self.atomtype), len(self.atomtype), 16+1), torch.nan)
        for block in range(int(param_info[0])):
            line = lines.pop(0).split()
            k1, k2 = int(line.pop(0)), int(line.pop(0))
            self.rbondparam[k1, k2, 1:9] = torch.tensor([float(x) for x in line])
            self.rbondparam[k2, k1, 1:9] = torch.tensor([float(x) for x in line])
            line = lines.pop(0).split()
            self.rbondparam[k1, k2, 9:17] = torch.tensor([float(x) for x in line])
            self.rbondparam[k2, k1, 9:17] = torch.tensor([float(x) for x in line])

        # off-diagonal parameters
        param_info = lines.pop(0).split()
        self.roffdiagparam = torch.full((len(self.atomtype), len(self.atomtype), 6+1), torch.nan)
        for block in range(int(param_info[0])):
            line = lines.pop(0).split()
            k1, k2 = int(line.pop(0)), int(line.pop(0))
            self.roffdiagparam[k1, k2, 1:7] = torch.tensor([float(x) for x in line])
            self.roffdiagparam[k2, k1, 1:7] = torch.tensor([float(x) for x in line])

        # angle parameters
        param_info = lines.pop(0).split()
        self.rvalangparam = torch.full((len(self.atomtype), len(self.atomtype), len(self.atomtype), 7+1), torch.nan)
        for block in range(int(param_info[0])):
            line = lines.pop(0).split()
            k1, k2, k3 = int(line.pop(0)), int(line.pop(0)), int(line.pop(0))
            self.rvalangparam[k1, k2, k3, 1:8] = torch.tensor([float(x) for x in line])
            self.rvalangparam[k3, k2, k1, 1:8] = torch.tensor([float(x) for x in line])

        # torsion parameters
        param_info = lines.pop(0).split()
        self.rtorsparam = torch.full((len(self.atomtype), len(self.atomtype), len(self.atomtype), len(self.atomtype), 7+1), torch.nan)
        for block in range(int(param_info[0])):
            line = lines.pop(0).split()
            k1, k2, k3, k4 = int(line.pop(0)), int(line.pop(0)), int(line.pop(0)), int(line.pop(0))
            self.rtorsparam[k1, k2, k3, k4, 1:8] = torch.tensor([float(x) for x in line])
            self.rtorsparam[k4, k3, k2, k1, 1:8] = torch.tensor([float(x) for x in line])

        # hydrogen bond parameters
        param_info = lines.pop(0).split()
        self.rhbondparam = torch.full((len(self.atomtype), len(self.atomtype), len(self.atomtype), 4+1), torch.nan)
        for block in range(int(param_info[0])):
            line = lines.pop(0).split()
            k1, k2, k3 = int(line.pop(0)), int(line.pop(0)), int(line.pop(0))
            self.rhbondparam[k1, k2, k3, 1:5] = torch.tensor([float(x) for x in line])
            self.rhbondparam[k3, k2, k1, 1:5] = torch.tensor([float(x) for x in line])