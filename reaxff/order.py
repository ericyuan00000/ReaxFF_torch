import torch
import torch.nn as nn


class BondOrder(nn.Module):
    '''
    ReaxFF bond order calculation

    Parameters:
        rob1: sigma bond radius inverse, 1 / r_o^sigma (Angstrom^-1)
        rob2: pi bond radius inverse, 1 / r_o^pi (Angstrom^-1)
        rob3: double pi bond radius inverse, 1 / r_o^pipi (Angstrom^-1)
        bop1: sigma bond order parameter, p_bo1
        bop2: sigma bond order parameter, p_bo2
        pdp: pi bond order parameter, p_bo3
        ptp: pi bond order parameter, p_bo4
        pdo: double pi bond order parameter, p_bo5
        popi: double pi bond order parameter, p_bo6
        aval: valancy, Val_i
        vval3: number of lone pairs, Val'boc_i
        vpar1: overcoordination parameter, p_boc1
        vpar2: overcoordination parameter, p_boc2
        bo132: bond order correction, p_boc3
        bo131: bond order correction, p_boc4
        bo133: bond order correction, p_boc5
    '''
    def __init__(self, rob1, rob2, rob3, bop1, bop2, pdp, ptp, pdo, popi, aval, vval3, vpar1, vpar2, bo132, bo131, bo133):
        super(BondOrder, self).__init__()
        self.rob1 = rob1
        self.rob2 = rob2
        self.rob3 = rob3
        self.bop1 = bop1
        self.bop2 = bop2
        self.pdp = pdp
        self.ptp = ptp
        self.pdo = pdo
        self.popi = popi
        self.aval = aval
        self.vval3 = vval3
        self.vpar1 = vpar1
        self.vpar2 = vpar2
        self.vp132 = torch.sqrt(bo132.unsqueeze(-1) * bo132.unsqueeze(-2))    # p_boc3 = sqrt(p_boc3_i * p_boc3_j)
        self.vp131 = torch.sqrt(bo131.unsqueeze(-1) * bo131.unsqueeze(-2))    # p_boc4 = sqrt(p_boc4_i * p_boc4_j)
        self.vp133 = torch.sqrt(bo133.unsqueeze(-1) * bo133.unsqueeze(-2))    # p_boc5 = sqrt(p_boc5_i * p_boc5_j)


    def forward(self, rij):
        '''
        Inputs:
            rij: distance between two atoms
        '''
        # equation 2
        borsi = torch.exp(self.bop1 * (rij * self.rob1) ** self.bop2)
            # BO'_ij^sigma = exp(p_bo1 * (rij / r_o^sigma) ^ p_bo2)
        bopi = torch.exp(self.pdp * (rij * self.rob2) ** self.ptp)
            # BO'_ij^pi = exp(p_bo3 * (rij / r_o^pi) ^ p_bo4)
        bopi2 = torch.exp(self.pdo * (rij * self.rob3) ** self.popi)
            # BO'_ij^pipi = exp(p_bo5 * (rij / r_o^pipi) ^ p_bo6)
        bor = borsi + bopi + bopi2
            # BO'_ij = BO'_ij^sigma + BO'_ij^pi + BO'_ij^pipi
        # equation 3
        aboi = torch.sum(bor, dim=-1)
        ovi = -self.aval + aboi
            # Delta'_i = -Val_i + sum_j BO'_ij
        ovi2 = -self.vval3 + aboi
            # Delta'boc_i = -Val'boc_i + sum_j BO'_ij
        # equation 4
        exp1 = torch.exp(-self.vpar1 * ovi)
        ovcor1 = exp1.unsqueeze(-1) + exp1.unsqueeze(-2)
            # f2(Delta'_i, Delta'_j) = exp(-p_boc1 * Delta'_i) + exp(-p_boc1 * Delta'_j)
        exp2 = torch.exp(-self.vpar2 * ovi)
        ovcor2 = - torch.log((exp2.unsqueeze(-1) + exp2.unsqueeze(-2)) / 2) / self.vpar2
            # f3(Delta'_i, Delta'_j) = -1 / p_boc2 * ln(1 / 2 * (exp(-p_boc2 * Delta'_i) + exp(-p_boc2 * Delta'_j)
        coortot = ((self.aval.unsqueeze(-1) + ovcor1) / (self.aval.unsqueeze(-1) + ovcor1 + ovcor2) + (self.aval.unsqueeze(-2) + ovcor1) / (self.aval.unsqueeze(-2) + ovcor1 + ovcor2)) / 2
            # f1(Delta'_i, Delta'_j) = 1 / 2 * ((Val_i + f2(Delta'_i, Delta'_j)) / (Val_i + f2(Delta'_i, Delta'_j) + f3(Delta'_i, Delta'_j)) + (Val_j + f2(Delta'_i, Delta'_j)) / (Val_j + f2(Delta'_i, Delta'_j) + f3(Delta'_i, Delta'_j)))
        bocor1 = 1 / (1 + torch.exp(-self.vp132 * (self.vp131 * bor ** 2 - ovi2) + self.vp133))
            # f4(BO'_ij, Delta'boc_ij) = 1 / (1 + exp(-sqrt(p_boc3_i * p_boc3_j) * (sqrt(p_boc4_i * p_boc4_j) * BO'_ij^2 - Delta'boc_ij) + sqrt(p_boc5_i * p_boc5_j)))
