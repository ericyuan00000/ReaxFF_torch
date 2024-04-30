import torch
import torch.nn as nn

import ase


class ReaxffBondOrder(nn.Module):
    '''
    ReaxFF bond order calculation

    Parameters:
        vpar: Gerneral parameters
        ratomparam: Atom parameters
        rbondparam: Bond parameters
        roffdiagparam: Off-diagonal parameters
    '''
    def __init__(self, params):
        super(ReaxffBondOrder, self).__init__()
        self.atomtype = params.atomtype
        self.vpar = params.vpar
        self.ratomparam = params.ratomparam
        self.rbondparam = params.rbondparam
        self.roffdiagparam = params.roffdiagparam

    
    def process_params(self, atomic_numbers):
        '''
        Parameters:
            atomic_numbers: atomic numbers of atoms in the system, shape (n_data, n_atoms)

        Returns:
            rob1: sigma bond radius inverse, 1 / r_o^sigma
            rob2: pi bond radius inverse, 1 / r_o^pi
            rob3: double pi bond radius inverse, 1 / r_o^pipi
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

        Note:
            The parameters are calculated according to the atomic numbers of the atoms in the system.
            The parameters are only calculated once (1) at the beginning of the simulation or (2) during the parsing of training data.
        '''
        n_data, n_atoms = atomic_numbers.shape
        atomic_ids = torch.zeros_like(atomic_numbers, dtype=torch.long)

        rob1 = torch.full((n_atoms, n_atoms), torch.nan)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if self.rbondparam.get((atomic_ids[i], atomic_ids[j]), 0)[4] > 0:
                    rob1[i, j] = 1 / self.rbondparam[(atomic_ids[i], atomic_ids[j])][4]
                else:
                    rob1[i, j] = 2 / (self.ratomparam[atomic_ids[i]][1] + self.ratomparam[atomic_ids[j]][1])
                if rob1[i, j] <= 1.0e-15:
                    rob1[i, j] = 0.5 * (self.ratomparam[atomic_ids[i]][1] + self.ratomparam[atomic_ids[j]][1])

        rob2 = torch.full((n_atoms, n_atoms), torch.nan)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if self.rbondparam.get((atomic_ids[i], atomic_ids[j]), 0)[5] > 0:
                    rob2[i, j] = 1 / self.rbondparam[(atomic_ids[i], atomic_ids[j])][5]    # TODO: check if this is correct regarding the inverse
                else:
                    rob2[i, j] = 2 / (self.ratomparam[atomic_ids[i]][7] + self.ratomparam[atomic_ids[j]][7])
                if rob2[i, j] <= 1.0e-15:
                    rob2[i, j] = 0.5 * (self.ratomparam[atomic_ids[i]][7] + self.ratomparam[atomic_ids[j]][7])

        rob3 = torch.full((n_atoms, n_atoms), torch.nan)
        for i in range(n_atoms):
            for j in range(n_atoms):
                if self.rbondparam.get((atomic_ids[i], atomic_ids[j]), 0)[6] > 0:
                    rob3[i, j] = 1 / self.rbondparam[(atomic_ids[i], atomic_ids[j])][6]    # TODO: check if this is correct regarding the inverse
                else:
                    rob3[i, j] = 2 / (self.ratomparam[atomic_ids[i]][17] + self.ratomparam[atomic_ids[j]][17])
                if rob3[i, j] <= 1.0e-15:
                    rob3[i, j] = 0.5 * (self.ratomparam[atomic_ids[i]][17] + self.ratomparam[atomic_ids[j]][17])
        
        bop1 = torch.full((n_atoms, n_atoms), torch.nan)
        bop1 = torch.tensor([self.rbondparam[(i, j)][13] for (i, j) in zip(atomic_ids, atomic_ids)])
        bop2 = torch.tensor([self.rbondparam[(i, j)][14] for (i, j) in zip(atomic_ids, atomic_ids)])
        pdp = torch.tensor([self.rbondparam[(i, j)][10] for (i, j) in zip(atomic_ids, atomic_ids)])
        ptp = torch.tensor([self.rbondparam[(i, j)][11] for (i, j) in zip(atomic_ids, atomic_ids)])
        pdo = torch.tensor([self.rbondparam[(i, j)][5] for (i, j) in zip(atomic_ids, atomic_ids)])
        popi = torch.tensor([self.rbondparam[(i, j)][7] for (i, j) in zip(atomic_ids, atomic_ids)])

        aval = torch.tensor([self.ratomparam[i][2] for i in atomic_ids])
        vval3 = torch.tensor([self.ratomparam[i][28] for i in atomic_ids])    # TODO: check if this is correct regarding the first row elements

        vpar1 = self.vpar[1]
        vpar2 = self.vpar[2]
        bo132 = torch.tensor([self.ratomparam[i][21] for i in atomic_ids])
        bo131 = torch.tensor([self.ratomparam[i][20] for i in atomic_ids])
        bo133 = torch.tensor([self.ratomparam[i][22] for i in atomic_ids])

        return rob1, rob2, rob3, bop1, bop2, pdp, ptp, pdo, popi, aval, vval3, vpar1, vpar2, bo132, bo131, bo133


    def set_parameters(self, rob1, rob2, rob3, bop1, bop2, pdp, ptp, pdo, popi, aval, vval3, vpar1, vpar2, bo132, bo131, bo133):
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
        self.bo132 = bo132
        self.bo131 = bo131
        self.bo133 = bo133


    def forward(self, distances):
        '''
        Parameters:
            distances: interatomic distances

        Returns:
            bo: sigma bond order, BO_ij^sigma
            bopi: pi bond order, BO_ij^pi
            bopi2: double pi bond order, BO_ij^pipi
            abo: total bond order, BO_ij
            ovi4: overcoordination, Delta_i
        '''
        rij = distances

        # BO'_ij^sigma = exp(p_bo1 * (rij / r_o^sigma) ^ p_bo2)
        borsi = torch.exp(self.bop1 * (rij * self.rob1) ** self.bop2)
        # BO'_ij^pi = exp(p_bo3 * (rij / r_o^pi) ^ p_bo4)
        bopi = torch.exp(self.pdp * (rij * self.rob2) ** self.ptp)
        # BO'_ij^pipi = exp(p_bo5 * (rij / r_o^pipi) ^ p_bo6)
        bopi2 = torch.exp(self.pdo * (rij * self.rob3) ** self.popi)
        # BO'_ij = BO'_ij^sigma + BO'_ij^pi + BO'_ij^pipi
        bor = borsi + bopi + bopi2
        
        # BO'_i = sum_j BO'_ij
        aboi = torch.sum(bor, dim=-1)
        # Delta'_i = -Val_i + sum_j BO'_ij
        ovi = -self.aval + aboi
        # Delta'boc_i = -Val'boc_i + sum_j BO'_ij
        ovi2 = -self.vval3 + aboi

        # f2(Delta'_i, Delta'_j) = exp(-p_boc1 * Delta'_i) + exp(-p_boc1 * Delta'_j)
        exp1 = torch.exp(-self.vpar1 * ovi)
        ovcor1 = exp1.unsqueeze(-1) + exp1.unsqueeze(-2)
        # f3(Delta'_i, Delta'_j) = -1 / p_boc2 * ln(1 / 2 * (exp(-p_boc2 * Delta'_i) + exp(-p_boc2 * Delta'_j)
        exp2 = torch.exp(-self.vpar2 * ovi)
        ovcor2 = - torch.log(0.5 * (exp2.unsqueeze(-1) + exp2.unsqueeze(-2))) / self.vpar2
        # f1(Delta'_i, Delta'_j) = 1 / 2 * ((Val_i + f2(Delta'_i, Delta'_j)) / (Val_i + f2(Delta'_i, Delta'_j) + f3(Delta'_i, Delta'_j)) + (Val_j + f2(Delta'_i, Delta'_j)) / (Val_j + f2(Delta'_i, Delta'_j) + f3(Delta'_i, Delta'_j)))
        frac1 = (self.aval.unsqueeze(-1) + ovcor1) / (self.aval.unsqueeze(-1) + ovcor1 + ovcor2)
        coortot = 0.5 * (frac1 + torch.transpose(frac1, -1, -2))
        # f4 = 1 / (1 + exp(-p_boc3 * (p_boc4 * BO'_ij^2 - Delta'boc_i) + p_boc5))
        vp132 = torch.sqrt(self.bo132.unsqueeze(-1) * self.bo132.unsqueeze(-2))    # p_boc3 = sqrt(p_boc3_i * p_boc3_j)
        vp131 = torch.sqrt(self.bo131.unsqueeze(-1) * self.bo131.unsqueeze(-2))    # p_boc4 = sqrt(p_boc4_i * p_boc4_j)
        vp133 = torch.sqrt(self.bo133.unsqueeze(-1) * self.bo133.unsqueeze(-2))    # p_boc5 = sqrt(p_boc5_i * p_boc5_j)
        bocor1 = torch.sigmoid(vp132 * (vp131 * bor ** 2 - ovi2.unsqueeze(-1)) - vp133)
        # f5 = 1 / (1 + exp(-p_boc3 * (p_boc4 * BO'_ij^2 - Delta'boc_j) + p_boc5))
        bocor2 = torch.transpose(bocor1, -1, -2)
        # BO_ij^sigma = BO'_ij^sigma * f1 * f4 * f5  # TODO: check if this is correct regarding bond order cutoff "cutoff = 0.01 * vpar(30)"
        bo = borsi * coortot * bocor1 * bocor2
        # BO_ij^pi = BO'_ij^pi * f1^2 * f4 * f5
        bopi = bopi * coortot ** 2 * bocor1 * bocor2
        # BO_ij^pipi = BO'_ij^pipi * f1^2 * f4 * f5
        bopi2 = bopi2 * coortot ** 2 * bocor1 * bocor2
        # BO_ij = BO_ij^sigma + BO_ij^pi + BO_ij^pipi
        abo = bo + bopi + bopi2

        # Delta_i = -Val_i + sum_j BO_ij
        ovi4 = -self.aval + torch.sum(abo, dim=-1)

        return bo, bopi, bopi2, abo, ovi4
        

