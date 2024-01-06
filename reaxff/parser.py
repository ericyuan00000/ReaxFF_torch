import pickle
import torch


def read_reaxff_params(filename):
    '''
    Parameters:
        filename: path to the parameter pickle file
    '''
    params = pickle.load(open(filename, 'rb'))

    # General parameters
    vpar = torch.tensor(params['vpar'])

    # Atomic parameters
    atomtypes = list(params['ratomparam'].keys())
    ratomparam = torch.tensor(list(params['ratomparam'].values()))

    # Bond parameters
    n_param = len(next(iter(params['rbondparam'].values())))
    rbondparam = torch.full((len(atomtypes), len(atomtypes), n_param), torch.nan)
    for (i, j) in params['rbondparam'].keys():
        rbondparam[i, j] = torch.tensor(params['rbondparam'][(i, j)])
        rbondparam[j, i] = torch.tensor(params['rbondparam'][(i, j)])

    # Off-diagonal parameters
    n_param = len(next(iter(params['roffdiagparam'].values())))
    roffdiagparam = torch.full((len(atomtypes), len(atomtypes), n_param), torch.nan)
    for (i, j), param in params['roffdiagparam'].items():
        roffdiagparam[i, j] = torch.tensor(param)
        roffdiagparam[j, i] = torch.tensor(param)

    # Angle parameters
    n_param = len(next(iter(params['rangparam'].values())))
    rangparam = torch.full((len(atomtypes), len(atomtypes), len(atomtypes), n_param), torch.nan)
    for (i, j, k) in params['rangparam'].keys():
        rangparam[i, j, k] = torch.tensor(params['rangparam'][(i, j, k)])
        rangparam[k, j, i] = torch.tensor(params['rangparam'][(i, j, k)])

    # Torsion parameters
    rtorangparam = params['rtorangparam']

    # Hydrogen bond parameters
    rhbondparam = params['rhbondparam']