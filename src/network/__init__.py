from .network import *
from .transformer import *


def get_network(type):
    if type == 'mlp':
        return DensityNetwork
    elif type == 'dyc':
        return DynamicNetwork
    else:
        NotImplementedError('Unknown network typeß!')

