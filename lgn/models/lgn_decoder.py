import torch
import logging

from lgn.cg_lib import normsq4
from lgn.models.lgn_graphnet import LGNGraphNet

class LGNEncoder(LGNGraphNet):
    """
    The encoder of the LGN autoencoder.

    Parameters
    ----------
    num_latent_particles : `int`
        The number of particles in the latent sapce.
    tau_latent_scalar : `int`
        The multiplicity of scalars per particle in the latent space.
    tau_latent_vector : `int`
        The multiplicity of vectors per particle in the latent space.
    num_output_particles : `int`
        The number of particles of jets in the latent space.
        For the hls4ml 150-p jet data, this should be 150.
    tau_output_scalars : `int`
        Multiplicity of Lorentz scalars (0,0) in the latent_space.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle invariant mass -p^2).
    tau_output_vectors : `int`
        Multiplicity of Lorentz 4-vectors (1,1) in the latent_space.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle 4-momentum).
    maxdim : `list` of `int`
        Maximum weight in the output of CG products, expanded or truncated to list of
        length len(num_channels) - 1.
    num_basis_fn : `int`
        The number of basis function to use.
    num_channels : `list` of `int`
        Number of channels that the outputs of each CG layer are mixed to.
    max_zf : `list` of `int`
        Maximum weight in the output of the spherical harmonics, expanded or truncated to list of
        length len(num_channels) - 1.
    weight_init : `str`
        The type of weight initialization. The choices are 'randn' and 'rand'.
    level_gain : `list` of `floats`
        The gain at each level. (args.level_gain = [1.])
    activation : `str`
        Optional, default: 'leakyrelu'
        The activation function for lgn.LGNCG
    scale : `float` or `int`
        Optional, default: 1.
        Scaling parameter for input node features.
    mlp : `bool`
        Optional, default: True
        Whether to include the extra MLP layer on scalar features in nodes.
    mlp_depth : `int`
        Optional, default: None
        The number of hidden layers in CGMLP.
    mlp_width : `list` of `int`
        Optional, default: None
        The number of perceptrons in each CGMLP layer
    device : `torch.device`
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : `torch.dtype`
        Optional, default: torch.float64
        The data type to which the module is initialized.
    cg_dict : `CGDict`
        Optional, default: None
        Clebsch-gordan dictionary for taking the CG decomposition.
    """
    def __init__(self, num_latent_particles, tau_latent_scalar, tau_latent_vector,
                 num_output_particles, tau_output_scalars, tau_output_vectors,
                 maxdim, num_basis_fn, num_channels, max_zf, weight_init, level_gain,
                 activation='leakyrelu', scale=1., mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=torch.float64, cg_dict=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        logging.info(f'Initializing encoder with device: {device} and dtype: {dtype}')

        super().__init__(num_input_particles=num_latent_particles, input_basis='canonical',
                         tau_input_scalars=tau_latent_scalar, tau_input_vectors=tau_latent_vector,
                         num_output_partcles=num_output_particles, tau_output_scalars=tau_output_scalars, tau_output_vectors=tau_output_vectors,
                         max_zf=max_zf, maxdim=maxdim, num_channels=num_channels,
                         weight_init=weight_init, level_gain=level_gain, num_basis_fn=num_basis_fn,
                         activation=activation, mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                         device=device, dtype=dtype, cg_dict=cg_dict)

        self.scale = scale

    '''
    The forward pass of the LGN GNN.

    Parameters
    ----------
    node_features : `dict`
        The dictionary of node_features from the encoder. The keys are (0,0) (scalar) and (1,1) (vector)
    covariance_test : `bool`
        Optional, default: False
        If False, return prediction (scalar reps) only.
        If True, return both generated output and full node features, where the full node features
        will be used to test Lorentz covariance.
    nodes_all : `list` of GVec
        Optional, default: None
        The full node features in the encoder.

    Returns
    -------
    node_features : `dict`
        The dictionary that stores all relevant irreps.
    If covariance_test is True, also:
        nodes_all : `list` of GVec
            The full node features in both encoder and decoder.
    '''
    def forward(self, data, covariance_test=False, nodes_all=None):
        # Get data
        node_scalars, node_ps, node_mask, edge_mask = self._prepare_input(data)

        # Can be simplied as self.graph_net(node_scalars, node_ps, node_mask, edge_mask, covariance_test)
        if not covariance_test:
            latent_features, node_mask, edge_mask = super(LGNEncoder, self).forward(node_scalars, node_ps, node_mask, edge_mask, covariance_test)
            return latent_features, edge_mask, edge_mask
        else:
            latent_features, node_mask, edge_mask, nodes_all = super(LGNEncoder, self).forward(node_scalars, node_ps, node_mask, edge_mask, covariance_test)
            return latent_features, edge_mask, edge_mask, nodes_all
    """
    Extract input from data.

    Parameters
    ----------
    data : `dict`
        The jet data.

    Returns
    -------
    scalars : `torch.Tensor`
        Tensor of scalars for each node.
    node_ps: : `torch.Tensor`
        Momenta of the nodes
    node_mask : `torch.Tensor`
        Node mask used for batching data.
    edge_mask: `torch.Tensor`
        Edge mask used for batching data.
    """
    def _prepare_input(self, data):

        node_ps = data['p4'].to(device=self.device, dtype=self.dtype) * self.scale

        data['p4'].requires_grad_(True)

        node_mask = data['node_mask'].to(device=self.device, dtype=torch.uint8)
        edge_mask = data['edge_mask'].to(device=self.device, dtype=torch.uint8)

        scalars = torch.ones_like(node_ps[:,:, 0]).unsqueeze(-1)
        scalars = normsq4(node_ps).abs().sqrt().unsqueeze(-1)

        if 'scalars' in data.keys():
            scalars = torch.cat([scalars, data['scalars'].to(device=self.device, dtype=self.dtype)], dim=-1)

        return scalars, node_ps, node_mask, edge_mask