import torch
import logging

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions
from lgn.cg_lib.zonal_functions import p_cplx_to_rep, rep_to_p
from lgn.g_lib import GTau, GVec

from lgn.models.lgn_cg import LGNCG
from lgn.nn import RadialFilters
from lgn.nn import MixReps

from lgn.models.utils import adapt_var_list


class LGNDecoder(CGModule):
    """
    The encoder of the LGN autoencoder.

    Attributes
    ----------
    tau_latent_scalar : int
        The multiplicity of scalars per particle in the latent space.
    tau_latent_vector : int
        The multiplicity of vectors per particle in the latent space.
    num_output_particles : int
        The number of particles of jets in the latent space.
        For the hls4ml 150-p jet data, this should be 150.
    tau_output_scalars : int
        Multiplicity of Lorentz scalars (0,0) in the latent_space.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle invariant mass -p^2).
    tau_output_vectors : int
        Multiplicity of Lorentz 4-vectors (1,1) in the latent_space.
        For the hls4ml 150-p jet data, it should be 1 (namely the particle 4-momentum).
    maxdim : list of int
        Maximum weight in the output of CG products, expanded or truncated to list of
        length len(num_channels) - 1.
    num_basis_fn : int
        The number of basis function to use.
    num_channels : list of int
        Number of channels that the outputs of each CG layer are mixed to.
    max_zf : list of int
        Maximum weight in the output of the spherical harmonics, expanded or truncated to list of
        length len(num_channels) - 1.
    weight_init : str
        The type of weight initialization. The choices are 'randn' and 'rand'.
    level_gain : list of `floats`
        The gain at each level. (args.level_gain = [1.])
    activation : str
        Optional, default: 'leakyrelu'
        The activation function for lgn.LGNCG
    scale : float or int
        Optional, default: 1.
        Scaling parameter for input node features.
    mlp : bool
        Optional, default: True
        Whether to include the extra MLP layer on scalar features in nodes.
    mlp_depth : int
        Optional, default: None
        The number of hidden layers in CGMLP.
    mlp_width : list of int
        Optional, default: None
        The number of perceptrons in each CGMLP layer
    device : `torch.device`
        Optional, default: None, in which case it will be set to
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        The device to which the module is initialized.
    dtype : `torch.dtype`
        Optional, default: None, in which case it will be set to torch.float64
        The data type to which the module is initialized.
    cg_dict : `CGDict`
        Optional, default: None
        Clebsch-gordan dictionary for taking the CG decomposition.
    """

    def __init__(self, tau_latent_scalars, tau_latent_vectors,
                 num_output_particles, tau_output_scalars, tau_output_vectors,
                 maxdim, num_basis_fn, num_channels, max_zf, weight_init, level_gain,
                 activation='leakyrelu', mlp=True, mlp_depth=None, mlp_width=None,
                 device=None, dtype=None, cg_dict=None):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if dtype is None:
            dtype = torch.float64

        num_cg_levels = len(num_channels) - 1

        level_gain = adapt_var_list(level_gain, num_cg_levels)
        maxdim = adapt_var_list(maxdim, num_cg_levels)
        max_zf = adapt_var_list(max_zf, num_cg_levels)

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)
        logging.info(f'Initializing decoder with device: {self.device} and dtype: {self.dtype}')

        # Member varibles
        self.input_basis = 'canonical'  # Will convert it from Cartesian
        self.tau_latent_scalars = tau_latent_scalars
        self.tau_latent_vectors = tau_latent_vectors
        self.tau_dict = {'input': GTau({(0, 0): tau_latent_scalars, (1, 1): tau_latent_vectors})}

        self.num_output_particles = num_output_particles

        self.num_cg_levels = num_cg_levels
        self.num_basis_fn = num_basis_fn
        self.max_zf = max_zf
        self.num_channels = num_channels

        self.mlp = mlp
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.activation = activation

        tau_mix_to_graph = GTau({**{weight: self.num_output_particles for weight in [(0, 0), (1, 1)]},
                                 **{(l, l): 1 for l in range(2, max_zf[0] + 1)}})
        self.latent_to_graph = MixReps(self.tau_dict['input'], tau_mix_to_graph, device=self.device, dtype=self.dtype)

        tau_mix_to_cg = GTau({weight: num_channels[0] for weight in [(0, 0), (1, 1)]})
        self.input_func_node = MixReps(GTau({weight: 1 for weight in [(0, 0), (1, 1)]}),
                                       tau_mix_to_cg, device=self.device, dtype=self.dtype)

        self.zonal_fns_in = ZonalFunctions(max(self.max_zf), basis=self.input_basis,
                                           dtype=dtype, device=device, cg_dict=cg_dict)
        self.zonal_fns = ZonalFunctionsRel(max(self.max_zf), basis=self.input_basis,
                                           dtype=dtype, device=device, cg_dict=cg_dict)

        # Position functions
        self.rad_funcs = RadialFilters(self.max_zf, self.num_basis_fn, self.num_channels, self.num_cg_levels,
                                       input_basis=self.input_basis, device=self.device, dtype=self.dtype)
        tau_pos = self.rad_funcs.tau

        tau_input_node = self.input_func_node.tau

        self.lgn_cg = LGNCG(maxdim, self.max_zf, tau_input_node, tau_pos, self.num_cg_levels, self.num_channels,
                            level_gain, weight_init, mlp=self.mlp, mlp_depth=self.mlp_depth, mlp_width=self.mlp_width,
                            activation=self.activation, device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        self.tau_cg_levels_node = self.lgn_cg.tau_levels_node
        self.tau_dict['cg_layers'] = self.tau_cg_levels_node.copy()

        self.tau_output = {weight: 1 for weight in self.tau_cg_levels_node[-1].keys()}
        self.tau_output[(0, 0)] = 1
        self.tau_output[(1, 1)] = 1
        self.tau_dict['output'] = self.tau_output
        self.mix_to_output = MixReps(self.tau_cg_levels_node[-1], self.tau_output, device=self.device, dtype=self.dtype)

        logging.info(f'Decoder initialized. Number of parameters: {sum(p.nelement() for p in self.parameters() if p.requires_grad)}')

    def forward(self, latent_features, covariance_test=False, nodes_all=None):
        '''
        The forward pass of the LGN GNN.

        Parameters
        ----------
        node_features : `dict`
            The dictionary of node_features from the encoder. The keys are (0,0) (scalar) and (1,1) (vector)
        covariance_test : bool
            Optional, default: False
            If False, return prediction (scalar reps) only.
            If True, return both generated output and full node features, where the full node features
            will be used to test Lorentz covariance.
        nodes_all : list of GVec
            Optional, default: None
            The full node features in the encoder.

        Returns
        -------
        node_features : `dict`
            The dictionary that stores all relevant irreps.
        If covariance_test is True, also:
            nodes_all : list of GVec
                The full node features in both encoder and decoder.
        '''
        if covariance_test and (nodes_all is None):
            raise ValueError('covariance_test is set to True, but the full node features from the encoder is not passed in!')
        # Get data
        node_ps, node_scalars, node_mask, edge_mask = self._prepare_input(latent_features)

        zonal_functions_in, _, _ = self.zonal_fns_in(node_ps)

        zonal_functions, norms, sq_norms = self.zonal_fns(node_ps, node_ps)

        decoder_nodes_all = []

        if self.num_cg_levels > 0:
            rad_func_levels = self.rad_funcs(norms, edge_mask * (norms != 0).byte())
            node_reps_in = self.input_func_node(zonal_functions_in)
            decoder_nodes_all.append(node_reps_in)
        else:
            rad_func_levels = []
            node_reps_in = self.input_func_node(node_scalars, node_mask)
            decoder_nodes_all.append(node_reps_in)

        # CG layers
        decoder_cg_nodes = self.lgn_cg(node_reps_in, node_mask, rad_func_levels, zonal_functions)
        for i in range(len(decoder_cg_nodes)):
            decoder_nodes_all.append(decoder_cg_nodes[i])

        node_features = decoder_cg_nodes[-1]

        # Mix to output
        # node_all[-1] is the updated feature in the last layer
        generated_features = self.mix_to_output(node_features)
        generated_features = GVec({weight: generated_features[weight] for weight in [(0, 0), (1, 1)]})  # Truncate higher order irreps than (1, 1)

        decoder_nodes_all.append(generated_features)
        generated_ps = generated_features[(1, 1)].clone()
        generated_ps = rep_to_p(generated_ps)  # Convert to Cartesian coordinates
        # Remove the dimension of multiplicity (i.e. each particle is represented by a single 4-vector), like in input jet
        generated_ps = generated_ps.squeeze(-2)

        if not covariance_test:
            return generated_ps
        else:
            for i in range(len(decoder_cg_nodes)):
                nodes_all.append(decoder_cg_nodes[i])
            nodes_all.append(generated_features)
            return generated_features, nodes_all


    def _prepare_input(self, latent_features):
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
        node_features = self.latent_to_graph(latent_features)
        node_features = {weight: value.squeeze(-3) for weight, value in node_features.items()}
        node_ps = node_features[(1, 1)].to(device=self.device)
        node_scalars = node_features[(0, 0)].to(device=self.device)

        batch_size = node_ps.shape[1]
        node_mask = torch.zeros(2, batch_size, self.num_output_particles).to(device=self.device)
        edge_mask = torch.zeros(2, batch_size, self.num_output_particles, self.num_output_particles).to(device=self.device)

        node_ps = p_cplx_to_rep(node_ps)[(1, 1)].to(device=self.device)  # Convert to canonical basis
        return node_ps, node_scalars, node_mask, edge_mask
