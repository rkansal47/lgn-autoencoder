from args import setup_argparse
from utils.utils import create_model_folder, latest_epoch
from utils.train import train_loop
from utils.initialize import initialize_autoencoder, initialize_data, initialize_test_data, initialize_optimizers

from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev

import torch
import os.path as osp
import logging

from lgn.models.lgn_encoder import LGNEncoder
from lgn.models.lgn_decoder import LGNDecoder

from torch import device
from utils.emd_loss.emd_loss import emd_loss

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

f = open("emd_args.txt", "r")



args_dict = eval(f.read())
args = objectview(args_dict)
f.close()


args


encoder = LGNEncoder(num_input_particles=args.num_jet_particles,
                     tau_input_scalars=args.tau_jet_scalars,
                     tau_input_vectors=args.tau_jet_vectors,
                     map_to_latent=args.map_to_latent,
                     tau_latent_scalars=args.tau_latent_scalars,
                     tau_latent_vectors=args.tau_latent_vectors,
                     maxdim=args.maxdim, max_zf=[1],
                     num_channels=args.encoder_num_channels,
                     weight_init=args.weight_init, level_gain=args.level_gain,
                     num_basis_fn=args.num_basis_fn, activation=args.activation, scale=args.scale,
                     mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                     device=args.device, dtype=args.dtype)

decoder = LGNDecoder(tau_latent_scalars=args.tau_latent_scalars,
                     tau_latent_vectors=args.tau_latent_vectors,
                     num_output_particles=args.num_jet_particles,
                     tau_output_scalars=args.tau_jet_scalars,
                     tau_output_vectors=args.tau_jet_vectors,
                     maxdim=args.maxdim, max_zf=[1],
                     num_channels=args.decoder_num_channels,
                     weight_init=args.weight_init, level_gain=args.level_gain,
                     num_basis_fn=args.num_basis_fn, activation=args.activation,
                     mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                     cg_dict=encoder.cg_dict, device=args.device, dtype=args.dtype)

encoder.load_state_dict(torch.load('emd_testing/encoder_weights.pt',
                                   map_location='cpu'))
decoder.load_state_dict(torch.load('emd_testing/decoder_weights.pt',
                                   map_location='cpu'))

p4_gen = torch.load('emd_testing/p4_gen.pt', map_location='cpu')
p4_target = torch.load('emd_testing/p4_target.pt', map_location='cpu')

args

def get_eps(args):
    if args.dtype in [torch.float64, torch.double]:
        return 1e-16
    else:
        return 1e-12


emd_loss(p4_target, p4_gen, eps=get_eps(args), device=args.device)
