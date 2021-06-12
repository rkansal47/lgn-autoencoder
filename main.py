from args import setup_argparse
from utils.make_data import initialize_data
from utils.utils import create_model_folder
from utils.train import train_loop
from lgn.models.lgn_encoder import LGNEncoder
from lgn.models.lgn_decoder import LGNDecoder
from lgn.models.autotest.lgn_tests import lgn_tests
from lgn.models.autotest.utils import plot_all_dev

import torch
import warnings
import os.path as osp

import logging
import sys
sys.path.insert(1, 'lgn/')
if not sys.warnoptions:
    warnings.simplefilter("ignore")


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # torch.autograd.set_detect_anomaly(True)
    args = setup_argparse()

    data_path = osp.join(args.file_path, f"{args.jet_type}_{args.file_suffix}")

    train_loader, valid_loader, test_loader = initialize_data(path=data_path,
                                                              batch_size=args.batch_size,
                                                              num_train=args.num_train,
                                                              num_val=args.num_val,
                                                              num_test=args.num_test,
                                                              test_batch_size=args.test_batch_size)

    """Initializations"""
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
                         device=args.device, dtype=args.dtype)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), args.lr)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), args.lr)

    # Both on gpu
    if (next(encoder.parameters()).is_cuda and next(encoder.parameters()).is_cuda):
        logging.info('The models are initialized on GPU...')
    # One on cpu and the other on gpu
    elif (next(encoder.parameters()).is_cuda or next(encoder.parameters()).is_cuda):
        raise AssertionError("The encoder and decoder are not trained on the same device!")
    # Both on cpu
    else:
        logging.info('The models are initialized on CPU...')

    if not args.equivariance_test_only:
        logging.info(f'Training over {args.num_epochs} epochs...')

        '''Training'''
        # Load existing model
        if args.load_to_train:
            outpath = args.load_path
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.load_epoch}_encoder_weights.pth'), map_location=args.device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.load_epoch}_decoder_weights.pth'), map_location=args.device))
        # Create new model
        else:
            outpath = create_model_folder(args)

        outpath = create_model_folder(args)
        train_loop(args, train_loader, valid_loader, encoder, decoder, optimizer_encoder, optimizer_decoder, outpath, args.device)

        # Equivariance tests
        if args.equivariance_test:
            encoder.load_state_dict(torch.load(osp.join(outpath, f'weights_encoder/epoch_{args.num_epochs}_encoder_weights.pth'), map_location=args.test_device))
            decoder.load_state_dict(torch.load(osp.join(outpath, f'weights_decoder/epoch_{args.num_epochs}_decoder_weights.pth'), map_location=args.test_device))
            dev = lgn_tests(encoder, decoder, test_loader, args, alpha_max=args.alpha_max,
                            theta_max=args.theta_max, epoch=args.num_epochs, cg_dict=encoder.cg_dict)
            plot_all_dev(dev, osp.join(outpath, 'model_evaluations/equivariance_tests'))

        logging.info("Training completed!")
    else:
        if args.load_path is not None:
            loadpath = args.load_path
        else:
            raise RuntimeError("load-path cannot be None if equivariance-test-only is True!")
        load_epoch = args.load_epoch if args.load_epoch is not None else args.num_epochs

        encoder.load_state_dict(torch.load(osp.join(loadpath, f'weights_encoder/epoch_{load_epoch}_encoder_weights.pth'), map_location=args.test_device))
        decoder.load_state_dict(torch.load(osp.join(loadpath, f'weights_decoder/epoch_{load_epoch}_decoder_weights.pth'), map_location=args.test_device))

        dev = lgn_tests(encoder, decoder, test_loader, args, alpha_max=args.alpha_max,
                        theta_max=args.theta_max, epoch=args.num_epochs, cg_dict=encoder.cg_dict)
        plot_all_dev(dev, osp.join(loadpath, 'model_evaluations/equivariance_tests'))

        logging.info("Done!")
