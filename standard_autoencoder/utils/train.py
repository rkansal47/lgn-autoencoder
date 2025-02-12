import torch.nn as nn
import torch
import os.path as osp
import time
from utils.utils import make_dir, save_data, plot_eval_results, eps
from utils.jet_analysis import plot_p
import logging


def train(args, loader, encoder, decoder, optimizer_encoder, optimizer_decoder,
          epoch, outpath, is_train=True, device=None):

    if is_train:
        assert (optimizer_encoder is not None) and (optimizer_decoder is not None), "Please specify the optimizers."
        encoder.train()
        decoder.train()
        encoder_weight_path = make_dir(osp.join(outpath, "weights_encoder"))
        decoder_weight_path = make_dir(osp.join(outpath, "weights_decoder"))
    else:
        encoder.eval()
        decoder.eval()

    target_data = []
    generated_data = []
    epoch_total_loss = 0

    for i, data in enumerate(loader):
        p4_target = data
        p4_gen = decoder(encoder(p4_target, metric=args.encoder_metric), metric=args.decoder_metric)
        generated_data.append(p4_gen.cpu().detach())

        if device is not None:
            p4_target = p4_target.to(device=device)
        target_data.append(p4_target.cpu().detach())

        batch_loss = get_loss(args, p4_gen, p4_target.to(args.dtype))
        epoch_total_loss += batch_loss.item()

        # Backward propagation
        if is_train:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            batch_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            # save model
            if ((i % args.save_freq) == 0 and i > 0):
                torch.save(encoder.state_dict(), osp.join(
                    encoder_weight_path, f"epoch_{epoch}_encoder_weights.pth"))
                torch.save(decoder.state_dict(), osp.join(
                    decoder_weight_path, f"epoch_{epoch}_decoder_weights.pth"))

    generated_data = torch.cat(generated_data, dim=0).numpy()
    target_data = torch.cat(target_data, dim=0).numpy()

    epoch_avg_loss = epoch_total_loss.cpu().item() / len(loader)
    save_data(data=epoch_avg_loss, data_name='loss', is_train=is_train, outpath=outpath, epoch=epoch)

    # Save weights
    if is_train:
        torch.save(encoder.state_dict(), osp.join(encoder_weight_path, f"epoch_{epoch}_encoder_weights.pth"))
        torch.save(decoder.state_dict(), osp.join(decoder_weight_path, f"epoch_{epoch}_decoder_weights.pth"))

    return epoch_avg_loss, generated_data, target_data


@torch.no_grad()
def validate(args, loader, encoder, decoder, epoch, outpath, device):
    with torch.no_grad():
        epoch_avg_loss, generated_data, target_data = train(args, loader=loader, encoder=encoder, decoder=decoder,
                                                            optimizer_encoder=None, optimizer_decoder=None,
                                                            epoch=epoch, outpath=outpath, is_train=False, device=device)
    return epoch_avg_loss, generated_data, target_data


def train_loop(args, train_loader, valid_loader, encoder, decoder,
               optimizer_encoder, optimizer_decoder, outpath, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert (args.save_dir is not None), "Please specify directory of saving the models!"
    make_dir(args.save_dir)

    train_avg_losses = []
    valid_avg_losses = []
    dts = []

    outpath_train_jet_plots = make_dir(osp.join(outpath, 'model_evaluations/jet_plots/train'))
    outpath_valid_jet_plots = make_dir(osp.join(outpath, 'model_evaluations/jet_plots/valid'))

    for ep in range(args.num_epochs):
        epoch = args.load_epoch + ep if args.load_to_train else ep + 1

        # Training
        start = time.time()
        train_avg_loss, train_gen, train_target = train(args, train_loader, encoder, decoder,
                                                        optimizer_encoder, optimizer_decoder, epoch,
                                                        outpath, is_train=True, device=device)
        # Validation
        valid_avg_loss, valid_gen, valid_target = validate(args, valid_loader, encoder, decoder,
                                                           epoch, outpath, device=device)

        dt = time.time() - start
        save_data(data=dt, data_name='dts', is_train=None,
                  outpath=outpath, epoch=epoch)
        save_data(data=train_avg_loss, data_name='losses', is_train=True,
                  outpath=outpath, epoch=epoch)
        save_data(data=valid_avg_loss, data_name='losses', is_train=False,
                  outpath=outpath, epoch=epoch)
        if args.unit.lower() == 'tev':
            # Convert to GeV for plotting
            train_target *= 1000
            train_gen *= 1000
            valid_target *= 1000
            valid_gen *= 1000

        for target, gen, dir in zip((train_target, valid_target),
                                    (train_gen, valid_gen),
                                    (outpath_train_jet_plots, outpath_valid_jet_plots)):
            logging.debug("plotting")
            plot_p(args, target_data=target, gen_data=gen, save_dir=dir,
                   polar_max=args.polar_max, cartesian_max=args.cartesian_max,
                   jet_polar_max=args.jet_polar_max, jet_cartesian_max=args.jet_cartesian_max,
                   num_bins=args.num_bins, cutoff=args.cutoff, epoch=epoch)

        dts.append(dt)
        train_avg_losses.append(train_avg_loss)
        valid_avg_losses.append(valid_avg_loss)

        logging.info(f'{epoch=}/{args.num_epochs if not args.load_to_train else args.num_epochs + args.load_epoch}, '
                     f'train_loss={train_avg_loss}, valid_loss={valid_avg_loss}, {dt=}')

        if (epoch > 1) and (epoch % 10 == 0):
            plot_eval_results(args, data=(train_avg_losses[-10:], valid_avg_losses[-10:]),
                              data_name=f"losses from {epoch-10} to {epoch}",
                              outpath=outpath, start=epoch-10)
        if (epoch > 1) and (epoch % 100 == 0):
            plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                              data_name='Losses', outpath=outpath, start=epoch-10)

    # Save global data
    save_data(data=train_avg_losses, data_name='losses', is_train=True, outpath=outpath, epoch=-1)
    save_data(data=valid_avg_losses, data_name='losses', is_train=False, outpath=outpath, epoch=-1)
    save_data(data=dts, data_name='dts', is_train=None, outpath=outpath, epoch=-1)

    plot_eval_results(args, data=(train_avg_losses, valid_avg_losses),
                      data_name='Losses', outpath=outpath)
    plot_eval_results(args, data=dts, data_name='Time durations',
                      outpath=outpath)

    return train_avg_losses, valid_avg_losses, dts


def get_loss(args, p4_gen, p4_target):
    if args.loss_choice.lower() in ['chamfer', 'chamferloss', 'chamfer_loss']:
        from utils.chamfer_loss import ChamferLoss
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = chamferloss(p4_gen, p4_target, jet_features_weight=args.chamfer_jet_features_weight)  # output, target
        return batch_loss

    if args.loss_choice.lower() in ['emd', 'emdloss', 'emd_loss']:
        from utils.emd_loss import emd_loss
        batch_loss = emd_loss(p4_target, p4_gen, eps=eps(args), device=args.device)  # true, output
        return batch_loss

    if args.loss_choice.lower() in ['mse', 'mseloss', 'mse_loss']:
        mseloss = nn.MSELoss()
        batch_loss = mseloss(p4_gen, p4_target)  # output, target
        return batch_loss

    if args.loss_choice.lower() in ['hybrid', 'combined', 'mix']:
        from utils.chamfer_loss import ChamferLoss
        from utils.emd_loss import emd_loss
        chamferloss = ChamferLoss(loss_norm_choice=args.loss_norm_choice)
        batch_loss = args.chamfer_loss_weight * chamferloss(p4_gen, p4_target, jet_features_weight=args.chamfer_jet_features_weight) + \
            emd_loss(p4_target, p4_gen, eps=eps(args), device=args.device)
        return batch_loss
