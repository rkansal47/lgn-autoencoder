import os
import os.path as osp
import torch
import matplotlib.pyplot as plt


def create_model_folder(args):
    make_dir(args.save_dir)
    return make_dir(osp.join(args.save_dir, get_model_fname(args)))


def get_model_fname(args):
    model_fname = f"StandardAutoencoder_{args.jet_type}Jet_LatentDim{args.latent_node_size}"
    return model_fname


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def eps(args):
    if args.dtype in [torch.float64, torch.double]:
        return 1e-16
    else:
        return 1e-12


def save_data(data, data_name, is_train, outpath, epoch=-1):
    '''
    Save data like losses and dts. If epoch is -1, the data will be considered a global data, such as
    the losses over all epochs.
    '''
    outpath = make_dir(osp.join(outpath, "model_evaluations/pt_files"))
    if isinstance(data, torch.Tensor):
        data = data.cpu()

    if is_train is None:
        if epoch >= 0:
            torch.save(data, osp.join(outpath, f'{data_name}_epoch_{epoch+1}.pt'))
        else:
            torch.save(data, osp.join(outpath, f'{data_name}.pt'))
        return

    if epoch >= 0:
        if is_train:
            torch.save(data, osp.join(outpath, f'train_{data_name}_epoch_{epoch+1}.pt'))
        else:
            torch.save(data, osp.join(outpath, f'valid_{data_name}_epoch_{epoch+1}.pt'))
    else:
        if is_train:
            torch.save(data, osp.join(outpath, f'train_{data_name}.pt'))
        else:
            torch.save(data, osp.join(outpath, f'valid_{data_name}.pt'))


def plot_eval_results(args, data, data_name, outpath, global_data=False, start=None):
    '''
    Plot evaluation results
    '''
    outpath = make_dir(osp.join(outpath, "model_evaluations/evaluation_plots"))
    if args.load_to_train:
        start = args.load_epoch + 1
        end = start + args.num_epochs
    else:
        start = 1 if start is None else start
        end = args.num_epochs

    # (train, label)
    if type(data) in [tuple, list] and len(data) == 2:
        train, valid = data
        if global_data:
            x = [i for i in range(start, end+1)]
        else:
            x = [start + i for i in range(len(train))]

        if isinstance(train, torch.Tensor):
            train = train.detach().cpu().numpy()
        if isinstance(valid, torch.Tensor):
            valid = valid.detach().cpu().numpy()
        plt.plot(x, train, label='Train', alpha=0.8)
        plt.plot(x, valid, label='Valid', alpha=0.8)
        plt.legend()
    # only one type of data (e.g. dt)
    else:
        if global_data:
            x = [i for i in range(start, end+1)]
        else:
            x = [start + i for i in range(len(train))]
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        plt.plot(x, data)

    plt.xlabel('Epoch')
    plt.ylabel(data_name)
    plt.title(data_name)
    save_name = "_".join(data_name.lower().split(" "))
    plt.savefig(osp.join(outpath, f"{save_name}.pdf"), bbox_inches='tight')
    plt.close()