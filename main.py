import random
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset import CityscapesDataManager
from unet_solver import UNetSolver
from pix2pix_solver import Pix2PixSolver

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'pix2pix'],
                        help='choose which solver to use: standard unet or pix2pix gan')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of elements in batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in data loader')
    parser.add_argument('--print_every', type=int, default=500, help='print losses every N iteration')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (for standard UNet)')
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer used for training')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability (0.0 = off)')
    parser.add_argument('--dropout_mode', type=str, default='full', choices=['full', 'paper'],
                        help="Modalità Dropout: 'full' (applied to the entire generator), 'paper' (applied only to the half of the discriminator)")
    parser.add_argument('--n_layer_g', type=int, default=4, help='Number of layers for encoder/decoder -> total layer N+N')

    parser.add_argument('--lr_g', type=float, default=0.0002, help='Generator learning rate (Pix2Pix)')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate (Pix2Pix)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 parameter for Adam (Pix2Pix)')
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='Weight for L1 loss in Pix2Pix')
    parser.add_argument('--n_layer_d', type=int, default=3, help='Number of layers in discriminator')
    parser.add_argument('--gan_mode', type=str, default='vanilla', choices=['vanilla', 'lsgan'],
                        help='It select the GAN loss: vanilla (BCEWithLogits) o lsgan (MSE)')

    parser.add_argument('--norm_type', type=str, default='batch',choices=['batch', 'instance', 'none'],
                        help='Normalization Type: batch, instance o none')
    parser.add_argument('--feat', type=int, default=16, help='number of features in model')
    parser.add_argument('--loss_function', default='L1', choices=['L1', 'MSE'], help='loss function used for training')

    parser.add_argument('--dataset_path', type=str, default='./data', help='path were to save/get the dataset')
    parser.add_argument('--augment', action='store_true', help='Attiva Horizontal Flip e Color Jittering')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')

    parser.add_argument('--telegram_token', type=str, help='telegram bot token')
    parser.add_argument('--telegram_chat_id', type=str, help='telegram bot chat_id')

    return parser.parse_args()


def main(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    data_manager = CityscapesDataManager(args)
    trainloader, testloader = data_manager.get_loaders()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps") # For MacOs
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Device: ", device)
    print(f"Model Type: {args.model_type}")

    # Select the cGAN mode or the basic UNet
    if args.model_type == 'pix2pix':
        solver = Pix2PixSolver(train_loader=trainloader,
                               test_loader=testloader,
                               device=device,
                               writer=writer,
                               args=args)

    else:
        solver = UNetSolver(train_loader=trainloader,
                            test_loader=testloader,
                            device=device,
                            writer=writer,
                            args=args)

    solver.train()


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)