import argparse
import os

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss, create_loss
from lib.train.trainer import Trainer

import torch
def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ## FOR REPRODUCIBILITY OF RESULTS
    seed = 1777777
    utils.reproducibility(args, seed)

    utils.make_dirs(args.save)
    utils.save_arguments(args, args.save)
    print(args)
    training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
                                                                                               path='.././datasets')
    model, optimizer = medzoo.create_model(args)
    criterion = create_loss('CrossEntropyLoss')
    criterion = DiceLoss(classes=args.classes, weight=torch.tensor([0.1, 1, 1, 1]).cuda())

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    # trainer = Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
    #                   valid_data_loader=val_generator, lr_scheduler=None)

    # trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
    #                         valid_data_loader=val_generator)

    trainer = Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                        valid_data_loader=val_generator)
    print("START TRAINING...")
    trainer.training()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dataset_name', type=str, default="brats2019")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--samples_train', type=int, default=100)
    parser.add_argument('--samples_val', type=int, default=100)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--loadData', default=True)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='VNET',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args

if __name__ == '__main__':
    main()
