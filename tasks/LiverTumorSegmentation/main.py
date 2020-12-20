import pprint
import os, time
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torch.optim as optim

from src.utils.logging import create_logger, save_checkpoint
from src.utils.measure import AverageMeter
from LITSDataset import LITSDataset

class Segmentation(object):
    def __init__(self, config, network=None, dataset=None):
        self.config = config

        # TODO: set network
        if network is not None:
            self.network = network

        # TODO: set dataset
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = LITSDataset

        # TODO: set loss
        self.loss = None

    def train(self):
        print(self.config['log_dir'])
        logger, final_output_dir, tb_log_dir = create_logger(self.config['log_dir'],
                                                             self.config['description'],
                                                             'train')

        logger.info(pprint.pformat(self.config))

        writer = SummaryWriter(log_dir=tb_log_dir)

        # TODO: set model
        model = self.network(in_channels=1)
        model = torch.nn.DataParallel(model, device_ids=[self.config.gpus]).cuda()

        # TODO: set data loader
        train_dataset = self.dataset(128, 128, 64,
                                    self.config['data_dir'] + "\\train",
                                    self.config['data_dir'] + "\\label",
                                    aug=[])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        best_perf = 0.0
        last_epoch = -1
        begin_epoch = 0

        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint.pth'
        )

        # TODO: set optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=1e-4)

        if os.path.exists(checkpoint_file) and self.config['auto_resume']:
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))

        # TODO: set learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=last_epoch)

        for epoch in range(begin_epoch, self.config['epoch']):
            model.train()

            losses = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            lr_scheduler.step()

            end = time.time()
            for idx, (image, label, contour_label, shape_label) in enumerate(train_loader):
                data_time.update(time.time() - end)

                image = image.type(torch.cuda.FloatTensor)
                label = label.type(torch.cuda.LongTensor)

                ''' class weight calculation
                true_class = np.round_(float(label.sum()) / label.reshape((-1)).size(0), 2)
                class_weights = torch.Tensor([true_class, 1 - true_class]).type(torch.cuda.FloatTensor)
                '''

                # TODO: set model, input, output and loss
                output = model(image)
                loss = self.loss(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), image.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                # TODO: add validation stage

                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f}'.format(
                    epoch + 1, idx + 1, len(train_loader), batch_time=batch_time,
                    speed=image.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses)
                logger.info(msg)

            pref_indicator = 0 # TODO: set metric function
            if pref_indicator > best_perf:
                best_perf = pref_indicator
                best_model = True
                best_model_state_file = os.path.join(
                    final_output_dir, 'best_model.pth'
                )
                logger.info('=> saving best model state to {}'.format(
                    best_model_state_file)
                )
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': self.config['description'],
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': pref_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, model, epoch + 1)

            # TODO: can add other measure to tensorboard
            writer.add_scalar('loss', losses.avg, epoch + 1)

        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(model.module.state_dict(), final_model_state_file)
        writer.close()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model):
        # TODO: validation
        valid_loss = 0

        with torch.no_grad():
            model.eval()

        model.train()
        return valid_loss

    def test(self):
        # TODO: add test metric, show/save result
        pass

if __name__ == '__main__':
    config = yaml.load(open("./tasks/LiverTumorSegmentation/config.yaml", "r"), Loader=yaml.FullLoader)

    seg = Segmentation(config)
    seg.train()