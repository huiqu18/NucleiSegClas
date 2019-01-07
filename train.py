import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
from tensorboardX import SummaryWriter

from model import ResUNet34, UNet
import utils
from data_folder import DataFolder
from params import Params, get_transforms
from loss import perceptual_loss, vgg16_feat


def main():
    global params, best_iou, num_iter, tb_writer, logger, logger_results
    best_iou = 0
    params = Params()
    params.save_params('{:s}/params.txt'.format(params.paths['save_dir']))
    tb_writer = SummaryWriter('{:s}/tb_logs'.format(params.paths['save_dir']))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in params.train['gpu'])

    # set up logger
    logger, logger_results = setup_logging(params)

    # ----- create model ----- #
    model_name = params.model['name']
    if model_name == 'ResUNet34':
        model = ResUNet34(params.model['out_c'], fixed_feature=params.model['fix_params'])
    elif params.model['name'] == 'UNet':
        model = UNet(3, params.model['out_c'])
    else:
        raise NotImplementedError()

    logger.info('Model: {:s}'.format(model_name))
    # if not params.train['checkpoint']:
    #     logger.info(model)
    model = nn.DataParallel(model)
    model = model.cuda()
    global vgg_model
    logger.info('=> Using VGG16 for perceptual loss...')
    vgg_model = vgg16_feat()
    vgg_model = nn.DataParallel(vgg_model).cuda()
    cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), params.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=params.train['weight_decay'])

    # ----- get pixel weights and define criterion ----- #
    if not params.train['weight_map']:
        criterion = torch.nn.NLLLoss().cuda()
    else:
        logger.info('=> Using weight maps...')
        criterion = torch.nn.NLLLoss(reduction='none').cuda()

    if params.train['beta'] > 0:
        logger.info('=> Using perceptual loss...')
        global criterion_perceptual
        criterion_perceptual = perceptual_loss()

    data_transforms = {'train': get_transforms(params.transform['train']),
                       'val': get_transforms(params.transform['val'])}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'val']:
        img_dir = '{:s}/{:s}'.format(params.paths['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(params.paths['label_dir'], x)
        if params.train['weight_map']:
            weight_map_dir = '{:s}/{:s}'.format(params.paths['weight_map_dir'], x)
            dir_list = [img_dir, weight_map_dir, target_dir]
            postfix = ['weight.png', 'label_with_contours.png']
            num_channels = [3, 1, 3]
        else:
            dir_list = [img_dir, target_dir]
            postfix = ['label_with_contours.png']
            num_channels = [3, 3]
        dsets[x] = DataFolder(dir_list, postfix, num_channels, data_transforms[x])
    train_loader = DataLoader(dsets['train'], batch_size=params.train['batch_size'], shuffle=True,
                              num_workers=params.train['workers'])
    val_loader = DataLoader(dsets['val'], batch_size=params.train['val_batch_size'], shuffle=False,
                            num_workers=params.train['workers'])

    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if params.train['checkpoint']:
        if os.path.isfile(params.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(params.train['checkpoint']))
            checkpoint = torch.load(params.train['checkpoint'])
            params.train['start_epoch'] = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(params.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(params.train['checkpoint']))

    # ----- training and validation ----- #
    num_iter = params.train['num_epochs'] * len(train_loader)

    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(params.train['lr']))
    logger.info("=> Batch size: {:d}".format(params.train['batch_size']))
    # logger.info("=> Number of training iterations: {:d}".format(num_iter))
    logger.info("=> Training epochs: {:d}".format(params.train['num_epochs']))
    logger.info("=> beta: {:.1f}".format(params.train['beta']))

    for epoch in range(params.train['start_epoch'], params.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, params.train['num_epochs']))
        train_results = train(train_loader, model, optimizer, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_iou_nuclei, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            val_results = validate(val_loader, model, criterion)
            val_loss, val_loss_ce, val_loss_var, val_iou_nuclei, val_iou = val_results

        # check if it is the best accuracy
        combined_iou = (val_iou_nuclei + val_iou) / 2
        is_best = combined_iou > best_iou
        best_iou = max(combined_iou, best_iou)

        cp_flag = (epoch+1) % params.train['checkpoint_freq'] == 0

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer': optimizer.state_dict(),
        }, epoch, is_best, params.paths['save_dir'], cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch+1, train_loss, train_loss_ce, train_loss_var, train_iou_nuclei,
                                    train_iou, val_loss, val_iou_nuclei, val_iou))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_ce': train_loss_ce,
                               'train_loss_var': train_loss_var, 'val_loss': val_loss}, epoch)
        tb_writer.add_scalars('epoch_accuracies',
                              {'train_iou_nuclei': train_iou_nuclei, 'train_iou': train_iou,
                               'val_iou_nuclei': val_iou_nuclei, 'val_iou': val_iou}, epoch)
    tb_writer.close()


def train(train_loader, model, optimizer, criterion, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(5)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        if params.train['weight_map']:
            input, weight_map, target = sample
            weight_map = weight_map.float().div(20)
            if weight_map.dim() == 4:
                weight_map = weight_map.squeeze(1)
            weight_map_var = weight_map.cuda()
        else:
            input, target = sample

        # no classification
        if params.model['out_c'] == 3:
            target[target==2] = 1
            target[target==3] = 1
            target[target==4] = 2

        # no edge or classification
        if params.model['out_c'] == 2:
            target[target>0] = 1

        # for b in range(input.size(0)):
        #     utils.show_figures((input[b, 0, :, :].numpy(), target[b,0,:,:].numpy(), weight_map[b, :, :]))

        if target.dim() == 4:
            target = target.squeeze(1)

        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)

        log_prob_maps = F.log_softmax(output, dim=1)
        if params.train['weight_map']:
            loss_map = criterion(log_prob_maps, target_var)
            loss_map *= weight_map_var
            loss_CE = loss_map.sum() / (loss_map.size(0) * loss_map.size(1) * loss_map.size(2))
        else:
            loss_CE = criterion(log_prob_maps, target_var)
        loss = loss_CE

        if params.train['beta'] != 0:
            prob_maps = F.softmax(output, dim=1)
            pred_map = torch.argmax(prob_maps, dim=1, keepdim=True)
            pred_map = (pred_map==4).repeat(1,3,1,1).float()    # only care about the contours
            target_map = (target_var==4).unsqueeze(1).repeat(1,3,1,1).float()
            pred_feat = vgg_model(pred_map)
            target_feat = vgg_model(target_map)
            loss_perceptual = criterion_perceptual(pred_feat, target_feat)
            loss = loss_CE + params.train['beta'] * loss_perceptual

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        iou = utils.accuracy(pred, target.numpy(), num_class=params.model['out_c']-1)
        if params.model['out_c'] == 5:
            iou_nuclei = utils.accuracy(np.uint8((pred > 0) * (pred < 4)),
                                        np.uint8((target.numpy() > 0) * (target.numpy() < 4)), num_class=2)
        else:
            iou_nuclei = utils.accuracy(np.uint8((pred == 1)), np.uint8((target.numpy() == 1)), num_class=2)

        if params.train['beta'] != 0:
            result = [loss.item(), loss_CE.item(), loss_perceptual.item(), iou_nuclei, iou]
        else:
            result = [loss.item(), loss_CE.item(), 0, iou_nuclei, iou]

        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input_var, output, target_var, log_prob_maps, loss

        if i % params.train['print_freq'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_Per {r[2]:.4f}'
                        '\tIoU-nuclei {r[3]:.4f}'
                        '\tIoU {r[4]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_Per {r[2]:.4f}'
                '\tIoU-nuclei {r[3]:.4f}'
                '\tIoU {r[4]:.4f}'.format(epoch, params.train['num_epochs'], r=results.avg))

    return results.avg


def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(5)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        if params.train['weight_map']:
            input, weight_map, target = sample
            weight_map = weight_map.float().div(20)
            if weight_map.dim() == 4:
                weight_map = weight_map.squeeze(1)
            weight_map_var = weight_map.cuda()
        else:
            input, target = sample

        # no classification
        if params.model['out_c'] == 3:
            target[target == 2] = 1
            target[target == 3] = 1
            target[target == 4] = 2

        # no edge or classification
        if params.model['out_c'] == 2:
            target[target > 0] = 1

        if target.dim() == 4:
            target = target.squeeze(1)

        target_var = target.cuda()

        size = params.train['input_size']
        overlap = params.train['val_overlap']
        output = utils.split_forward(model, input, size, overlap, params.model['out_c'])

        log_prob_maps = F.log_softmax(output, dim=1)
        if params.train['weight_map']:
            loss_map = criterion(log_prob_maps, target_var)
            loss_map *= weight_map_var
            loss_CE = loss_map.sum() / (loss_map.size(0) * loss_map.size(1) * loss_map.size(2))
        else:
            loss_CE = criterion(log_prob_maps, target_var)
        loss = loss_CE

        if params.train['beta'] != 0:
            prob_maps = F.softmax(output, dim=1)
            pred_map = torch.argmax(prob_maps, dim=1, keepdim=True)
            pred_map = pred_map.repeat(1, 3, 1, 1).float()
            target_map = target_var.unsqueeze(1).repeat(1, 3, 1, 1).float()
            pred_feat = vgg_model(pred_map)
            target_feat = vgg_model(target_map)
            loss_perceptual = criterion_perceptual(pred_feat, target_feat)
            loss = loss_CE + params.train['beta'] * loss_perceptual

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        iou = utils.accuracy(pred, target.numpy(), num_class=params.model['out_c'] - 1)
        if params.model['out_c'] == 5:
            iou_nuclei = utils.accuracy(np.uint8((pred > 0) * (pred < 4)),
                                        np.uint8((target.numpy() > 0) * (target.numpy() < 4)), num_class=2)
        else:
            iou_nuclei = utils.accuracy(np.uint8((pred == 1)), np.uint8((target.numpy() == 1)), num_class=2)

        if params.train['beta'] != 0:
            result = [loss.item(), loss_CE.item(), loss_perceptual.item(), iou_nuclei, iou]
        else:
            result = [loss.item(), loss_CE.item(), 0, iou_nuclei, iou]

        results.update(result, input.size(0))

        del output, target_var, log_prob_maps, loss

    logger.info('\t=> Val Avg:   Loss {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_Per {r[2]:.4f}'
                '\tIoU-nuclei {r[3]:.4f}'
                '\tIoU {r[4]:.4f}'.format(r=results.avg))

    return results.avg


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


def setup_logging(params):
    mode = 'a' if params.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(params.paths['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(params.paths['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(params.paths['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_Per\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


def write_txt(results, filename, mode='w'):
    """ Save the result of losses and iou scores for each epoch/iteration
        results: a list of numbers
    """
    with open(filename, mode) as file:
        num = len(results)
        for i in range(num-1):
            file.write('{:.4f}\t'.format(results[i]))
        file.write('{:.4f}\n'.format(results[num-1]))


if __name__ == '__main__':
    main()
