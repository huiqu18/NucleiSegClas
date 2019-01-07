"""
This script is used to test the trained model using test dataset and produce final
segmentation maps.

Author: Hui Qu
"""

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage import measure
from scipy import misc
from model import ResUNet34, UNet
import utils

from params import Params, get_transforms


def main():
    params = Params()
    img_dir = params.test['img_dir']
    label_dir = params.test['label_dir']
    save_dir = params.test['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_path = params.test['model_path']
    save_flag = params.test['save_flag']
    tta = params.test['tta']

    params.save_params('{:s}/test_params.txt'.format(params.test['save_dir']), test=True)

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False
    if eval_flag:
        test_results = dict()
        # recall, precision, F1, dice, iou, haus
        tumor_result = utils.AverageMeter(7)
        lym_result = utils.AverageMeter(7)
        stroma_result = utils.AverageMeter(7)
        all_result = utils.AverageMeter(7)
        conf_matrix = np.zeros((3, 3))

    # data transforms
    test_transform = get_transforms(params.transform['test'])

    model_name = params.model['name']
    if model_name == 'ResUNet34':
        model = ResUNet34(params.model['out_c'], fixed_feature=params.model['fix_params'])
    elif params.model['name'] == 'UNet':
        model = UNet(3, params.model['out_c'])
    else:
        raise NotImplementedError()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    # img_names = ['193-adca-5']
    # total_time = 0.0
    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        if eval_flag:
            label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
            gt = misc.imread(label_path)

        input = test_transform((img,))[0].unsqueeze(0)

        print('\tComputing output probability maps...')
        prob_maps = get_probmaps(input, model, params)
        if tta:
            img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_hf = test_transform((img_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_vf = test_transform((img_vf,))[0].unsqueeze(0)  # vertical flip input
            input_hvf = test_transform((img_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_hf = get_probmaps(input_hf, model, params)
            prob_maps_vf = get_probmaps(input_vf, model, params)
            prob_maps_hvf = get_probmaps(input_hvf, model, params)

            # re flip
            prob_maps_hf = np.flip(prob_maps_hf, 2)
            prob_maps_vf = np.flip(prob_maps_vf, 1)
            prob_maps_hvf = np.flip(np.flip(prob_maps_hvf, 1), 2)

            # rotation 90 and flips
            img_r90 = img.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_r90 = test_transform((img_r90,))[0].unsqueeze(0)
            input_r90_hf = test_transform((img_r90_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_r90_vf = test_transform((img_r90_vf,))[0].unsqueeze(0)  # vertical flip input
            input_r90_hvf = test_transform((img_r90_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_r90 = get_probmaps(input_r90, model, params)
            prob_maps_r90_hf = get_probmaps(input_r90_hf, model, params)
            prob_maps_r90_vf = get_probmaps(input_r90_vf, model, params)
            prob_maps_r90_hvf = get_probmaps(input_r90_hvf, model, params)

            # re flip
            prob_maps_r90 = np.rot90(prob_maps_r90, k=3, axes=(1, 2))
            prob_maps_r90_hf = np.rot90(np.flip(prob_maps_r90_hf, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf = np.rot90(np.flip(prob_maps_r90_vf, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf = np.rot90(np.flip(np.flip(prob_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

            # utils.show_figures((np.array(img), np.array(img_r90_hvf),
            #                     np.swapaxes(np.swapaxes(prob_maps_r90_hvf, 0, 1), 1, 2)))

            prob_maps = (prob_maps + prob_maps_hf + prob_maps_vf + prob_maps_hvf
                         + prob_maps_r90 + prob_maps_r90_hf + prob_maps_r90_vf + prob_maps_r90_hvf) / 8

        pred = np.argmax(prob_maps, axis=0)  # prediction
        pred_inside = pred.copy()
        pred_inside[pred == 4] = 0  # set contours to background
        pred_nuclei_inside_labeled = measure.label(pred_inside > 0)

        pred_tumor_inside = pred_inside == 1
        pred_lym_inside = pred_inside == 2
        pred_stroma_inside = pred_inside == 3
        pred_3types_inside = pred_tumor_inside + pred_lym_inside * 2 + pred_stroma_inside * 3

        # find the correct class for each segmented nucleus
        N_nuclei = len(np.unique(pred_nuclei_inside_labeled))
        N_class = len(np.unique(pred_3types_inside))
        intersection = np.histogram2d(pred_nuclei_inside_labeled.flatten(), pred_3types_inside.flatten(),
                                      bins=(N_nuclei, N_class))[0]
        classes = np.argmax(intersection, axis=1)
        tumor_nuclei_indices = np.nonzero(classes == 1)
        lym_nuclei_indices = np.nonzero(classes == 2)
        stroma_nuclei_indices = np.nonzero(classes == 3)

        # solve the problem of one nucleus assigned with different labels
        pred_tumor_inside = np.isin(pred_nuclei_inside_labeled, tumor_nuclei_indices)
        pred_lym_inside = np.isin(pred_nuclei_inside_labeled, lym_nuclei_indices)
        pred_stroma_inside = np.isin(pred_nuclei_inside_labeled, stroma_nuclei_indices)

        # remove small objects
        pred_tumor_inside = morph.remove_small_objects(pred_tumor_inside, params.post['min_area'])
        pred_lym_inside = morph.remove_small_objects(pred_lym_inside, params.post['min_area'])
        pred_stroma_inside = morph.remove_small_objects(pred_stroma_inside, params.post['min_area'])

        # connected component labeling
        pred_tumor_inside_labeled = measure.label(pred_tumor_inside)
        pred_lym_inside_labeled = measure.label(pred_lym_inside)
        pred_stroma_inside_labeled = measure.label(pred_stroma_inside)
        pred_all_inside_labeled = pred_tumor_inside_labeled * 3 \
                                  + (pred_lym_inside_labeled * 3 - 2) * (pred_lym_inside_labeled>0) \
                                  + (pred_stroma_inside_labeled * 3 - 1) * (pred_stroma_inside_labeled>0)

        # dilation
        pred_tumor_labeled = morph.dilation(pred_tumor_inside_labeled, selem=morph.selem.disk(params.post['radius']))
        pred_lym_labeled = morph.dilation(pred_lym_inside_labeled, selem=morph.selem.disk(params.post['radius']))
        pred_stroma_labeled = morph.dilation(pred_stroma_inside_labeled, selem=morph.selem.disk(params.post['radius']))
        pred_all_labeled = morph.dilation(pred_all_inside_labeled, selem=morph.selem.disk(params.post['radius']))

        # utils.show_figures([pred, pred2, pred_labeled])

        if eval_flag:
            print('\tComputing metrics...')
            gt_tumor = (gt % 3 == 0) * gt
            gt_lym = (gt % 3 == 1) * gt
            gt_stroma = (gt % 3 == 2) * gt

            tumor_detect_metrics = utils.accuracy_detection_clas(pred_tumor_labeled, gt_tumor, clas_flag=False)
            lym_detect_metrics = utils.accuracy_detection_clas(pred_lym_labeled, gt_lym, clas_flag=False)
            stroma_detect_metrics = utils.accuracy_detection_clas(pred_stroma_labeled, gt_stroma, clas_flag=False)
            all_detect_metrics = utils.accuracy_detection_clas(pred_all_labeled, gt, clas_flag=True)

            tumor_seg_metrics = utils.accuracy_object_level(pred_tumor_labeled, gt_tumor, hausdorff_flag=False)
            lym_seg_metrics = utils.accuracy_object_level(pred_lym_labeled, gt_lym, hausdorff_flag=False)
            stroma_seg_metrics = utils.accuracy_object_level(pred_stroma_labeled, gt_stroma, hausdorff_flag=False)
            all_seg_metrics = utils.accuracy_object_level(pred_all_labeled, gt, hausdorff_flag=True)

            tumor_metrics = [*tumor_detect_metrics[:-1], *tumor_seg_metrics]
            lym_metrics = [*lym_detect_metrics[:-1], *lym_seg_metrics]
            stroma_metrics = [*stroma_detect_metrics[:-1], *stroma_seg_metrics]
            all_metrics = [*all_detect_metrics[:-1], *all_seg_metrics]
            conf_matrix += np.array(all_detect_metrics[-1])

            # save result for each image
            test_results[name] = {
                'tumor': tumor_metrics,
                'lym': lym_metrics,
                'stroma': stroma_metrics,
                'all': all_metrics
            }

            # update the average result
            tumor_result.update(tumor_metrics)
            lym_result.update(lym_metrics)
            stroma_result.update(stroma_metrics)
            all_result.update(all_metrics)

        # save image
        if save_flag:
            print('\tSaving image results...')
            misc.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), pred.astype(np.uint8) * 50)
            misc.imsave('{:s}/{:s}_prob_tumor.png'.format(prob_maps_folder, name), prob_maps[1, :, :])
            misc.imsave('{:s}/{:s}_prob_lym.png'.format(prob_maps_folder, name), prob_maps[2, :, :])
            misc.imsave('{:s}/{:s}_prob_stroma.png'.format(prob_maps_folder, name), prob_maps[3, :, :])
            # np.save('{:s}/{:s}_prob.npy'.format(prob_maps_folder, name), prob_maps)
            # np.save('{:s}/{:s}_seg.npy'.format(seg_folder, name), pred_all_labeled)
            final_pred = Image.fromarray(pred_all_labeled.astype(np.uint16))
            final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored = np.zeros((ori_h, ori_w, 3))
            pred_colored_instance = np.zeros((ori_h, ori_w, 3))
            pred_colored[pred_tumor_labeled>0] = np.array([255, 0, 0])
            pred_colored[pred_lym_labeled>0] = np.array([0, 255, 0])
            pred_colored[pred_stroma_labeled>0] = np.array([0, 0, 255])
            filename = '{:s}/{:s}_seg_colored_3types.png'.format(seg_folder, name)
            misc.imsave(filename, pred_colored)
            for k in range(1, pred_all_labeled.max() + 1):
                pred_colored_instance[pred_all_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            misc.imsave(filename, pred_colored_instance)

            # img_overlaid = utils.overlay_edges(label_img, pred_labeled2, img)
            # filename = '{:s}/{:s}_comparison.png'.format(seg_folder, name)
            # misc.imsave(filename, img_overlaid)

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    # print('Time: {:4f}'.format(total_time/counter))

    print('=> Processed all {:d} images'.format(counter))
    if eval_flag:
        print('Average: clas_acc\trecall\tprecision\tF1\tdice\tiou\thausdorff\n'
              'tumor: {t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}, {t[3]:.4f}, {t[4]:.4f}, {t[5]:.4f}, {t[6]:.4f}\n'
              'lym: {l[0]:.4f}, {l[1]:.4f}, {l[2]:.4f}, {l[3]:.4f}, {l[4]:.4f}, {l[5]:.4f}, {l[6]:.4f}\n'
              'stroma: {s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}, {s[3]:.4f}, {s[4]:.4f}, {s[5]:.4f}, {s[6]:.4f}\n'
              'all: {a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}, {a[3]:.4f}, {a[4]:.4f}, {a[5]:.4f}, {a[6]:.4f}'
              .format(t=tumor_result.avg, l=lym_result.avg, s=stroma_result.avg, a=all_result.avg))

        header = ['clas_acc', 'recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
        save_results(header, tumor_result.avg, lym_result.avg, stroma_result.avg, all_result.avg,
                     test_results, conf_matrix, '{:s}/test_result.txt'.format(save_dir))


def get_probmaps(input, model, params):
    size = params.test['patch_size']
    overlap = params.test['overlap']

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, params.model['out_c'])
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


def save_results(header, tumor_result, lym_result, stroma_result, all_result,
                 test_results, conf_matrix, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(tumor_result)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average results:\n')
        file.write('Tumor nuclei:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(tumor_result[i]))
        file.write('{:.4f}\n'.format(tumor_result[N - 1]))
        file.write('Lym nuclei:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(lym_result[i]))
        file.write('{:.4f}\n'.format(lym_result[N - 1]))
        file.write('Stroma nuclei:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(stroma_result[i]))
        file.write('{:.4f}\n'.format(stroma_result[N - 1]))
        file.write('All nuclei:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(all_result[i]))
        file.write('{:.4f}\n'.format(all_result[N - 1]))
        file.write('\n')

        # confusion matrix
        file.write('Confusion matrix:\n')
        file.write('\t\t{m0[0]:d}\t{m0[1]:d}\t{m0[2]:d}\n'
                   '\t\t{m1[0]:d}\t{m1[1]:d}\t{m1[2]:d}\n'
                   '\t\t{m2[0]:d}\t{m2[1]:d}\t{m2[2]:d}\n'
                   .format(m0=conf_matrix[0, :].astype(np.int32),
                           m1=conf_matrix[1, :].astype(np.int32),
                           m2=conf_matrix[2, :].astype(np.int32)))

        # classification accuracy
        acc_tumor = conf_matrix[0, 0] / np.sum(conf_matrix[:, 0])
        acc_lym = conf_matrix[1, 1] / np.sum(conf_matrix[:, 1])
        acc_stroma = conf_matrix[2, 2] / np.sum(conf_matrix[:, 2])
        acc_all = (conf_matrix[0, 0] + conf_matrix[1, 1] + conf_matrix[2, 2]) / np.sum(conf_matrix)
        file.write('classification accuracy from conf matrix:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                   .format(acc_all*100, acc_tumor*100, acc_lym*100, acc_stroma*100))

        # results for each image
        for key, val_dict in sorted(test_results.items()):
            file.write('{:s}:\n'.format(key))
            for type, vals in val_dict.items():
                file.write('\t{:s}:'.format(type))
                for value in vals:
                    file.write('\t{:.4f}'.format(value))
                file.write('\n')


if __name__ == '__main__':
    main()
