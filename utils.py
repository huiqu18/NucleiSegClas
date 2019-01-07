"""
This script defines some functions that can be used in other scripts, such as computing accuracy
during training, validation and test phases.

Author: Hui Qu
"""

import numpy as np
import random
import torch
import math
import skimage.morphology as morph
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from sklearn.metrics import confusion_matrix
from skimage import measure
import skimage.morphology as ski_morph


def accuracy(pred, target, num_class=4):
    """ Computes the accuracy during training or validation phase """
    batch_size = target.shape[0]
    iou = 0.0

    for i in range(batch_size):
        iou += compute_iou(pred[i, :, :], target[i, :, :], num_class)

    return iou/batch_size


def compute_iou(pred, target, num_class):
    """ Compute the pixel-level iou score between
    predicted img and groundtruth target
    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    # show_figures((pred, target))

    iou = 0.0
    count = 0
    for i in range(1, num_class):
        pred_i = pred == i
        target_i = target == i
        if np.sum(target_i) > 0:
            tp = np.sum(pred_i * target_i)  # true postives
            fp = np.sum(pred_i * (1-target_i))  # false postives
            fn = np.sum((1-pred_i) * target_i)  # false negatives
            iou += float(tp) / (tp+fp+fn+1e-10)
            count += 1

    return iou / (count + 1e-8)


def accuracy_detection_clas(pred, gt, clas_flag=False):
    """ compute F1 score and/or classification accuracy """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = morph.label(pred)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt)
    Ng = len(np.unique(gt_labeled)) - 1

    # show_figures((pred_labeled, gt_labeled))

    TP = 0.0  # true positive
    FP = 0.0  # false positive
    clas_list = list()
    for i in range(1, Ns + 1):
        pred_i = np.where(pred_labeled == i, 1, 0)
        img_and = np.logical_and(gt_labeled, pred_i)

        # get intersection objects in target
        overlap_parts = img_and * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_i, overlap_parts))

        # no intersection object
        if obj_no.size == 0:
            FP += 1
            continue

        # find max overlap object
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[np.argmax(obj_areas)]  # ground truth object number

        gt_obj_area = np.sum(gt_labeled == gt_obj)  # ground truth object area
        overlap_area = np.sum(overlap_parts == gt_obj)

        if float(overlap_area) / gt_obj_area >= 0.5:
            TP += 1

            if clas_flag:
                pred_cla = np.unique(pred * pred_i)[1] % 3
                gt_cla = np.unique(gt * (gt_labeled==gt_obj))[1] % 3
                clas_list.append([pred_cla, gt_cla])
        else:
            FP += 1

    FN = Ng - TP  # false negative

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    if clas_flag:
        N = len(clas_list)
        clas_list = np.array(clas_list)
        correct = np.sum(clas_list[:,0] == clas_list[:,1])
        clas_acc = correct / N
        conf_mat = confusion_matrix(clas_list[:,1], clas_list[:,0])
    else:
        clas_acc = -1
        conf_mat = None

    return clas_acc, recall, precision, F1, conf_mat


def accuracy_object_level(pred, gt, hausdorff_flag=True):
    """ Compute the object-level metrics between predicted and
    groundtruth: dice, iou, hausdorff """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = morph.label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                pred_cand_indices = find_candidates(gt_i, pred_labeled)

                for j in pred_cand_indices:
                    pred_j = np.where(pred_labeled == j, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_i)
                gt_ind = np.argwhere(gt_i)
                haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        if hausdorff_flag:
            hausdorff_g += gamma_i * haus_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                gt_cand_indices = find_candidates(pred_j, gt_labeled)

                for i in gt_cand_indices:
                    gt_i = np.where(gt_labeled == i, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_j)
                haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        if hausdorff_flag:
            hausdorff_s += sigma_j * haus_j

    return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2


def find_candidates(obj_i, objects_labeled, radius=50):
    """
    find object indices in objects_labeled in a window centered at obj_i

    """
    if radius > 400:
        return np.array([])

    h, w = objects_labeled.shape
    x, y = center_of_mass(obj_i)
    x, y = int(x), int(y)
    r1 = x-radius if x-radius >= 0 else 0
    r2 = x+radius if x+radius <= h else h
    c1 = y-radius if y-radius >= 0 else 0
    c2 = y+radius if y+radius < w else w
    indices = np.unique(objects_labeled[r1:r2, c1:c2])
    indices = indices[indices != 0]

    if indices.size == 0:
        indices = find_candidates(obj_i, objects_labeled, 2*radius)

    return indices


def overlay_edges(gt, pred, ori_img):
    ''' overlay boundaries on image '''
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(ori_img, np.ndarray):
        ori_img = np.array(ori_img)

    # edges of groundtruth labels
    indices_gt = np.unique(gt)
    indices_gt = indices_gt[indices_gt!=0]
    edges_gt = np.zeros(gt.shape, np.bool)
    for i in indices_gt:
        target_i = gt==i
        edge_i = ski_morph.binary_dilation(target_i) - ski_morph.binary_erosion(target_i)
        edges_gt += edge_i

    # edges of prediction
    indices_pred = np.unique(pred)
    indices_pred = indices_pred[indices_pred != 0]
    edges_pred = np.zeros(pred.shape, np.bool)
    for i in indices_pred:
        pred_i = pred == i
        edge_i = ski_morph.binary_dilation(pred_i) - ski_morph.binary_erosion(pred_i)
        edges_pred += edge_i

    # show_figures((edges_pred, edges_gt))

    ori_img[edges_gt * edges_pred > 0, :] = np.array([0, 255, 0])  # correct
    ori_img[edges_gt * (1 - edges_pred) > 0, :] = np.array([255, 255, 0])  # true
    ori_img[(1 - edges_gt) * edges_pred > 0, :] = np.array([255, 0, 0])  # false pred
    return ori_img


def split_forward(model, input, size, overlap, outchannel=3):
    '''
    split the input image for forward process
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]

    output = output[:,:,:h0,:w0].cuda()

    return output


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i+1)
            plt.imshow(imgs[i])

    plt.show()


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

