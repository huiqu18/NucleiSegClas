"""
This script is used to prepare the dataset for training and testing.
- Step 1: create folders
- Step 2: move images and labels to each folder
- Step 3: compute mean and standard deviation of training images
- Step 4: compute class frequencies for training labels

Author: Hui Qu
"""


import os, glob, shutil
import numpy as np
from PIL import Image
from random import shuffle
from scipy import misc
import json
import glob


def main():
    data_dir = './data'
    train_data_dir = './data_for_train'

    img_dir = '{:s}/images'.format(data_dir)
    label_instance_dir = '{:s}/labels_instance'.format(data_dir)
    label_dir = '{:s}/labels'.format(data_dir)
    weightmap_dir = '{:s}/weight_maps'.format(data_dir)
    patch_folder = '{:s}/patches'.format(data_dir)

    create_folder(patch_folder)
    create_folder(train_data_dir)

    # ------ create label with contours from instance label
    create_labels_from_instance(label_instance_dir, label_dir)

    # ------ create weight maps from instance labels
    # use matlab code weight_map.m for parallel computing

    # ------ split large images into 250x250 patches
    print("Splitting large images into small patches...")
    split_patches(img_dir, '{:s}/images'.format(patch_folder))
    split_patches(label_dir, '{:s}/labels'.format(patch_folder), 'label_with_contours')
    split_patches(weightmap_dir, '{:s}/weight_maps'.format(patch_folder), 'weight')

    # ------ divide dataset into train, val and test sets
    organize_data_for_training(data_dir, train_data_dir)

    # ------ compute mean, std
    compute_mean_std(data_dir, train_data_dir)


def create_labels_from_instance(data_dir, save_dir):
    """ create labels from instance labels """
    from skimage import morphology

    create_folder(save_dir)

    print("Generating label with contours from instance label...")
    image_list = os.listdir(data_dir)
    for image_name in sorted(image_list):
        name = image_name.split('.')[0]
        if name[-5:] != 'label':
            continue

        image_path = os.path.join(data_dir, image_name)
        image = misc.imread(image_path)
        h, w = image.shape

        # extract edges
        id_max = np.max(image)
        contours = np.zeros((h, w), dtype=np.bool)
        for i in range(1, id_max+1):
            nucleus = image == i
            contours += morphology.dilation(nucleus) & (~morphology.erosion(nucleus))

        tumor_nuclei = (image % 3 == 0) * (image > 0)
        lym_nuclei = (image % 3 == 1) * (image > 0)
        stroma_nuclei = (image % 3 == 2) * (image > 0)

        label_with_contours = np.zeros((h, w, 3), np.uint8)
        label_with_contours[:, :, 0] = (tumor_nuclei + contours).astype(np.uint8) * 255
        label_with_contours[:, :, 1] = (lym_nuclei + contours).astype(np.uint8) * 255
        label_with_contours[:, :, 2] = (stroma_nuclei + contours).astype(np.uint8) * 255

        misc.imsave('{:s}/{:s}_with_contours.png'.format(save_dir, name), label_with_contours.astype(np.uint8))


def split_patches(data_dir, save_dir, postfix=None):
    import math
    """ split large image into small patches """
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if postfix and name[-len(postfix):] != postfix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = misc.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if postfix:
                misc.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(postfix)-1], k, postfix), seg_imgs[k])
            else:
                misc.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def organize_data_for_training(data_dir, train_data_dir):
    # --- Step 1: create folders --- #
    create_folder(train_data_dir)
    create_folder('{:s}/images'.format(train_data_dir))
    create_folder('{:s}/labels'.format(train_data_dir))
    create_folder('{:s}/weight_maps'.format(train_data_dir))
    create_folder('{:s}/images/train'.format(train_data_dir))
    create_folder('{:s}/images/val'.format(train_data_dir))
    create_folder('{:s}/images/test'.format(train_data_dir))
    create_folder('{:s}/labels/train'.format(train_data_dir))
    create_folder('{:s}/labels/val'.format(train_data_dir))
    create_folder('{:s}/weight_maps/train'.format(train_data_dir))
    create_folder('{:s}/weight_maps/val'.format(train_data_dir))

    # --- Step 2: move images and labels to each folder --- #
    print('Organizing data for training...')
    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # train
    for img_name in train_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/patches/images/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/images/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # labels
        for file in glob.glob('{:s}/patches/labels/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # weight maps
        for file in glob.glob('{:s}/patches/weight_maps/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/weight_maps/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
    # val
    for img_name in val_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/images/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/images/val/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # labels
        for file in glob.glob('{:s}/labels/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels/val/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # weight maps
        for file in glob.glob('{:s}/weight_maps/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/weight_maps/val/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
    # test
    for img_name in test_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/images/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/images/test/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)


def compute_mean_std(data_dir, train_data_dir):
    """ compute mean and standarad deviation of training images """
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels

    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    print('Computing the mean and standard deviation of training data...')

    for file_name in train_list:
        img_name = '{:s}/images/{:s}'.format(data_dir, file_name)
        img = misc.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3].astype(int)
        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel

    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    # normalization
    mean_values = mean_values / 255
    std_values = std_values / 255

    np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))
    np.savetxt('{:s}/mean_std.txt'.format(train_data_dir), np.array([mean_values, std_values]), '%.4f', '\t')


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


if __name__ == '__main__':
    main()
