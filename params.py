import os
import numpy as np
import my_transforms as T
import collections


class Params:
    def __init__(self):
        # --- file io --- #
        self.paths = dict()
        self.paths['data_dir'] = './data_for_train'  # path to data
        self.paths['save_dir'] = './experiments'  # path to save results
        if not os.path.exists(self.paths['save_dir']):
            os.mkdir(self.paths['save_dir'])
        self.paths['img_dir'] = '{:s}/images'.format(self.paths['data_dir'])
        self.paths['label_dir'] = '{:s}/labels'.format(self.paths['data_dir'])
        self.paths['weight_map_dir'] = '{:s}/weight_maps'.format(self.paths['data_dir'])

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'ResUNet34'
        self.model['fix_params'] = False
        self.model['in_c'] = 3       # input channel
        self.model['out_c'] = 5     # output channel

        # --- training params --- #
        self.train = dict()
        self.train['input_size'] = 224   # input size of the image
        self.train['num_epochs'] = 300   # number of training iterations
        self.train['batch_size'] = 8    # batch size
        self.train['val_batch_size'] = 1
        self.train['val_overlap'] = 80   # overlap size of patches for validation
        self.train['lr'] = 0.0001         # initial learning rate
        self.train['momentum'] = 0.9     # momentum for SGD
        self.train['weight_decay'] = 1e-4  # weight decay
        self.train['print_freq'] = 30    # iterations to print training results
        self.train['workers'] = 1        # number of workers to load images
        self.train['gpu'] = [0, ]        # select gpu devices
        self.train['weight_map'] = True
        self.train['beta'] = 0.1           # weight for perceptual loss
        self.train['checkpoint_freq'] = 30   # epoch to save checkpoints
        # --- resume training --- #
        self.train['start_epoch'] = 0    # start epoch
        self.train['checkpoint'] = ''    # checkpoint to resume training or evaluation

        # --- data transform --- #
        self.transform = dict()
        self.transform['train'] = {
            'random_resize': [0.75, 1.5],
            'horizontal_flip': True,
            'vertical_flip': True,
            'random_affine': 0.3,
            'random_elastic_deform': [6, 15],
            'random_rotation': 90,
            'random_crop': self.train['input_size'],
            'label_binarization': 2,
            'to_tensor': True,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.paths['data_dir']))
        }
        self.transform['val'] = {
            'label_binarization': 2,
            'to_tensor': True,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.paths['data_dir']))
        }
        self.transform['test'] = {
            'to_tensor': True,
            'normalize': np.load('{:s}/mean_std.npy'.format(self.paths['data_dir']))
        }

        # --- test parameters --- #
        self.test = dict()
        self.test['epoch'] = 'best'
        self.test['img_dir'] = './data_for_train/images/test'
        self.test['label_dir'] = './data/labels_instance'
        self.test['tta'] = False
        self.test['save_flag'] = True
        self.test['patch_size'] = 224
        self.test['overlap'] = 80
        self.test['save_dir'] = './experiments/1/{:s}'.format(self.test['epoch'])
        self.test['model_path'] = './experiments/1/checkpoints/checkpoint_{:s}.pth.tar'.format(self.test['epoch'])

        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 20  # minimum area for an object
        self.post['radius'] = 1

    def save_params(self, filename, test=False):
        file = open(filename, 'w')
        groups = ['paths', 'model', 'train', 'transform'] if not test else ['test', 'post']

        file.write("# ----- Parameters ----- #")
        for group, params in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in params.items():
                    file.write("{:s}:\n".format(name))
                    for t_name, t_val in val.items():
                        file.write("\t{:s}: {:s}\n".format(t_name, repr(t_val)))
            else:
                for name, val in params.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()


def get_transforms(p):
    """ data transforms for train, validation and test
        p: transform dictionary
    """
    t_list = list()
    if 'random_resize' in p:
        t_list.append(T.RandomResize(p['random_resize'][0], p['random_resize'][1]))

    if 'scale' in p:
        t_list.append(T.Scale(p['scale']))

    if 'horizontal_flip' in p:
        t_list.append(T.RandomHorizontalFlip())

    if 'vertical_flip' in p:
        t_list.append(T.RandomVerticalFlip())

    if 'random_affine' in p:
        t_list.append(T.RandomAffine(p['random_affine']))

    if 'random_elastic_deform' in p:
        t_list.append(T.RandomElasticDeform(p['random_elastic_deform'][0],
                                            p['random_elastic_deform'][1]))

    if 'random_rotation' in p:
        t_list.append(T.RandomRotation(p['random_rotation']))

    if 'random_crop' in p:
        t_list.append(T.RandomCrop(p['random_crop']))

    if 'label_binarization' in p:
        if p['label_binarization'] == 1:
            t_list.append(T.LabelBinarization())
        else:
            t_list.append(T.LabelBinarization2())

    if 'to_tensor' in p:
        t_list.append(T.ToTensor())

    if 'normalize' in p:
        t_list.append(T.Normalize(mean=p['normalize'][0], std=p['normalize'][1]))

    return T.Compose(t_list)


# params = Params()
# params.save_params('temp.txt')
