import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, Subset
from tqdm import tqdm

dir_train_img = Path('/data/datasets/unet_data/data/train/imgs/')
dir_train_mask = Path('/data/datasets/unet_data/data/train/masks/')
dir_val_img = Path('/data/datasets/unet_data/data/val/imgs/')
dir_val_mask = Path('/data/datasets/unet_data/data/val/masks/')
dir_test_img = Path('/data/datasets/unet_data/data/test/imgs/')
dir_test_mask = Path('/data/datasets/unet_data/data/test/masks/')

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        image = (torch.load(filename).squeeze(0).numpy()*255).astype(np.uint8)
        return Image.fromarray(image)
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def datasets_definiton(name):
    if name == '1-tool':
        train_tools = ['pattern_01_2_lines_angle_1']
        test_tools = ['pattern_31_rod']
    elif name == 'train-tools':
        train_tools = [ 'pattern_01_2_lines_angle_1',
                        'pattern_03_2_lines_angle_3',
                        'pattern_04_3_lines_angle_1',
                        'pattern_05_3_lines_angle_2',
                        'pattern_06_5_lines_angle_1',
                        'pattern_07_curves_degree_30_radios_10',
                        'pattern_09_curves_degree_120_radios_10',
                        'pattern_10_curves_degree_150_radios_10',
                        'pattern_11_curves_degree_30_radios_20',
                        'pattern_12_curves_degree_45_radios_20',
                        'pattern_14_curves_degree_150_radios_20',
                        'pattern_15_circle',
                        'pattern_17_ellipse_2',
                        'pattern_18_hex_1',
                        'pattern_20_hex_3',
                        'pattern_31_rod']

        test_tools = ['pattern_02_2_lines_angle_2',
                      'pattern_05_3_lines_angle_2',
                      'pattern_08_curves_degree_45_radios_10',
                      'pattern_13_curves_degree_120_radios_20',
                      'pattern_16_ellipse_1',
                      'pattern_19_hex_2',
                      'test_obj_hex_small_peg_seen',
                      'test_obj_square_small_peg_seen',
                      'test_obj_tilted_square_small_peg_seen'
                    ]
                      
    elif name == 'train-test-tools':
        train_tools = [ 'pattern_01_2_lines_angle_1',
                        'pattern_02_2_lines_angle_2',
                        'pattern_03_2_lines_angle_3',
                        'pattern_04_3_lines_angle_1',
                        'pattern_05_3_lines_angle_2',
                        'pattern_06_5_lines_angle_1',
                        'pattern_07_curves_degree_30_radios_10',
                        'pattern_08_curves_degree_45_radios_10',
                        'pattern_09_curves_degree_120_radios_10',
                        'pattern_10_curves_degree_150_radios_10',
                        'pattern_11_curves_degree_30_radios_20',
                        'pattern_12_curves_degree_45_radios_20',
                        'pattern_13_curves_degree_120_radios_20',
                        'pattern_14_curves_degree_150_radios_20',
                        'pattern_15_circle',
                        'pattern_16_ellipse_1',
                        'pattern_17_ellipse_2',
                        'pattern_18_hex_1',
                        'pattern_19_hex_2',
                        'pattern_20_hex_3',
                        'pattern_31_rod']
        test_tools = ['test_obj_hex_small_peg_seen',
                      'test_obj_square_small_peg_seen',
                      'test_obj_tilted_square_small_peg_seen']
    elif name == 'debug':
        train_tools = ['pattern_01_2_lines_angle_1']
        test_tools = ['test_obj_hex_small_peg_seen',
                      'test_obj_square_small_peg_seen',
                      'test_obj_tilted_square_small_peg_seen']
    else:
        raise ValueError(f'Unknown dataset: {name})')
    
    return train_tools, test_tools

def full_dataset(dataset_name, device='cpu'):
    train_tools, test_tools = datasets_definiton(dataset_name)
    train_datasets = [ToolDataset(dir_train_img, dir_train_mask, tool_name, device=device) for tool_name in train_tools]
    val_datasets = [ToolDataset(dir_val_img, dir_val_mask, tool_name, device=device) for tool_name in train_tools]
    test_datasets = [ToolDataset(dir_test_img, dir_test_mask, tool_name, device=device) for tool_name in train_tools]
    test_unseen_datasets = [ToolDataset(dir_test_img, dir_test_mask, tool_name, device=device) for tool_name in test_tools]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    test_unseen_dataset = ConcatDataset(test_unseen_datasets)

    len_dataset = len(train_datasets[0])
    train_viz_samples = [Subset(train_datasets[i], range(0, len_dataset, len_dataset//6)) for i in range(0, len(train_datasets))]
    train_viz_samples = ConcatDataset(train_viz_samples)
    val_len_dataset = len(val_datasets[0])
    val_viz_samples = [Subset(val_datasets[i], range(0, val_len_dataset, val_len_dataset//6)) for i in range(0, len(val_datasets))]
    val_viz_samples = ConcatDataset(val_viz_samples)
    test_len_dataset = len(test_datasets[0])
    test_viz_samples = [Subset(test_datasets[i], range(0, test_len_dataset, test_len_dataset//6)) for i in range(0, len(test_datasets))]
    test_viz_samples = ConcatDataset(test_viz_samples)
    test_unseen_len_dataset = len(test_unseen_datasets[0])
    test_unseen_viz_samples = [Subset(test_unseen_datasets[i], range(0, test_unseen_len_dataset, test_unseen_len_dataset//6)) for i in range(0, len(test_unseen_datasets))]
    test_unseen_viz_samples = ConcatDataset(test_unseen_viz_samples)

    print('------------------------------------------------------')
    print(f'Dataset: {dataset_name}')
    print(f'Number of training tools: {len(train_tools)}')
    print(f'Number of test tools: {len(test_tools)}')
    print(f'Number of training examples: {len(train_dataset)}')
    print(f'Number of validation examples: {len(val_dataset)}')
    print(f'Number of test examples: {len(test_dataset)}')
    print(f'Number of test unseen examples: {len(test_unseen_dataset)}')
    print(f'Number of training examples per tool: {len(train_dataset)/len(train_tools)}')
    print(f'Number of validation examples per tool: {len(val_dataset)/len(train_tools)}')
    print(f'Number of test examples per tool: {len(test_dataset)/len(train_tools)}')
    print(f'Number of test unseen examples per tool: {len(test_unseen_dataset)/len(test_tools)}')
    print('------------------------------------------------------')
    return train_dataset, val_dataset, test_dataset, test_unseen_dataset, (train_viz_samples, val_viz_samples, test_viz_samples, test_unseen_viz_samples)

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', device: str = 'cpu'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # import pdb; pdb.set_trace()
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
        self.device = device

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        # import pdb; pdb.set_trace()
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
    # images = images.to(dtype=torch.float32, memory_format=torch.channels_last)
    #             true_masks = true_masks.to(dtype=torch.long)


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

class ToolDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, tool_name, scale=1, mask_suffix: str = '', device: str = 'cpu'):
        # super().__init__(images_dir, mask_dir, scale, mask_suffix='')
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        self.ids = [i for i in self.ids if tool_name in i]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        print(f'Creating dataset with {len(self.ids)} examples')
        # logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        self.mask_values = [0, 255]  #list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        self.device = device
        # logging.info(f'Unique mask values: {self.mask_values}')
        self.images = []
        self.masks = []

        for idx in tqdm(range(len(self.ids))):
            name = self.ids[idx]
            mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

            mask = load_image(mask_file[0])
            img = load_image(img_file[0])

            img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

            self.images.append(torch.as_tensor(img.copy()).float().contiguous().to(self.device))
            self.masks.append(torch.as_tensor(mask.copy()).long().contiguous().to(self.device))
        

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'mask': self.masks[idx]
        }
