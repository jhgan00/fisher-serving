import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import random

class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', coco_file_name=None, transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        coco_path = os.path.join(self.root_dir, set, coco_file_name)

        self.coco = COCO(os.path.join(coco_path))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        # #Debug
        # for i in range(10):
        #     idx = i
        #     img = self.load_image(idx)
        #     annot = self.load_annotations(idx)
        #
        #     for j in range(annot.shape[0]):
        #         x0 = int(annot[j, 0] + 0.5)
        #         y0 = int(annot[j, 1] + 0.5)
        #         x1 = int(annot[j, 2] + 0.5)
        #         y1 = int(annot[j, 3] + 0.5)
        #         cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
        #
        #         x0_0 = int(annot[j, 5] * (x1 - x0) + x0 + 0.5)
        #         y0_0 = int(annot[j, 6] * (y1 - y0) + y0 + 0.5)
        #         x1_0 = int(annot[j, 7] * (x1 - x0) + x0 + 0.5)
        #         y1_0 = int(annot[j, 8] * (y1 - y0) + y0 + 0.5)
        #
        #         x0_1 = int(annot[j, 9] * (x1 - x0) + x0 + 0.5)
        #         y0_1 = int(annot[j, 10] * (y1 - y0) + y0 + 0.5)
        #         x1_1 = int(annot[j, 11] * (x1 - x0) + x0 + 0.5)
        #         y1_1 = int(annot[j, 12] * (y1 - y0) + y0 + 0.5)
        #
        #         cv2.line(img, (x0_0, y0_0), (x1_0, y1_0), (255, 0, 255), 3)
        #         cv2.line(img, (x0_1, y0_1), (x1_1, y1_1), (255, 255, 0), 3)
        #
        #     img = cv2.resize(img, dsize=(int(img.shape[1] / 2 + 0.5), int(img.shape[0] / 2+ 0.5)), interpolation=cv2.INTER_AREA)
        #
        #     cv2.imshow('test', img)
        #     cv2.waitKey(0)

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]

        file_name = image_info['file_name']
        if file_name[:4] == 'swim' or file_name[:4] == 'fish':
            if file_name[5:7] == 'rb':
                file_name = 'rb/' + file_name
            elif file_name[5:7] == 'bp':
                file_name = 'bp/' + file_name
            elif file_name[5:7] == 'kr':
                file_name = 'kr/' + file_name
            elif file_name[5:7] == 'of':
                file_name = 'of/' + file_name
            elif file_name[5:7] == 'rs':
                file_name = 'rs/' + file_name
        #
        # if file_name[:6] == './objt':
        #     file_name = 'images/' + file_name[2:]

        # file_name = os.path.join(file_name.split('/')[1], file_name.split('/')[2])

        path = os.path.join(self.root_dir, self.set_name, file_name)
        img = cv2.imread(path)
        if img is None:
            print('empty')
            print(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #return img.astype(np.float32) / 255.
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 13))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 13))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotation[0, 5:13] = a['keypoints']
            #annotation[0, 9:13] = a['keypoints2']

            # current annotation[0, :4] is [x, y, w, h]
            annotation[0, 5] = (annotation[0, 5] - annotation[0, 0]) / (float)(annotation[0, 2])
            annotation[0, 7] = (annotation[0, 7] - annotation[0, 0]) / (float)(annotation[0, 2])
            annotation[0, 9] = (annotation[0, 9] - annotation[0, 0]) / (float)(annotation[0, 2])
            annotation[0, 11] = (annotation[0, 11] - annotation[0, 0]) / (float)(annotation[0, 2])

            annotation[0, 6] = (annotation[0, 6] - annotation[0, 1]) / (float)(annotation[0, 3])
            annotation[0, 8] = (annotation[0, 8] - annotation[0, 1]) / (float)(annotation[0, 3])
            annotation[0, 10] = (annotation[0, 10] - annotation[0, 1]) / (float)(annotation[0, 3])
            annotation[0, 12] = (annotation[0, 12] - annotation[0, 1]) / (float)(annotation[0, 3])

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 13)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 13)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class RandomFlipX(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            x0_0 = annots[:, 5].copy()
            x0_1 = annots[:, 7].copy()
            x1_0 = annots[:, 9].copy()
            x1_1 = annots[:, 11].copy()

            annots[:, 5] = 1 - x0_0
            annots[:, 7] = 1 - x0_1
            annots[:, 9] = 1 - x1_0
            annots[:, 11] = 1 - x1_1

            sample = {'img': image, 'annot': annots}

        return sample



class RandomFlipY(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_y=0.1):
        if np.random.rand() < flip_y:
            image, annots = sample['img'], sample['annot']
            image = image[::-1, :, :]

            rows, cols, channels = image.shape

            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()

            y_tmp = y1.copy()

            annots[:, 1] = rows - y2
            annots[:, 3] = rows - y_tmp

            y0_0 = annots[:, 6].copy()
            y0_1 = annots[:, 8].copy()
            y1_0 = annots[:, 10].copy()
            y1_1 = annots[:, 12].copy()

            annots[:, 6] = 1 - y0_0
            annots[:, 8] = 1 - y0_1
            annots[:, 10] = 1 - y1_0
            annots[:, 12] = 1 - y1_1

            sample = {'img': image, 'annot': annots}

        return sample

class GaussianBlur(object):

    def __init__(self, kernel_size=9):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        kernel_size = random.randrange(self.kernel_size)
        if kernel_size % 2 ==0:
            kernel_size = kernel_size + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.5)

        return {'img': image, 'annot': annots}


class RandomHSV(object):

    def __init__(self, d_h=5, d_s=10, d_v=10):
        self.d_h = d_h
        self.d_s = d_s
        self.d_v = d_v

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        #h, s, v = cv2.split(hsv)

        h_p = random.randrange(0, 100)
        if h_p < 20:
            d_h = random.randrange(-self.d_h, self.d_h)
            hsv[:, :, 0] = hsv[:, :, 0] + d_h
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)

        h_s = random.randrange(0, 100)
        if h_s < 20:
            d_s = random.randrange(-self.d_s, self.d_s)
            hsv[:, :, 1] = hsv[:, :, 1] + d_s
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        h_v = random.randrange(0, 100)
        if h_v < 20:
            d_h = random.randrange(-self.d_v, self.d_v)
            hsv[:, :, 2] = hsv[:, :, 2] + d_h
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        #final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return {'img': image, 'annot': annots}


class RandomNoise(object):

    def __init__(self, weight = 50):
        self.weight = weight
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        h, w, c = image.shape
        noise = np.random.randint(0, 50, (h, w))  # design jitter/noise here
        zitter = np.zeros_like(image)
        zitter[:, :, 1] = noise

        image = cv2.add(image, zitter)

        return {'img': image, 'annot': annots}



class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32)/255. - self.mean) / self.std), 'annot': annots}
