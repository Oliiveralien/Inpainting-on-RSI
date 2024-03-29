import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
from base_dataset import BaseDataset
from torch.utils.data import Dataset, DataLoader
import pickle as pkl

ALLMASKTYPES = ['bbox', 'seg', 'random_bbox', 'random_free_form', 'val']

class InpaintDataset(BaseDataset):
    """
    Dataset for Inpainting task
    Params:
        img_flist_path(str): The file which contains img file path list (e.g. test.flist)
        mask_flist_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
        random_bbox_size(tuple): if use random bbox mask, it define the size of the mask (default:(32,32))
        random_bbox_margin(tuple): if use random bbox, it define the margin of the bbox which means the distance between the mask and the margin of the image
                                    (default:(64,64))
    Return:
        img, *mask
    """
    def __init__(self, img_flist_path, mask_flist_paths_dict,
                resize_shape=(256, 256), transforms_oprs=['random_crop', 'to_tensor'],
                random_bbox_size=(128, 128), random_bbox_margin=(32, 32),
                random_ff_setting={'img_shape':[256,256],'mv':5, 'ma':4.0, 'ml':40, 'mbw':10}, random_bbox_number=5):

        with open(img_flist_path, 'r') as f:
            self.img_paths = f.read().splitlines()

        self.mask_paths = {}
        for mask_type in mask_flist_paths_dict:
            # print(mask_type)
            assert mask_type in ALLMASKTYPES
            if 'random' in mask_type:
                self.mask_paths[mask_type] = ['' for i in self.img_paths]
                # print('wc: {}'.format(self.mask_paths['random_free_form']))
                # print(len(self.mask_paths[mask_type]))
            else:
                with open(mask_flist_paths_dict[mask_type]) as f:
                    self.mask_paths[mask_type] = f.read().splitlines()
                    # print('tr: {}'.format(self.mask_paths['val']))
                    # print(len(self.mask_paths[mask_type]))

        self.resize_shape = resize_shape
        self.random_bbox_size = random_bbox_size
        self.random_bbox_margin = random_bbox_margin
        self.random_ff_setting = random_ff_setting
        self.random_bbox_number = random_bbox_number
        self.transform_initialize(resize_shape, transforms_oprs)



    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # create the paths for images and masks

        img_path = self.img_paths[index]
        error = 1

        while not os.path.isfile(img_path) or error == 1:
            try:
                img = self.transforms_fun(self.read_img(img_path))
                error = 0
            except:
                index = np.random.randint(0, high=len(self))
                img_path = self.img_paths[index]
                error = 1
        mask_paths = {}
        for mask_type in self.mask_paths:
            # print(mask_type)
            # print(len(self))
            # print(self.mask_paths[mask_type])
            mask_paths[mask_type] = self.mask_paths[mask_type][index]

        img = self.transforms_fun(self.read_img(img_path))

        masks = {mask_type:255*self.transforms_fun(self.read_mask(mask_paths[mask_type], mask_type))[:1, :,:] for mask_type in mask_paths}
        return img*255, masks

    def read_img(self, path):
        """
        Read Image
        """
        img = Image.open(path).convert("RGB")
        return img


    def read_mask(self, path, mask_type):
        """
        Read Masks now only support bbox
        """
        if mask_type == 'random_bbox':
            bboxs = []
            for i in range(self.random_bbox_number):
                bbox = InpaintDataset.random_bbox(self.resize_shape, self.random_bbox_margin, self.random_bbox_size)
                bboxs.append(bbox)
        elif mask_type == 'random_free_form':
            mask = InpaintDataset.random_ff_mask(self.random_ff_setting)
            return Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8))
        elif 'val' in mask_type:
            mask = InpaintDataset.read_val_mask(path)
            return Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8))
        else:
            bbox = InpaintDataset.read_bbox(path)
            bboxs = [bbox]
        # print(bboxs, self.resize_shape)
        mask = InpaintDataset.bbox2mask(bboxs, self.resize_shape)
        # print('final', mask.shape)
        return Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8))

    @staticmethod
    def read_val_mask(path):
        """
        Read masks from val mask data
        """
        mask = pkl.load(open(path, 'rb'))
        return mask


    @staticmethod
    def read_bbox(path):
        """
        The general method for read bbox file by juding the file type
        Return:
            bbox:[y, x, height, width], shape: (height, width)
        """
        if filename[-3:] == 'pkl' and 'Human' in filename:
            return InpaintDataset.read_bbox_ch(filename)
        elif filename[-3:] == 'pkl' and 'COCO' in filename:
            return InpaintDataset.read_bbox_pkl(filename)
        else:
            return InpaintDataset.read_bbox_xml(path)

    @staticmethod
    def read_bbox_xml(path):
        """
        Read bbox for voc xml
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        with open(filename, 'r') as reader:
            xml = reader.read()
        soup = BeautifulSoup(xml, 'xml')
        size = {}
        for tag in soup.size:
            if tag.string != "\n":
                size[tag.name] = int(tag.string)
        objects = soup.find_all('object')
        bndboxs = []
        for obj in objects:
            bndbox = {}
            for tag in obj.bndbox:
                if tag.string != '\n':
                    bndbox[tag.name] = int(tag.string)

            bbox = [bndbox['ymin'], bndbox['xmin'], bndbox['ymax']-bndbox['ymin'], bndbox['xmax']-bndbox['xmin']]
            bndboxs.append(bbox)
        #print(bndboxs, size)
        return bndboxs, (size['height'], size['width'])

    @staticmethod
    def read_bbox_pkl(path):
        """
        Read bbox from coco pkl
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        aux_dict = pkl.load(open(path, 'rb'))
        bbox = aux_dict["bbox"]
        shape = aux_dict["shape"]
        #bbox = random.choice(bbox)
        #fbox = bbox['fbox']
        return [[int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]], (shape[1], shape[0])

    @staticmethod
    def read_bbox_ch(path):
        """
        Read bbox from crowd human pkl
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        aux_dict = pkl.load(open(path, 'rb'))
        bboxs = aux_dict["bbox"]
        bbox = random.choice(bboxs)
        extra = bbox['extra']
        shape = aux_dict["shape"]
        while 'ignore' in extra and extra['ignore'] == 1 and bbox['fbox'][0] < 0 and bbox['fbox'][1] < 0:
            bbox = random.choice(bboxs)
            extra = bbox['extra']
        fbox = bbox['fbox']
        return [[fbox[1],fbox[0],fbox[3],fbox[2]]], (shape[1], shape[0])

    @staticmethod
    def read_seg_img(path):
        pass

    @staticmethod
    def random_bbox(shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)

        """
        img_height = shape[0]
        img_width = shape[1]
        height, width = bbox_shape
        ver_margin, hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)

    @staticmethod
    def random_ff_mask(config):
        """Generate a random free form mask with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)
        """

        h,w = config['img_shape']
        mask = np.zeros((h,w))
        num_v = 12+np.random.randint(config['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(config['ma'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(config['ml'])
                brush_w = 10+np.random.randint(config['mbw'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return mask.reshape(mask.shape+(1,)).astype(np.float32)

    @staticmethod
    def bbox2mask(bboxs, shape):
        """Generate mask tensor from bbox.

        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

        Returns:
            tf.Tensor: output with shape [1, H, W, 1]

        """
        height, width = shape
        mask = np.zeros(( height, width), np.float32)
        #print(mask.shape)
        for bbox in bboxs:
            h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
            w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
            mask[bbox[0]+h:bbox[0]+bbox[2]-h,
                 bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
        #print("after", mask.shape)
        return mask.reshape(mask.shape+(1,)).astype(np.float32)