import os
import numpy as np
np.random.seed(123)
import math
import json

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.randomize = kwargs['randomize']
        self.img_root = os.path.join(kwargs['img_root'])
        self.file_lst = kwargs['file_lst']

        self.dir_lst = [os.path.join(self.img_root,str(i)+".json") for i in self.file_lst]

        # 'img_root': 'data/',   # MODIFY PATH ACCORDINGLY
        # 'file_lst': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],   # MODIFY PATH ACCORDINGLY
        # 'randomize': True

        # read data info from lists
        self.list_im = []
        self.list_lab = []

        for i,filedir in enumerate(self.dir_lst):
            lab = self.file_lst[i]

            with open(filedir, 'r') as f:
                #thisdata = np.array(json.load(f))
                thisd = json.load(f)
                self.list_im.extend(thisd)
                self.list_lab.extend([lab for i in range(len(thisd))])
    
        self.list_im = np.array(self.list_im, np.object)
        self.list_im = self.list_im/10000

        self.list_lab = np.array(self.list_lab, np.int64)


        self.num = self.list_im.shape[0]

        # permutation
        perm = np.random.permutation(self.num)
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]
        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.list_im.shape[1], self.list_im.shape[2]))
        labels_batch = np.zeros(batch_size)

        for i in range(batch_size):
            images_batch[i, ...] = self.list_im[self._idx]
            labels_batch[i, ...] = self.list_lab[self._idx]

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0

        return images_batch, labels_batch

    def size(self):
        return self.num

    def reset(self):
        self._idx = 0