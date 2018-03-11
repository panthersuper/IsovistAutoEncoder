        
import os
import numpy as np
np.random.seed(123)
import math
import json
from fnmatch import fnmatch
import scipy.io as spio
import sys

# img_root = './test_data'

# count = 0

# list_im = []
# for path, subdirs, files in os.walk(img_root):
#     for name in files:
#         if fnmatch(name, '*.json'):
#             img_dir = os.path.join(img_root,name)

#             with open(img_dir, 'r') as f:
#                 #thisdata = np.array(json.load(f))

#                 # image = json.load(f)

#                 # image = np.array(image, np.object)

#                 # sys.exit(0)
#                 # os.remove(img_dir)
#                 count +=1

#                 print(count)

from PIL import Image
import numpy as np

img_dir = "./data/[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2]_7719.0.json"
with open(img_dir, 'r') as f:
    #thisdata = np.array(json.load(f))

    image = json.load(f)

    image = np.array(image)
    # print(image)

    # # image = np.reshape(image,(30,60))
    image = image/10000*255

    print(image)
    print(np.shape(image))
    im = Image.fromarray(image)
    im.show()
