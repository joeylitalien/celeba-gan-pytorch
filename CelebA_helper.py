import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
root = "data/celebA_original/train/"
save_root = "data/celebA_all/train/"
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root):
    os.mkdir(save_root)
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    fname = save_root + img_list[i][:-3] + "png"
    plt.imsave(fname=fname, arr=img, format="png")

    if (i % 1000) == 0:
        print("%d images complete" % i)
