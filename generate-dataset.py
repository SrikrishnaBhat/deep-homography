from tempfile import TemporaryFile
from PIL import Image
import numpy as np
import glob
import os.path
import random as rd
import cv2
import matplotlib.pyplot as plt
from time import time

# Load a random image from the dataset
def load_random_image(path_source, size):
    img = rd.choice(glob.glob(os.path.join(path_source, '*.jpg')))
    img_grey = Image.open(img).convert('L').resize(size)
    img_data = np.asarray(img_grey)
    return img_data


def save_to_file(images, offsets, path_dest, index=0):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    outfile = open(os.path.join(path_dest, 'set-{}.npz'.format(index)), 'wb')#TemporaryFile(dir=path_dest, suffix='.npz')
    np.savez(outfile, images=images, offsets=offsets)
    outfile.close()


# Function to generate dataset
def generate_dataset(path_source, path_dest, rho, height, width, data, box):
    images = np.empty([box, box, 2], dtype=np.uint8)
    offsets = np.empty([8], dtype=np.int8)
    start_time = time()
    k = 0
    step_size = 1000
    print('Generating dataset')
    start_time = time()
    for i in range(0, data):
        if i%step_size == 0:
            print('Generated {} in {} seconds'.format(i, time() - start_time))
        img = load_random_image(path_source, [width, height]).astype(np.uint16)
        src = np.empty([4, 2], dtype=np.uint8)
        dst = np.zeros([4, 2])
        # Get upper left corner from the range rho<=x<=(width/2-1) and rho<=y<=(height/2 - 1) to avoid image excess
        src[0][0] = rd.randint(rho, int(width/2) - 1)
        src[0][1] = rd.randint(rho, int(height/3) - 1)
        # Upper right
        src[1][0] = src[0][0] + box
        src[1][1] = src[0][1]
        # Lower left
        src[2][0] = src[0][0]
        src[2][1] = src[0][1] + box
        # Lower right
        src[3][0] = src[1][0]
        src[3][1] = src[2][1]
        # Generate offsets:
        offset = np.empty(8, dtype=np.int8)
        for j in range(8):
            offset[j] = rd.randint(-rho, rho)
        # Destination points:
        dst[0][0] = src[0][0] + offset[0]
        dst[0][1] = src[0][1] + offset[1]
        # Upper right
        dst[1][0] = src[1][0] + offset[2]
        dst[1][1] = src[1][1] + offset[3]
        # Lower left
        dst[2][0] = src[2][0] + offset[4]
        dst[2][1] = src[2][1] + offset[5]
        # Lower right
        dst[3][0] = src[3][0] + offset[6]
        dst[3][1] = src[3][1] + offset[7]

        h, status = cv2.findHomography(src, dst)
        print(img.shape, h, (width, height), img.dtype)
        img_warped = np.asarray(cv2.warpPerspective(img, h, (width, height))).astype(np.uint8)
        x = int(src[0][0])
        y = int(src[0][1])
        images[:, :, 0] = img[y:y+box, x:x+box]
        images[:, :, 1] = img_warped[y:y+box, x:x+box]
        save_to_file(images, offset, path_dest, i)


# Group dataset
def group_dataset (path, new_path, box=128, size=64):
    group_images = np.empty([size, box, box, 2]).astype(np.uint8)
    group_offsets = np.empty([size, 8]).astype(np.int8)
    i = 0
    j = 0
    k = 0
    start_time = time()
    for npz in glob.glob(os.path.join(path, '*.npz')):
        archive = np.load(npz)
        group_images[i, :, :, :] = archive['images']
        group_offsets[i, :] = archive['offsets']
        i = i + 1
        j = j + 1
        if i % size == 0:
            print('Grouped {} in {} seconds'.format(j, time() - start_time))
            i = 0
            save_to_file(group_images, group_offsets, new_path, k)
            k += 1


# Generate dataset for training
train_data_path = 'train2014/'  # path to training dataset
train_size = 100000
train_box_size = 128
train_height = 240
train_width = 320
train_rho = 32
# print('generating training data')
# generate_dataset(train_data_path, 'train-data', train_rho, train_height, train_width, train_size, train_box_size)

# Generate dataset for validation
val_data_path = 'val2014/'  # path to validation dataset
val_size = 50000
# print('Generating val-data')
# generate_dataset(val_data_path, 'val-data', train_rho, train_height, train_width, val_size, train_box_size)

# Generate dataset for testing
# test_data_path = 'test2014/'  # path to testing dataset
# test_size = 5000
# test_box_size = 128
# test_height = 240
# test_width = 320
# test_rho = 32
# print('Generating test-data')
# generate_dataset(test_data_path, 'test-data', test_rho, test_height, test_width, test_size, test_box_size)

# Show sample image
# archive = np.load('train-data/tmp__0n7fza.npz')
# im = archive['images']
# im1 = im[:, :, 0]
# im2 = im[:, :, 1]
# plt.imshow(im1, cmap='gray')
# plt.figure()
# plt.imshow(im2, cmap='gray')
# plt.show()


# Group datasets into batch_sizes (default is 64)
#group_dataset('train-data-small', 'train-data-combined')
group_dataset('val-data', 'val-data-combined-full')
