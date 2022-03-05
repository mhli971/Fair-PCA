import numpy as np
from tqdm import tqdm


def LFW_load_img():  # load in images only once
    # 13232 images starting from img0 to img13231.txt
    data_size = 13232
    img_size = 1764

    # we also have white, black, asian, we will generate these altogether

    images = np.zeros((data_size, img_size))

    print('Begin loading images')
    for i in tqdm(range(data_size)):
        img_tmp = np.loadtxt(f'data/images/img{i}.txt').reshape(1, 1764)
        images[i, :] = img_tmp

    # normalization
    images = images / 255

    return images


def LFWProcess(images, identifier):  # sex, asian, black, white
    identifier = np.loadtxt(f'data/images/{identifier}.txt')

    # center the images
    images_centered = images - np.mean(images, axis=0, keepdims=True)

    images_A = images[identifier == 0]
    images_B = images[identifier == 1]
    assert len(images_A) + len(images_B) == len(images)

    images_A = images_A - np.mean(images_A, axis=0, keepdims=True)
    images_B = images_B - np.mean(images_B, axis=0, keepdims=True)

    M = images_centered
    A = images_A
    B = images_B

    return M, A, B
