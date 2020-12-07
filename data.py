import numpy as np
from skimage.transform import resize

# read yuv file
def read_yuv(path, wid, hei, frame):
    f = open(path, 'rb')
    y = np.zeros((frame, hei, wid))
    for i in range(frame):
        for j in range(hei):
            for k in range(wid):
                y[i, j, k] = ord(f.read(1)) / 255.0
        skip = f.read(int(0.5 * wid * hei))
    f.close()
    return y

# crop data
def crop(im, wid, hei, frame):
    result = np.zeros((frame, int((wid / 32) * (hei / 32)), 32, 32))
    for i in range(frame):
        for j in range(0, hei-31, 32):
            for k in range(0, wid-31, 32):
                result[i, int(j / 32) * int(wid / 32) + int(k / 32), :, :] = im[i, j:j+32, k:k+32]
    return result

# generate training data
def gen_data(dat):
    result = np.zeros((dat.shape[0], 2 * dat.shape[1], 2 * dat.shape[2]))
    for i in range(dat.shape[0]):
        result[i, :, :] = resize(dat[i, :, :], (2 * dat.shape[1], 2 * dat.shape[2]), order = 3)
    return result

# reconstruct data
def crop_inv(crop_data, wid, hei):
    result = np.zeros((hei, wid))
    for i in range(0, hei-31, 32):
        for j in range(0, wid-31, 32):
            result[i:i+32, j:j+32] = crop_data[int(i/32) * int(wid/32) + int(j/32), :, :, 0]
    return result
