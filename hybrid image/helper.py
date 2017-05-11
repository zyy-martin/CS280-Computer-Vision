import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

# ------------------------------ useful functions ------------------------------- #
def gaussian_kernel(sigma, width = 5):
    
    kernel_val = np.zeros((width, width))
    center = (width - 1)/2
    for i in range(width):
        for j in range(width):
            dist_square = (center - i) ** 2 + (center - j) ** 2
            kernel_val[i, j] = np.exp(-dist_square/2.0/sigma**2)
    return kernel_val/np.sum(kernel_val)


def padding(img, width = 5):
    
    pad_len = int((width - 1)/2)
    return np.pad(img, [(pad_len, pad_len), (pad_len, pad_len), (0,0)], 'constant')


def convolution(kernel, patch):
    return np.sum(kernel * patch)


def Filter(img, sigma, width, type = "Gaussian"):
    
    if type == "Gaussian":
        d1, d2, d3 = img.shape
        new_img = np.zeros((d1, d2, d3))
        center = (width - 1)/2
        pad_img = padding(img, width)
        kernel = gaussian_kernel(sigma, width)

        for k in range(d3):
            for i in range(d1):
                for j in range(d2):
                    new_img[i, j, k] = convolution(kernel, pad_img[i:2*center+i+1, j:2*center+j+1, k])
        return new_img
    elif type == "Laplacian":
        return img - Filter(img, sigma, width, "Gaussian")


def normalize(ref_img, tgt_img):

    d3 = tgt_img.shape[2]
    new_img = np.zeros(tgt_img.shape)

    for k in range(d3):
        ref_min, ref_max = np.min(ref_img[:,:,k]), np.max(ref_img[:,:,k])
        tgt_min, tgt_max = np.min(tgt_img[:,:,k]), np.max(tgt_img[:,:,k])
        new_img[:,:,k] = ref_min + float(ref_max - ref_min)/float(tgt_max - tgt_min) * (tgt_img[:,:,k] - tgt_min)

    return new_img


def sharpen(img, sigma, width = 5, level = 0):
    return normalize(img, img + level * Filter(img, sigma, width, "Laplacian"))


def rgb_to_gray(img):
    return 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]

def crop(img1, img2):
    col_index = np.sum(img1[:,:,0], 0) * np.sum(img1[:,:,0], 0) != 0
    row_index = np.sum(img2[:,:,0], 1) * np.sum(img2[:,:,0], 1) != 0
    new_img1 = img1[row_index,:,:][:,col_index,:]
    new_img2 = img2[row_index,:,:][:,col_index,:]
    return new_img1, new_img2
# --------------------------- functions for hybrid image ------------------------ #

def low_pass_filter(img, sigma, width):
    return Filter(img, sigma, width, "Gaussian")


def high_pass_filter(img, sigma, width):
    return Filter(img, sigma, width, "Laplacian")


def hybrid_image(img1, img2, sigma1, sigma2, width, alpha):
    return (1 - alpha) * low_pass_filter(img1, sigma1, width) + alpha * high_pass_filter(img2, sigma2, width)


def resample(img, x_dist, y_dist):
    d1 = img.shape[0]
    d2 = img.shape[1]
    x_index = np.array(range(d1))
    y_index = np.array(range(d2))
    x_index = x_index[x_index%x_dist==0]
    y_index = y_index[y_index%y_dist==0]
    return img[x_index, :, :][:, y_index, :]


def pyramid(img, N, width, sigma, dist, filtering = low_pass_filter):
    img_pyramid = [img]
    new_img = np.copy(img)
    for k in range(N):
        new_img = resample(filtering(new_img, sigma, width), dist, dist)
        img_pyramid.append(new_img)
    return img_pyramid
