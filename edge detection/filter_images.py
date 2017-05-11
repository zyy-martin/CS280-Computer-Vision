import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt


def gaussian_kernel(sigma, width=5):
    kernel_val = np.zeros((width, width))
    center = (width - 1) / 2
    for i in range(width):
        for j in range(width):
            dist_square = (center - i) ** 2 + (center - j) ** 2
            kernel_val[i, j] = np.exp(-dist_square / 2.0 / sigma ** 2)
    return kernel_val / np.sum(kernel_val)

def elongated_gaussian_kernel(sigma_x, sigma_y, width=5):
    kernel_val = np.zeros((width, width))
    center = (width - 1) / 2
    for i in range(width):
        for j in range(width):
            x2= (center - i) ** 2
            y2 = (center - j) ** 2
            kernel_val[i, j] = np.exp(-x2 / 2.0 / sigma_x ** 2  -y2 / 2.0 / sigma_y ** 2)
    return kernel_val / np.sum(kernel_val)

def get_direction(theta):
    colors = [[244,66,66], [244,149,65],[244,211,65],[208,244,65], [106,244,65], [65,244,172], [65,178,244], [65,100,244], [121,65,244], [208,65,244],[244,65,178], [244,65,112]]
    degree = theta / 3.1415926 * 180
    res = np.zeros((theta.shape[0],theta.shape[1],3))
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if np.isnan(degree[i,j]):
                res[i,j] = np.array([0,0,0])
            else:
                res[i,j] = colors[int(degree[i,j] / 15)]
    return res

def plot_pie():
    labels = ['[$0,\pi/6$]','[$\pi/6,\pi/3$]','[$\pi/3,\pi/2$]','[$\pi/2,2\pi/3$]','[$2\pi/3,5\pi/6$]','[$5\pi/6,\pi$]','[$\pi,7\pi/6$]','[$7\pi/6,4\pi/3$]','[$4\pi/3,3\pi/2$]','[$3\pi/2,5\pi/3$]','[$5\pi/3,11\pi/6$]','[$11\pi/6,2\pi$]']
    sizes = [1,1,1,1,1,1,1,1,1,1,1,1]
    colors = ['#f44242', '#f49541','#f4d341','#d0f441', '#6af441', '#41f4ac', '#41b2f4', '#4164f4', '#7941f4', '#d041f4','#f441b2', '#f44170']
    plt.pie(sizes, labels=labels, colors=colors,startangle=0,counterclock=True)
    plt.show()


def difference_filter(img, threshold = 0.05):
    length, width, dummy = img.shape

    # diff filter
    Dx = np.array([1,-1]).reshape((1,2))
    Dy = np.array([[1],[-1]]).reshape((2,1))

    # gradient
    gradient_x = np.zeros((length,width,3))
    gradient_y = np.zeros((length,width,3))

    # convolve each channel in x, y directions
    for i in range(3):
        gradient_x[:, :, i] = signal.convolve2d(img[:, :, i], Dx, mode='same')
        gradient_y[:, :, i] = signal.convolve2d(img[:, :, i], Dy, mode='same')

    # initialize gradient norm
    gradient_norm = np.zeros((length,width))

    # if norm(x,y) < threshold, mark(x,y) = -1
    mark = np.ones((length,width)) * -1
    norm = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            for k in range(3):
                norm_k = np.sqrt(gradient_x[i,j,k]**2 +  gradient_y[i,j,k]**2)
                norm[i,j] += gradient_x[i,j,k]**2 + gradient_y[i,j,k]**2
                if norm_k > gradient_norm[i,j] and norm_k > threshold:
                    mark[i,j] = k
                    gradient_norm[i, j] = norm_k
            norm[i,j] = np.sqrt(norm[i,j])
    # initialize theta
    theta = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            if mark[i,j] < 0:
                theta[i,j] = None
            else:
                theta[i,j] = np.arctan(gradient_y[i,j,mark[i,j]]/gradient_x[i,j,mark[i,j]])

    return norm, theta


def derivative_gaussian_filter(img, sigma, threshold=0.05, size = 5):
    (length, width, dummy) = img.shape
    # diff filter
    Dx = np.array([1,-1]).reshape((1,2))
    Dy = np.array([[1],[-1]]).reshape((2,1))

    # gaussian filter
    gaussian = gaussian_kernel(sigma,size)
    # derivative of gaussian
    derivative_x = signal.convolve2d(gaussian, Dx, mode='same')
    derivative_y = signal.convolve2d(gaussian, Dy, mode='same')
    gradient_x = np.zeros((length,width,3))
    gradient_y = np.zeros((length,width,3))

    # convolve each channel in x, y directions
    for i in range(3):
        gradient_x[:, :, i] = signal.convolve2d(img[:, :, i], derivative_x, mode='same')
        gradient_y[:, :, i] = signal.convolve2d(img[:, :, i], derivative_y, mode='same')

    # initialize gradient norm
    gradient_norm = np.zeros((length,width))

    # if norm(x,y) < threshold, mark(x,y) = -1
    mark = np.ones((length,width)) * -1
    norm = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            for k in range(3):
                norm_k = np.sqrt(gradient_x[i,j,k]**2 +  gradient_y[i,j,k]**2)
                norm[i, j] += gradient_x[i, j, k] ** 2 + gradient_y[i, j, k] ** 2
                if norm_k > gradient_norm[i,j] and norm_k > threshold:
                    mark[i,j] = k
                    gradient_norm[i, j] = norm_k
            norm[i, j] = np.sqrt(norm[i, j])
    # initialize theta
    theta = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            if mark[i,j] < 0:
                theta[i,j] = None
            else:
                theta[i,j] = np.arctan(gradient_y[i,j,mark[i,j]]/gradient_x[i,j,mark[i,j]])
    return norm, theta


def oriented_filter(img, sigma_x, sigma_y, angle, threshold=0.05, size = 5):
    (length, width, dummy) = img.shape

    # diff filter
    Dx = np.array([1, -1]).reshape((1, 2))
    Dy = np.array([[1], [-1]]).reshape((2, 1))

    # elongated gaussian filter
    gaussian = elongated_gaussian_kernel(sigma_x,sigma_y, size)

    # derivative of gaussian
    derivative_x = signal.convolve2d(gaussian, Dx, mode='same')
    derivative_y = signal.convolve2d(gaussian, Dy, mode='same')

    # rotate
    derivative_x=ndimage.interpolation.rotate(derivative_x,angle,reshape=False)
    derivative_y=ndimage.interpolation.rotate(derivative_y,angle,reshape=False)

    # initialize gradient
    gradient_x = np.zeros((length, width, 3))
    gradient_y = np.zeros((length, width, 3))

    # convolve each channel in x, y directions
    for i in range(3):
        gradient_x[:, :, i] = signal.convolve2d(img[:, :, i], derivative_x, mode='same')
        gradient_y[:, :, i] = signal.convolve2d(img[:, :, i], derivative_y, mode='same')
    # initialize gradient norm
    gradient_norm = np.zeros((length, width))

    # if norm(x,y) < threshold, mark(x,y) = -1
    mark = np.ones((length, width)) * -1
    norm = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            for k in range(3):
                norm_k = np.sqrt(gradient_x[i, j, k] ** 2 + gradient_y[i, j, k] ** 2)
                norm[i, j] += gradient_x[i, j, k] ** 2 + gradient_y[i, j, k] ** 2
                if norm_k > gradient_norm[i, j] and norm_k > threshold:
                    gradient_norm[i, j] = norm_k
                    mark[i, j] = k
            norm[i, j] = np.sqrt(norm[i, j])
    # initialize theta
    theta = np.zeros((length, width))
    for i in range(length):
        for j in range(width):
            if mark[i, j] < 0:
                theta[i, j] = None
            else:
                theta[i, j] = np.arctan(gradient_y[i, j, mark[i, j]] / gradient_x[i, j, mark[i, j]])
    return norm, theta

def plot_filter(angle, sigma_x, sigma_y, width, sub1, sub2):
    filter = elongated_gaussian_kernel(sigma_x, sigma_y, width)
    Dx = np.array([1, -1]).reshape((1, 2))
    Dy = np.array([[1], [-1]]).reshape((2, 1))
    derivative_x = signal.convolve2d(filter, Dx, mode='same')
    derivative_y = signal.convolve2d(filter, Dy, mode='same')
    derivative_x=ndimage.interpolation.rotate(derivative_x,angle,reshape=False)
    derivative_y=ndimage.interpolation.rotate(derivative_y,angle,reshape=False)
    plt.subplot(sub1)
    plt.imshow(derivative_x,cmap='gray')
    plt.title('angle = '+str(angle)+', x direction')
    plt.subplot(sub2)
    plt.imshow(derivative_y,cmap='gray')
    plt.title('angle = '+str(angle)+', y direction')



