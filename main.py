import numpy as np
from scipy import stats
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def rgb2gray(img):
    return np.dot(img[:255, :3], [0.299, 0.587, 0.114])


def main():
    sea_img = mpimg.imread('sea.jpg')
    lavanda_img = mpimg.imread('lavanda.jpg')
    sea_gray_img = rgb2gray(sea_img)
    lavanda_gray_img = rgb2gray(lavanda_img)
    #print(sea_img)
    #plt.imshow(sea_gray_img)
    #plt.show()
    plt.imshow(sea_gray_img, cmap=plt.get_cmap('gray'))
    plt.show()
    plt.imshow(lavanda_gray_img, cmap=plt.get_cmap('gray'))
    plt.show()
    sea_hist = plt.hist(sea_gray_img.ravel(), bins=25, label="sea histogram")
    plt.show()
    lavanda_hist = plt.hist(lavanda_gray_img.ravel(), bins=25, label="lavanda histogram")
    plt.show()
    print(f'Sea hist mean {np.mean(sea_hist[0])}')
    print(f'Lavanda hist mean {np.mean(lavanda_hist[0])}')
    print(f'Sea hist std {np.std(sea_hist[0])}')
    print(f'Lavanda hist std {np.std(lavanda_hist[0])}')
    print(f'Sea hist mode {max(set(sea_hist[0]), key=list(sea_hist[0]).count)}')
    print(f'Lavanda hist mode {max(set(lavanda_hist[0]), key=list(lavanda_hist[0]).count)}')
    print(f'Sea hist median {np.median(sea_hist[0])}')
    print(f'Lavanda hist median {np.median(lavanda_hist[0])}')
    print(f'Coorcoef of hist {np.corrcoef(sea_hist[0], lavanda_hist[0])[0][1]}')
    print(f'Coorcoef of images {np.corrcoef(np.array([sea_gray_img.flatten(), lavanda_gray_img.flatten()]))[0][1]}')
    if stats.chisquare(sea_hist[0])[1]:
        print('the null hypothesis rejected for first image')
    else:
        print('the null hypothesis is true for first image')
    if stats.chisquare(lavanda_hist[0])[1]:
        print('the null hypothesis rejected for second image')
    else:
        print('the null hypothesis is true for second image')

main()
