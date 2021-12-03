import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt 

def spec(f_shift):
    f_sh = np.abs(f_shift)
    min_val = np.amin(f_sh)
    f_sh[f_sh == 0] = min_val
    s = 40*np.log10(f_sh)
    return s, min_val

def DFFTnp(img, f_name):
    f = np.fft.fft2(img)
    f_f_shift = np.fft.fftshift(f)
    magnitude_s, _min = spec(f_f_shift)
    for a in f_f_shift[0:337]: a[508:520] = _min
    for a in f_f_shift[345:]: a[508:520] = _min
    for a in f_f_shift[335:345]: a[0:508] = _min
    for a in f_f_shift[335:345]: a[515:] = _min
    res, empty = spec(f_f_shift)
    plt.subplot(141),plt.imshow(magnitude_s, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('magnitude_s '), plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(res, cmap = 'gray', vmin=0, vmax=255)
    res = np.real(np.fft.ifft2(np.fft.ifftshift(f_f_shift)))
    plt.subplot(143),plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('picture '+ f_name), plt.xticks([]), plt.yticks([])
    plt.subplot(144),plt.imshow(res, cmap = 'gray', vmin=0, vmax=255)
    plt.title('result'), plt.xticks([]), plt.yticks([])
    plt.show()

images = glob.glob('./' + '*.png')

for f_name in images:
    img = np.float32(cv.imread(f_name,0))
    DFFTnp(img, f_name)
