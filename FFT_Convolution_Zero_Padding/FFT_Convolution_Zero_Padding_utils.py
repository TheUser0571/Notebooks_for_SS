import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import cv2 as cv
from interactive_kit import imviewer as viewer

def load_image(path='images/pixelart.png', display=False):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    if display:
        # Display image
        plt.close('all')
        view = viewer(img, title=path.split('/')[-1], subplots=(1,1))
    
    return img

def get_gaussian_kernel(size, sig, display=False):
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
    dst = np.sqrt(x*x+y*y)

    # Calculating Gaussian array
    gauss = np.exp(-(dst**2) / (2.0 * sig**2))
    
    if display:
        # Display kernel
        plt.close('all')
        view = viewer(gauss, subplots=(1,1))
    
    return gauss

def convolve_2d(img, kernel, mode='same', boundary='fill', fillvalue=0, display=False):
    img_convolved = signal.convolve2d(img, kernel, mode=mode, boundary=boundary, fillvalue=fillvalue)
    
    if display:
        plt.close('all')
        view = viewer([img, img_convolved], subplots=(1,2))
    
    return img_convolved

def zero_padding(x, kernel, mode='same'):
    if mode == 'same':
        kernel_padded = np.zeros(x.shape)
        kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
        return x, kernel_padded
    elif mode == 'full':
        kernel_padded = np.zeros((x.shape[0] + kernel.shape[0] - 1, x.shape[1] + kernel.shape[1] - 1))
        kernel_padded[:kernel.shape[0], :kernel.shape[1]] = kernel
                                 
        x_padded = np.zeros((x.shape[0] + kernel.shape[0] - 1, x.shape[1] + kernel.shape[1] - 1))
        x_padded[:x.shape[0], :x.shape[1]] = x
        return x_padded, kernel_padded
    else:
        raise ValueError('zero_padding: mode should be "same" or "full"')

def FFT_filtering(x, kernel, mode='same'):
    x_padded, kernel_padded = zero_padding(x, kernel, mode=mode)
    x_FT = np.fft.fft2(x_padded)
    kernel_FT = np.fft.fft2(kernel_padded)
    x_filtered_FT = np.fft.ifft2(x_FT * kernel_FT).real
    if mode == 'full':
        return x_filtered_FT[kernel.shape[0]//2-1:-kernel.shape[0]//2, kernel.shape[1]//2-1:-kernel.shape[1]//2]
    return x_filtered_FT

def visual_comparison(img, kernel):
    img_convolved = convolve_2d(img, kernel)
    img_fft_filtered = FFT_filtering(img, kernel)
    img_fft_filtered_padded = FFT_filtering(img, kernel, mode='full')
    plt.close('all')
    titles = ['Original', 'FFT filtered (no padding)', 'Convolution', 'FFT filtered (zero padding)']
    view = viewer([img, img_fft_filtered, img_convolved, img_fft_filtered_padded], title=titles, subplots=(2,2))
    
def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))

def calc_error(img, kernel):
    img_convolved = convolve_2d(img, kernel)
    img_fft_filtered = FFT_filtering(img, kernel)
    img_fft_filtered_padded = FFT_filtering(img, kernel, mode='full')
    print(f'The RMS error between the convolution and padded FFT filter is {rmse(img_fft_filtered_padded, img_convolved):.3e}')
    print(f'The RMS error between the convolution and non-padded FFT filter is {rmse(img_fft_filtered, img_convolved):.3f}')
    
# Timing comparison
def time_comparison(img, kernel, runs=10):
    convolution_time = 0
    for i in range(runs):
        start_time = time.time()
        img_convolved = convolve_2d(img, kernel, mode='same')
        end_time = time.time()
        convolution_time += end_time - start_time
    convolution_time = convolution_time * 1e3 / runs

    fft_time = 0
    for i in range(runs):
        start_time = time.time()
        img_fft = FFT_filtering(img, kernel, mode='full')
        end_time = time.time()
        fft_time += end_time - start_time
    fft_time = fft_time * 1e3 / runs
    return np.round(convolution_time), np.round(fft_time)

def display_time_comparison(img, runs=2):
    sizes = np.linspace(3, 30, 10, dtype=int)
    
    conv_times = []
    fft_times = []
    for size in sizes:
        c_t, f_t = time_comparison(img, get_gaussian_kernel(size=size, sig=10), runs=runs)
        conv_times.append(c_t)
        fft_times.append(f_t)
    
    plt.close('all')
    plt.figure()
    plt.title('Time comparison between convolution and FFT filtering')
    plt.plot(sizes, conv_times, label='Convolution')
    plt.plot(sizes, fft_times, label='FFT')
    plt.legend()
    plt.xlabel('Kernel size')
    plt.ylabel('Runtime [ms]')
    plt.xticks(sizes)
    plt.show()