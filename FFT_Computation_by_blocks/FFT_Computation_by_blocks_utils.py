import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

def visualize_signal_and_kernel(x, kernel, mu=0, sig=1, kernel_sig=10):
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), num='Signal and kernel display')
    axs[0].plot(x)
    axs[0].set_title(f'Random noise ($\mu={mu}$, $\sigma={sig}$)')
    axs[0].set_xlabel('n')
    axs[1].plot(kernel)
    axs[1].set_title(f'Gaussian kernel ($\sigma={kernel_sig}$)')
    axs[1].set_xlabel('n')
    plt.tight_layout()
    plt.show()

def create_signal_and_kernel(num_samples=3000, mu=0, sig=1, kernel_size=100, kernel_sig=10):
    x = np.random.normal(mu, sig, size=num_samples)
    kernel = signal.gaussian(kernel_size, kernel_sig)

    # Visualize the signal and kernel
    visualize_signal_and_kernel(x, kernel, mu=mu, sig=sig, kernel_sig=kernel_sig)

    return x, kernel

def convolution_filter(x, kernel):
    # np.convolve uses DFT filtering internally...
    # return np.convolve(x, kernel, mode=mode)

    # Manual convolution
    out = np.zeros(x.shape[0])
    for i in range(1, out.shape[0]+1):
        if i < kernel.shape[0]:
            out[i-1] = np.sum(kernel[i-1::-1] * x[:i])
        # Only used if the 'full' convolution is done
#        elif i > x.shape[0]:
#            out[i-1] = np.sum(kernel[-1:(i-x.shape[0])-1:-1] * x[i-kernel.shape[0]:])
        else:
            out[i-1] = np.sum(kernel[::-1] * x[i-kernel.shape[0]:i])
    return out

def zero_padding(x, kernel):
    x_padded = np.hstack([x, np.zeros(kernel.shape[0]-1)])
    kernel_padded = np.hstack([kernel, np.zeros(x.shape[0]-1)])    
    return x_padded, kernel_padded

def visualize_zero_padding(x, kernel):
    x_padded, kernel_padded = zero_padding(x, kernel)
    
    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(9,3), num='Zero padding visualization')
    axs[0].set_title('Padded random noise')
    axs[0].plot(x_padded)
    axs[0].set_xlabel('n')
    axs[1].set_title('Padded Gaussian kernel')
    axs[1].plot(kernel_padded)
    axs[1].set_xlabel('n')
    plt.tight_layout()
    plt.show()

def DFT_filtering(x, kernel):
    x_padded, kernel_padded = zero_padding(x, kernel)
    x_FT = np.fft.fft(x_padded)
    kernel_FT = np.fft.fft(kernel_padded)
    x_filtered_FT = np.fft.ifft(x_FT * kernel_FT).real
    # Return the filtered signal of the same length as the input signal
    return x_filtered_FT[:x.shape[0]]

def visualize_DFT_filtering(x_filtered_FT, x_filtered):
    plt.close('all')
    fig, axs = plt.subplots(2, 1, figsize=(9,4), num='DFT filtering visualization')
    axs[0].set_title('Traditional filtering')
    axs[0].plot(x_filtered)
    axs[0].set_xlabel('n')

    axs[1].set_title('DFT filtering')
    axs[1].plot(x_filtered_FT)
    axs[1].set_xlabel('n')
    plt.tight_layout()
    plt.show()

def overlap_add(x, kernel, n):
    M = kernel.shape[0]
    N = x.shape[0]/n
    # Check block sizes
    assert N == int(N), 'The length of x is not a multiple of n.'
    N = int(N)

    # Block separation and zero padding
    x_blocks = np.zeros((n, N + M - 1))
    for i in range(n):
        x_blocks[i, :N] = x[i * N:(i + 1) * N]

    # Zero padding kernel
    kernel_padded = np.hstack([kernel, np.zeros(N - 1)])

    # DFT filtering
    x_filtered = np.zeros(x_blocks.shape)

    kernel_FT = np.fft.fft(kernel_padded) 
    x_FT = np.fft.fft(x_blocks)
    x_filtered_blocks = np.fft.ifft(x_FT * kernel_FT.T).real

    x_filtered = np.zeros(x.shape[0] + M - 1)

    # Creation of the final signal by addition of the index-shifted blocks
    for i in range(n):
        x_filtered[i * N:(i+1) * N + M - 1] += x_filtered_blocks[i]

    return x_filtered_blocks, x_filtered

def visualize_overlap_add(x_filtered_blocks, x_filtered_overlap_add, x_filtered, M):
    indices = np.linspace(0, x_filtered_overlap_add.shape[0]-1, x_filtered_overlap_add.shape[0])
    n = x_filtered_blocks.shape[0]
    N = x_filtered_blocks.shape[1] - (M - 1)

    plt.close('all')
    fig, axs = plt.subplots(3, 1, figsize=(9,6), num='Overlap-add visualization')

    # Display the individual blocks and their overlap (with shifted indices)
    for i in range(n):
        if i == n-1:
            # Display the signal with the same length as the original signal
            axs[0].plot(indices[i * N:(i+1) * N], x_filtered_blocks[i, :-M+1], label=f'$x_{i}$')
        else:
            axs[0].plot(indices[i * N:(i+1) * N + M - 1], x_filtered_blocks[i], label=f'$x_{i}$')

    axs[0].set_title('DFT filtered blocks')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('n')

    # Display the final overlap add filtered signal
    axs[1].plot(x_filtered_overlap_add[:-M+1])
    axs[1].set_title('"Overlap add" filtered signal')
    axs[1].set_xlabel('n')

    # Display the conventionally filtered signal
    axs[2].plot(x_filtered)
    axs[2].set_title('Traditionally filtered signal')
    axs[2].set_xlabel('n')

    plt.tight_layout()
    plt.show()

def overlap_save(x, kernel, n):
    M = kernel.shape[0]
    N = x.shape[0]/n
    # Check block sizes
    assert N == int(N), 'The length of x is not a multiple of n.'
    N = int(N)

    # Create blocks
    x_blocks = np.zeros((n, N + M - 1))
    # Creation of extended blocks
    for i in range(x_blocks.shape[0]):
        x_blocks[i] = np.hstack([x_blocks[i - 1, -(M - 1):], x[i * N: (i + 1) * N]])

    kernel_padded = np.hstack([kernel, np.zeros(N - 1)])

    # DFT filtering
    kernel_FT = np.fft.fft(kernel_padded) 
    x_FT = np.fft.fft(x_blocks)
    x_filtered_blocks = np.fft.ifft(x_FT * kernel_FT.T).real

    # Combining the blocks to form the final filtered signal (remove the first M-1 values from all blocks).
    x_filtered_overlap_save = x_filtered_blocks[:, M - 1:]

    return x_filtered_blocks, x_filtered_overlap_save.ravel()

def visualize_overlap_save(x_filtered_blocks, x_filtered_overlap_save, x_filtered, M):
    indices = np.linspace(0, x_filtered_overlap_save.shape[0]-1, x_filtered_overlap_save.shape[0])
    n = x_filtered_blocks.shape[0]
    N = x_filtered_blocks.shape[1] - (M - 1)

    plt.close('all')
    fig, axs = plt.subplots(3, 1, figsize=(9,6), num='overlap-save visualization')

    # Display the individual blocks and their overlap (with shifted indices)
    for i in range(n):
        if i == 0:
            axs[0].plot(indices[:N], x_filtered_blocks[i, M-1:], label=f'$x_{i}$')
        # Only needed when doing the 'full' convolution
#        elif i == n - 1:
#            axs[0].plot(indices[i * N - (M - 1):], x_filtered_blocks[i, :2*(M-1)], label=f'$x_{i}$')
        else:
            axs[0].plot(indices[i * N - (M - 1):(i+1) * N], x_filtered_blocks[i], label=f'$x_{i}$')

    axs[0].set_title('DFT filtered blocks')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('n')

    # Display the final overlap add filtered signal
    axs[1].plot(x_filtered_overlap_save)
    axs[1].set_title('"Overlap save" filtered signal')
    axs[1].set_xlabel('n')

    # Display the conventionally filtered signal
    axs[2].plot(x_filtered)
    axs[2].set_title('Traditionally filtered signal')
    axs[2].set_xlabel('n')

    plt.tight_layout()
    plt.show()

def rms(x, y):
    return np.sqrt(np.mean((x - y)**2))

def calc_error(fun, x, kernel, n, x_filtered):
    return rms(fun(x, kernel, n)[1][:x_filtered.shape[0]], x_filtered)

def execution_time(fun, x, kernel, n=None, runs=1000):
    if n is not None:
        start_time = time.time()
        for i in range(runs):
            _ = fun(x, kernel, n)
        end_time = time.time()
    else:
        start_time = time.time()
        for i in range(runs):
            _ = fun(x, kernel)
        end_time = time.time()
    return (end_time - start_time) / runs * 1e3

def visualize_time_and_error(x, kernel, runs=1000, ns=np.linspace(2, 10, 5, dtype=int)):
    # Get time of DFT method
    DFT_time = execution_time(DFT_filtering, x, kernel, runs=runs)

    # Get time of convolution method
    convolution_time = execution_time(convolution_filter, x, kernel, runs=runs)

    x_filtered = convolution_filter(x, kernel)

    execution_time_overlap_add = []
    error_overlap_add = []
    execution_time_overlap_save = []
    error_overlap_save = []

    for n in ns:
        execution_time_overlap_add.append(execution_time(overlap_add, x, kernel, n, runs))
        error_overlap_add.append(calc_error(overlap_add, x, kernel, n, x_filtered))
        execution_time_overlap_save.append(execution_time(overlap_save, x, kernel, n, runs))
        error_overlap_save.append(calc_error(overlap_save, x, kernel, n, x_filtered))

    plt.close('all')
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), num='Time and error display')

    axs[0].semilogy(ns, execution_time_overlap_add, label='Overlap add')
    axs[0].semilogy(ns, execution_time_overlap_save, label='Overlap save')
    axs[0].semilogy([ns[0], ns[-1]], [DFT_time, DFT_time], '--', color='black', label='DFT Baseline')
    axs[0].semilogy([ns[0], ns[-1]], [convolution_time, convolution_time], '--', color='brown', label='Convolution Baseline')
    axs[0].set_title('Execution time')
    axs[0].set_xlabel('Number of blocks')
    axs[0].set_ylabel('Time [ms]')
    axs[0].set_xticks(ns)
    axs[0].legend()

    axs[1].plot(ns, error_overlap_add, label='Overlap add')
    axs[1].plot(ns, error_overlap_save, label='Overlap save')
    axs[1].set_title('Error')
    axs[1].set_xlabel('Number of blocks')
    axs[1].set_ylabel('RMS')
    axs[1].legend()

    plt.tight_layout()
    plt.show()