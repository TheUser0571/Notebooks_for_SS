import matplotlib.pyplot as plt
import numpy as np
import IPython
from scipy import signal
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets

class Signal_undersampling():
    def __init__(self, T=2, a=2):
        self.out = Output(layout={'width': '980px', 'height': '400px'})
        self.axs = []
        
        self.samples = 81
        self.T = T
        self.t = np.linspace(-10, 10, self.samples)
        self.a = a
        self.s = np.sinc(self.t/self.a)**2
        self.s_FT = np.abs(np.fft.fftshift(np.fft.fft(self.s)))
        self.w = np.linspace(-np.pi, np.pi, len(self.s_FT))
        
        self.undersample_signal()
        
        self.init_figure()
        
        self.T_slider = widgets.IntSlider(value=self.T, min=1, max=20, step=1, description='T')
        self.T_slider.observe(self.on_T_change, names='value')
        
        self.samples_slider = widgets.IntSlider(value=self.samples, min=21, max=501, step=20, description='Samples')
        self.samples_slider.observe(self.on_samples_change, names='value')
        
        self.a_slider = widgets.FloatSlider(value=self.a, min=1, max=5, step=0.1, description='a')
        self.a_slider.observe(self.on_a_change, names='value')
        
        display(VBox([self.out, HBox([self.T_slider, self.samples_slider, self.a_slider])]))
        
    
    def init_figure(self):
        with self.out:
            self.fig = plt.figure('Signal undersampling demo',figsize=(8.5, 3.5))
            self.gs = self.fig.add_gridspec(2, 2)

            # Plot the signals
            self.axs.append(self.fig.add_subplot(self.gs[:, 0]))
            self.axs[0].set_title("$f(t)$")
            self.axs[0].plot(self.t, self.s, color='blue', label='$f(t)$')
            self.axs[0].plot(self.t, self.s_rec, '--', color='black', label='$\widetilde{f}(t)$')
            self.ml, self.sl, self.bl = self.axs[0].stem(self.t[::self.T], self.s_T[::self.T], linefmt='r', markerfmt='ro', label='$f_T(t)$')
            self.bl.set_visible(False)
            self.axs[0].set_xlabel('t')
            self.axs[0].legend()
            
            # Plot the FT
            self.axs.append(self.fig.add_subplot(self.gs[0, 1]))
            self.axs[1].set_title("$F(\omega)$")
            self.axs[1].plot(self.w, self.s_FT, color='blue')
            self.axs[1].set_xlabel('$\omega$ [rad]')
            positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            labels = ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$']
            self.axs[1].set_xticks(positions)
            self.axs[1].set_xticklabels(labels)
            
            # Plot the FT
            self.axs.append(self.fig.add_subplot(self.gs[1, 1]))
            self.axs[2].set_title("$F_T(\omega)$")
            self.axs[2].plot(self.w_T, np.abs(self.s_FT_T), color='black')
            # Recovery rectangle
            plt.fill([self.w0, self.w0, self.w1, self.w1], [0, np.max(np.abs(self.s_FT_T)), np.max(np.abs(self.s_FT_T)), 0], color='lightgreen', alpha=0.5)
            self.axs[2].set_xlabel('$\omega$ [rad]')
            positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            labels = ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$']
            self.axs[2].set_xticks(positions)
            self.axs[2].set_xticklabels(labels)
#             self.axs[2].set_ylim(self.axs[1].get_ylim())
            
            plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)
            plt.show()
    
    def sinc_interp(self, x, s, u):
        """
        Interpolates x, sampled at "s" instants
        Output y is sampled at "u" instants ("u" for "upsampled")     
        """

        if len(x) != len(s):
            raise Exception('x and s must be the same length')

        # Find the period    
        T = s[1] - s[0]

        sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
        y = np.dot(x, np.sinc(sincM/T))
        return y
    
    def undersample_signal(self):
        # Undersample
        self.s_T = np.zeros(len(self.s))
        self.s_T[::self.T] = self.s[::self.T]
        self.s_FT_T = np.fft.fftshift(np.fft.fft(self.s_T))
        self.w_T = np.linspace(-np.pi, np.pi, len(self.s_FT_T))
        # Apply lowpass filter
#         rect = np.zeros(len(self.s_FT_T))
#         rect[len(rect)//2 - int(len(rect)/self.T/2):len(rect)//2 + int(len(rect)/self.T/2) + 1] = self.T
#         self.s_rec_FT = self.s_FT_T * rect
#         self.s_rec = np.fft.ifft(np.fft.ifftshift(self.s_rec_FT)).real
        
        # Interpolate signal
        self.s_rec = self.sinc_interp(self.s_T[self.s_T!=0], self.t[::self.T], self.t)
        
        # Calculate lowpass filter frequencies
        self.w0 = self.w_T[len(self.w_T)//2 - int(len(self.w_T)/self.T/2)]
        self.w1 = self.w_T[len(self.w_T)//2 + int(len(self.w_T)/self.T/2)]
        
    def update_signal(self):
        self.t = np.linspace(-10, 10, self.samples)
        self.s = np.sinc(self.t/self.a)**2
        self.s_FT = np.abs(np.fft.fftshift(np.fft.fft(self.s)))
        self.w = np.linspace(-np.pi, np.pi, len(self.s_FT))
        self.axs[0].lines[0].set_data(self.t, self.s)
        self.axs[1].lines[0].set_data(self.w, self.s_FT)
        self.axs[1].set_ylim([-0.05*np.max(np.abs(self.s_FT)), np.max(np.abs(self.s_FT)) + 0.05*np.max(np.abs(self.s_FT))])

    def on_T_change(self, value):
        self.T = value['new']
        self.undersample_signal()
        self.update_undersampled_signal()
        
    def update_undersampled_signal(self):
        # Update the stem plot
        # Update markers
        self.ml.set_data(self.t[::self.T], self.s_T[::self.T])
        # Update lines
        x, y = self.ml.get_data()
        segments = [np.array([[x[i], 0], [x[i], y[i]]]) for i in range(len(x))]
        self.sl.set_segments(segments)
        
        # Update the FT plot
        self.axs[2].lines[0].set_data(self.w_T, np.abs(self.s_FT_T))
        self.axs[2].set_ylim([-0.05*np.max(np.abs(self.s_FT_T)), np.max(np.abs(self.s_FT_T)) + 0.05*np.max(np.abs(self.s_FT_T))])
        plt.fill([-np.pi, -np.pi, np.pi, np.pi], [-100, 100, 100, -100], color='white')
        plt.fill([self.w0, self.w0, self.w1, self.w1], [0, np.max(np.abs(self.s_FT_T)), np.max(np.abs(self.s_FT_T)), 0], color='lightgreen', alpha=0.5)
        
        # Update the reconstructed signal plot
        self.axs[0].lines[1].set_data(self.t, self.s_rec)
    
    def on_samples_change(self, value):
        self.samples = value['new']
        self.update_signal()
        self.undersample_signal()
        self.update_undersampled_signal()
    
    def on_a_change(self, value):
        self.a = value['new']
        self.update_signal()
        self.undersample_signal()
        self.update_undersampled_signal()