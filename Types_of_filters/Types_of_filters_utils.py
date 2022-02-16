import matplotlib.pyplot as plt
import numpy as np
import IPython
from scipy import signal
from scipy.io import wavfile
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets

class Filter_Demo():
    def __init__(self, filename, filter_method='butter'):
        self.out = Output(layout={'width': '980px', 'height': '400px'})
        self.axs = []
        self.fill_color = 'lightgreen'
        self.filter_method = filter_method
        
        # Read the audio signal from file
        self.SF, self.s = wavfile.read(filename)
        self.t = np.linspace(0, len(self.s)/self.SF, len(self.s))
        
        # Generate Fourier Transform of audio signal
        s_FT = np.abs(np.fft.fftshift(np.fft.fft(self.s)))
        self.s_FT = s_FT / s_FT.max()
        self.w_FT = np.linspace(-self.SF//2, self.SF//2, len(self.s))
        
        # Filter types
        self.filter_types = {'lowpass':0, 'highpass':1, 'bandpass':2, 'bandstop':3, 'notch':4}
        
        self.f_crit = self.SF//6
        self.filter = list(self.filter_types.keys())[0]
        self.filter_idx = self.filter_types[self.filter]
        self.s_filtered = None
        self.h = None
        self.w = None

        # Compute the initial filter
        self.update_filter(init=True)
        
        # Inizializate the figure
        self.init_figure()
        
        # Add audio players
        self.play_orig = Output(layout={'width': '320px', 'height': '60px'})
        self.play_filt = Output(layout={'width': '320px', 'height': '60px'})
        self.play_orig.append_display_data(IPython.display.Audio(self.s, rate=self.SF))
        self.play_filt.append_display_data(IPython.display.Audio(self.s_filtered, rate=self.SF))

        # Descriptive text
        self.text_orig = widgets.HTML(value="<h3>Original</h3>")
        self.text_filt = widgets.HTML(value="<h3>Filtered</h3>")

        # Add frequency sliders
        self.f0_slider = widgets.IntSlider(value=self.SF//6, min=50, max=self.SF//2-300, description='$\;f_0\,[Hz]$:',
                                          continuous_update=False, style={'description_width':'initial'})
        self.f0_slider.observe(self.f0_callback, names='value')
        self.f1_slider = widgets.IntSlider(value=2*self.SF//6, min=50, max=self.SF//2-300, description='$\;f_1\,[Hz]$:', 
                                           continuous_update=False, disabled=True, style={'description_width':'initial'})
        self.f1_slider.observe(self.f1_callback, names='value')
        
        self.apply_button = widgets.Button(description='Apply filter', layout=Layout(width='95%'))
        self.apply_button.on_click(self.apply_filter_callback)
        
        # Add dropdown menu for filter type
        self.filter_menu = widgets.Dropdown(options=self.filter_types.keys(), value=self.filter, 
                                            description='Filter type:', layout=Layout(width='max-content'))
        self.filter_menu.observe(self.menu_callback, names='value')
        
        # Add dropdown menu for filter method
        self.method_menu = widgets.Dropdown(options=['butter', 'ellip', 'cheby'], value=self.filter_method, 
                                            description='Filter method:', layout=Layout(width='max-content'))
        self.method_menu.observe(self.method_callback, names='value')
        
        display(VBox([self.out, HBox([VBox([self.filter_menu,  self.method_menu]), VBox([self.f0_slider, self.f1_slider, self.apply_button]), 
                                      VBox([HBox([self.text_orig, self.play_orig]), HBox([self.text_filt, self.play_filt])])])]))
        plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)
        
        self.apply_filter_callback()
        
    
    def init_figure(self):
        with self.out:
            self.fig = plt.figure('Types of filters demo', figsize=(8.5, 3.5))
            self.gs = self.fig.add_gridspec(2, 2)

            # Plot the FT
            self.axs.append(self.fig.add_subplot(self.gs[:, 0]))
            self.axs[0].set_title("Filter and signal spectrum modulus")
            self.axs[0].plot(self.w_FT, self.s_FT, color='blue', linewidth=0.2)
            self.axs[0].plot(self.w, self.h, color=self.fill_color, linewidth=0.7)
            self.axs[0].set_xlabel('f [Hz]')
            self.axs[0].fill(self.w, self.h, facecolor=self.fill_color)
            self.axs[0].legend(['Signal', 'Filter'], loc='upper right')
            
            # Plot the original waveform
            self.axs.append(self.fig.add_subplot(self.gs[0, -1]))
            self.axs[1].set_title('Original signal')
            self.axs[1].plot(self.t, self.s, color='blue', linewidth=0.2)
            self.axs[1].set_xlabel('t [s]')
            self.axs[1].set_xlim([np.min(self.t), np.max(self.t)])
            self.axs[1].set_ylim([np.min(self.s), np.max(self.s)])
            self.axs[1].get_yaxis().set_visible(False)
            
            # Plot the filtered waveform
            self.axs.append(self.fig.add_subplot(self.gs[1, -1]))
            self.axs[2].set_title("Filtered signal")
            self.axs[2].plot(self.t, self.s_filtered, color='blue', linewidth=0.2)
            self.axs[2].set_xlabel('t [s]')
            self.axs[2].set_xlim([np.min(self.t), np.max(self.t)])
            self.axs[2].set_ylim([np.min(self.s), np.max(self.s)])
            self.axs[2].get_yaxis().set_visible(False)
            plt.show()

    def update_filter(self, init=False):
        # Ensure that the lower frequency is first
        f_crit = np.sort(self.f_crit) if np.ndim(self.f_crit) > 0 else self.f_crit
        # Constructing the filter
        if self.filter == 'notch':
            b, a = signal.iirnotch(w0=f_crit, Q=30, fs=self.SF)
        else:
            if self.filter_method == 'ellip':
                # Elliptic
                b, a = signal.ellip(N=5, rp=0.01, rs=100, Wn=f_crit, btype=self.filter, fs=self.SF)
            elif self.filter_method == 'cheby':
                # Chebychev
                b, a = signal.cheby1(N=5, rp=0.01, Wn=f_crit, btype=self.filter, fs=self.SF)
            else:
                # Butterworth
                b, a = signal.butter(N=5, Wn=f_crit, btype=self.filter, fs=self.SF)
        # Frequency response
        w, h = signal.freqz(b, a, whole=True, fs=self.SF)
        self.h = np.abs(np.fft.fftshift(h))
        self.w = w - self.SF//2
        # Filtering
        self.s_filtered = signal.lfilter(b, a, self.s)
        if not init:
            self.axs[0].lines[1].set_data(self.w, self.h)
            x_lim = self.axs[0].get_xlim()
            y_lim = self.axs[0].get_ylim()
            # Clear the fill by over-filling with white
            self.axs[0].fill([-self.SF, -self.SF+1, self.SF-1, self.SF], [-1, 2, 2, -1], facecolor='white')
            # Create new fill
            if self.filter_idx % 2 == 1 or self.filter_idx == 4:
                self.axs[0].fill(self.w, np.concatenate([[0], self.h[1:-1], [0]]), facecolor=self.fill_color)
            else:
                self.axs[0].fill(self.w, self.h, facecolor=self.fill_color)
            self.axs[0].set_xlim(x_lim)
            self.axs[0].set_ylim(y_lim)

    def f0_callback(self, value):
        if self.filter_idx < 2 or self.filter_idx == 4:
            self.f_crit = value['new']
        else:
            self.f_crit[0] = value['new']
        self.update_filter()
    def f1_callback(self, value):
        if self.filter_idx > 1 and self.filter_idx != 4:
            self.f_crit[1] = value['new']
        self.update_filter()
        
    def apply_filter_callback(self, value=None):
        self.axs[2].lines[0].set_data(self.t, self.s_filtered)
        self.play_filt.clear_output()
        self.play_filt.append_display_data(IPython.display.Audio(self.s_filtered, rate=self.SF))
        
    def menu_callback(self, value):
        self.filter = value['new']
        self.filter_idx = self.filter_types[self.filter]
        if self.filter_idx < 2 or self.filter_idx == 4:
            self.f1_slider.disabled = True
            self.f_crit = self.f0_slider.value
        else:
            self.f1_slider.disabled = False
            self.f_crit = [self.f0_slider.value, self.f1_slider.value]
        # In case of notch filter, disable the filter method dropdown menu
        if self.filter_idx == 4:
            self.method_menu.disabled = True
        else:
            self.method_menu.disabled = False
        self.update_filter()
        
    def method_callback(self, value):
        self.filter_method = value['new']
        self.update_filter()