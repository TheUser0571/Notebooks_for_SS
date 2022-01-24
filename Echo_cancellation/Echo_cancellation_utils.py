import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from ipywidgets import HBox, Layout, Output, VBox
import ipywidgets as widgets
import IPython
from IPython.display import HTML
from scipy import signal

class FloatCheckCustom():
    def __init__(self, val, description_1='Value:', description_2='Value:', width='100px'):
        self.text_1 = widgets.FloatText(value=0, description=description_1, layout=Layout(width=width), style={'description_width':'initial'})
        self.text_1.observe(self.check_answer_1, names='value')
        self.text_2 = widgets.FloatText(value=0, description=description_2, layout=Layout(width=width), style={'description_width':'initial'})
        self.text_2.observe(self.check_answer_2, names='value')
        self.val = val
        self.incorrect_html = '<p style="color:Red;">Incorrect</p>'
        self.correct_html = '<p style="color:Green;">Correct</p>'
        self.html_1 = widgets.HTML('', layout=Layout(width='75px'))
        self.html_2 = widgets.HTML('', layout=Layout(width='75px'))
        self.display = HBox([self.text_1, self.html_1, self.text_2, self.html_2])
    
    def check_answer_1(self, value):
        if abs(value['new']) == self.val:
            self.html_1.value = self.correct_html
            # One needs to be negative, the other positive
            if value['new'] == -self.text_2.value:
                self.html_2.value = self.correct_html
            else:
                self.html_2.value = self.incorrect_html       
        else:
            self.html_1.value = self.incorrect_html
    
    def check_answer_2(self, value):
        if abs(value['new']) == self.val:
            self.html_2.value = self.correct_html
            # One needs to be negative, the other positive
            if value['new'] == -self.text_1.value:
                self.html_1.value = self.correct_html
            else:
                self.html_1.value = self.incorrect_html       
        else:
            self.html_2.value = self.incorrect_html

class EchoCancellation():
    def __init__(self, filename='Sample.wav'):
        # Load the original signal
        self.FS, self.x = wavfile.read(filename)
        # Convert to mono and shorten
        self.x = np.mean(self.x[len(self.x)//5:], axis=1)
        # Normalize
        self.x = self.x / np.max(np.abs(self.x))
        
        # Generate the echoed signal
        t0 = 0.5
        n0 = int(t0 * self.FS)
        a = 0.4
        echo = np.zeros(len(self.x))
        echo[n0:] = self.x[:-n0]
        self.y = self.x + a * echo
        
        # Q1
        self.Q1 = FloatCheck(minval=3.85, maxval=3.86, description='$T=$').display
        
        # Q2
        Q2_1 = FloatCheck(minval=1.25, maxval=1.25, description='$b_0=$', width='100px')
        Q2_2 = FloatCheck(minval=0.5, maxval=0.5, description='$b_1=$', width='100px')
        Q2_3 = FloatCheck(minval=0.5, maxval=0.5, description='$b_2=$', width='100px')
        Q2_4 = FloatCheckCustom(val=5, description_1='$m_1=$', description_2='$m_2=$', width='100px')
        self.Q2 = HBox([Q2_1.display, Q2_2.display, Q2_3.display, Q2_4.display])
        
        # Q3
        Q3_1 = FloatCheck(minval=0.35, maxval=0.5, description='$a=$', width='100px')
        Q3_2 = FloatCheck(minval=22000, maxval=26000, description='$n_0=$', width='120px')
        self.Q3 = HBox([Q3_1.display, Q3_2.display])
        
        # Q4
        self.Q4 = FloatCheck(minval=0.458, maxval=0.542, description='$T_e=$', width='100px').display

    def display_audio_file(self):
        # Plot
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(8, 2), num='Audio file display')
        ax.plot(np.linspace(0, len(self.y)/self.FS, len(self.y)), self.y, linewidth=0.7)
        ax.set_title('Audio file'); ax.set_xlabel('Time [s]'); plt.tight_layout(); plt.show()
        # Create audio widget
        display(IPython.display.Audio(self.y, rate=self.FS))
    
    def display_auto_corr_x(self):
        # Calculate auto correlation of x
        corr_x = signal.correlate(self.x, self.x, mode='same')
        corr_x = corr_x / np.max(np.abs(corr_x)) # Normalize

        # Plot
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), num='Auto correlation x display')
        ax.plot(np.linspace(-len(corr_x)//2, len(corr_x)//2-1, len(corr_x)), corr_x, linewidth=0.5)
        ax.grid(); ax.set_title('$R_x[m]$'); ax.set_xlabel('$m$'); plt.show()

    def display_auto_corr_y(self):
        # Calculate auto correlation of y
        corr_y = signal.correlate(self.y, self.y, mode='same')
        corr_y = corr_y / np.max(np.abs(corr_y)) # Normalize

        # Peak detection
        peaks, _ = signal.find_peaks(corr_y, height=0.3*np.max(corr_y), distance=1000)
        peaks_adjusted = peaks - len(corr_y)//2

        # Plot
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), num='Auto correlation y display')
        ax.plot(np.linspace(-len(corr_y)//2, len(corr_y)//2-1, len(corr_y)), corr_y, linewidth=0.5)
        for i in range(len(peaks)):
            ax.plot(peaks_adjusted[i], corr_y[peaks[i]], 'rx')
            ax.text(peaks_adjusted[i], corr_y[peaks[i]]+0.05, f'({peaks_adjusted[i]}, {corr_y[peaks[i]]:.3f})', horizontalalignment='center', color='red')
        ax.set_ylim([ax.get_ylim()[0], 1.15]); ax.grid(); ax.set_title('$R_y[m]$'); ax.set_xlabel('$m$'); plt.show()

    def display_x_filtered(self):
        # Get a and n0 from the answer above
        a = self.Q3.children[0].children[0].value
        n0 = int(self.Q3.children[1].children[0].value)

        # Filter signal
        den = np.zeros(n0 + 1)
        den[0] = 1.
        den[-1] = a
        x_est = signal.lfilter(np.array([1.]), den, self.y)

        # Plot signal
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(8, 2), num='Filtered x display')
        ax.plot(np.linspace(0, len(x_est)/self.FS, len(x_est)), x_est, linewidth=0.7)
        ax.set_title(r'$\tilde{x}[n]$'); ax.set_xlabel('Time [s]'); plt.tight_layout(); plt.show()
        # Create audio widget
        display(IPython.display.Audio(x_est, rate=self.FS))


class FloatCheck():
    def __init__(self, minval, maxval, description='Value:', width='100px'):
        self.text = widgets.FloatText(value=0, description=description, layout=Layout(width=width), style = {'description_width': 'initial'})
        self.text.observe(self.check_answer, names='value')
        self.min = minval
        self.max = maxval
        self.incorrect_html = '<p style="color:Red;">Incorrect</p>'
        self.correct_html = '<p style="color:Green;">Correct</p>'
        self.html = widgets.HTML('', layout=Layout(width='75px'))
        self.display = HBox([self.text, self.html])
    
    def check_answer(self, value):
        if value['new'] >= self.min and value['new'] <= self.max:
            self.html.value = self.correct_html
        else:
            self.html.value = self.incorrect_html