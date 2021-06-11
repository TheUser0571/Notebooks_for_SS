import numpy as np
from scipy.io import wavfile
from ipywidgets import HBox, Layout, Output, VBox
import ipywidgets as widgets
from IPython.display import HTML

class EchoCancellation():
    def __init__(self):
        # Load the original signal
        self.FS, self.x = wavfile.read('Sample.wav')
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
        Q2_1 = FloatCheck(minval=1.25, maxval=1.25, description='$a_0=$', width='100px')
        Q2_2 = FloatCheck(minval=0.5, maxval=0.5, description='$a_1=$', width='100px')
        Q2_3 = FloatCheck(minval=0.5, maxval=0.5, description='$a_2=$', width='100px')
        Q2_4 = FloatCheck(minval=-5, maxval=-5, description='$m_1=$', width='100px')
        Q2_5 = FloatCheck(minval=5, maxval=5, description='$m_2=$', width='100px')
        self.Q2 = HBox([Q2_1.display, Q2_2.display, Q2_3.display, Q2_4.display, Q2_5.display])
        
        # Q3
        Q3_1 = FloatCheck(minval=0.35, maxval=0.5, description='$a=$', width='100px')
        Q3_2 = FloatCheck(minval=22000, maxval=26000, description='$n_0=$', width='120px')
        self.Q3 = HBox([Q3_1.display, Q3_2.display])
        
        # Q4
        self.Q4 = FloatCheck(minval=0.458, maxval=0.542, description='$T_e=$', width='100px').display


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