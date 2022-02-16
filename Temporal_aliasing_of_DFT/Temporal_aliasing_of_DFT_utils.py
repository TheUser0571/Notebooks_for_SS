import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets
from scipy import signal
from scipy.interpolate import interp1d
from IPython.display import clear_output
from numpy.fft import fft, fftfreq, fftshift
from Signal_Operator_Library import discrete

class DFT_temporal_aliasing():
    
    def __init__(self):
        
        self.out = Output(layout={'width': '980px', 'height': '700px'})
        self.axs = []
        self.period = 1/5
        self.N = 21
        
        self.x = np.linspace(0, 20, num = 21, endpoint=True)
        self.f = np.vectorize(self.cubic_spline, otypes=[complex])
        self.y = self.f(self.x*self.period)
        
        self.x_period = np.linspace(-self.N*100-10, 10+self.N*100, num=(2*100*self.N+20)+1, endpoint=True)
        self.y_period = np.zeros(len(self.x_period))
        
        #Creating a matrix to store all the functions in the final range
        self.matrix = np.zeros((201,len(self.x_period)), dtype = "complex_")
        #Array with index
        #self.index = np.arange(5*(2*10*self.N+5))
        
        
        self.choose_N = widgets.IntSlider(value = self.N, min = 5, max = 50, step = 1, description = 'Periodization cosntant', 
                                          style={'description_width':'initial'}, layout={'width':'400px'})
        self.choose_N.observe(self.choose_N_callback, names='value')
        
        self.__init__figure()
        #box_layout = Layout(display='flex',
        #            flex_flow='column',justify_content='space-around',margin='solid')
        
        display(VBox([self.choose_N, self.out]))
        plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)
        
    def __init__figure(self):
        
        with self.out:
            
            self.fig = plt.figure('Temporal aliasing of DFT demo',figsize=(9,5),constrained_layout=False)
            self.gs = self.fig.add_gridspec(3,1)
    
            # Plot the original function
            self.axs.append(self.fig.add_subplot(self.gs[0,0]))
            self.axs[0].set_title("$f[n]$")
            self.axs[0].set_yticks([1e-4], minor=True)
            self.axs[0].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xticks([1e-4], minor=True)
            self.axs[0].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xlabel('n')
            markerline, stemlines, baseline = self.axs[0].stem(self.x, self.y.real, use_line_collection=True)
            baseline.set_visible(False)
            
            #Plot the N-periodization function
            self.axs.append(self.fig.add_subplot(self.gs[1,0]))
            self.N_periodization()
            
            #Plot the DFT
            self.axs.append(self.fig.add_subplot(self.gs[2,0]))
            self.plot_DFT()
            plt.show()
         
        
    def plot_N_period(self):
        
        self.axs[1].set_title("$f_N[n]$")
        self.axs[1].set_yticks([1e-4], minor=True)
        self.axs[1].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
        self.axs[1].set_xticks([1e-4], minor=True)
        self.axs[1].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
        self.axs[1].set_xlabel('n')
        self.x_period_cut = self.x_period[int((len(self.x_period)-1)/2-40):int((len(self.x_period)-1)/2+61)]
        self.x_period_cut_dft = self.x_period[int((len(self.x_period)-1)/2):int((len(self.x_period)-1)/2+21)]
        self.y_period_cut = self.y_period[int((len(self.x_period)-1)/2-50):int((len(self.x_period)-1)/2+51)]
        self.y_period_cut_dft = self.y_period[int((len(self.x_period)-1)/2-10):int((len(self.x_period)-1)/2+11)]
        self.markerline, self.stemline, baseline = self.axs[1].stem(self.x_period_cut, self.y_period_cut.real, linefmt='C2-',  markerfmt='C2o', use_line_collection=True)
        self.axs[1].axvspan(0, 20, ymin=0.05, color='grey', alpha=0.5)
        self.markerline.set_markersize(2)
        self.stemline.set_linewidth(1)
        baseline.set_visible(False)
        
    
    def plot_DFT(self):
        
        self.axs[2].set_title("$|F_N[m]|$")
        self.axs[2].set_yticks([1e-4], minor=True)
        self.axs[2].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
        self.axs[2].set_xticks([1e-4], minor=True)
        self.axs[2].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
        self.axs[2].set_xlabel('m')
        vector = np.vectorize(int)
        x_fourier = fftshift(fftfreq(len(self.x_period_cut_dft)))
        y_fourier = fftshift(discrete.DFT(self.y_period_cut_dft, np.arange(0,len(self.y_period_cut_dft))))
        markerline, stemlines, baseline = self.axs[2].stem(x_fourier, np.abs(y_fourier), linefmt='C3-',  markerfmt='C3o', use_line_collection=True)
        baseline.set_visible(False)
    
    def choose_N_callback(self, value):
        self.N = value['new']
        self.x_period = np.linspace(-self.N*100-10, 10+self.N*100, num=(2*100*self.N+20)+1, endpoint=True)
        self.y_period = np.zeros(len(self.x_period))
        self.matrix = np.zeros((201,len(self.x_period)), dtype = "complex_")
        #Array with index
        self.index = np.arange(len(self.x_period))
        
        self.N_periodization()
        self.axs[2].clear()
        self.plot_DFT()
    
        
        
    def cubic_spline(self,x):
        if(abs(x-2)<1): return(2/3.-abs(x-2)**2+(abs(x-2)**3)/2.)
        elif(abs(x-2)>=1 and abs(x-2)<2): return((2-abs(x-2))**3)/6.
        else: return 0
        
    def N_periodization(self):
        
        for i in range(0,int((len(self.matrix[:, 0])+1)/2)):
            center = i*self.N + (len(self.matrix[0])-1)/2
            arr = np.arange(-10,11)
            indices = arr+int(center)
            if i==0:
                self.matrix[i, indices] = self.y
            else: 
                self.matrix[2*i, indices] = self.y
                self.matrix[2*i-1, indices-2*i*self.N] = self.y
        
        self.y_period = np.sum(self.matrix, axis=0)
        self.axs[1].clear()
        self.plot_N_period()
        