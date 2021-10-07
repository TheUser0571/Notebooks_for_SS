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

class DTFT_properties():
    def __init__(self):
        self.out = Output(layout={'width': '980px', 'height': '800px'})
        self.axs = []
        self.shift_int = 0
        self.modulation_float = 0
        self.w0 = 2*np.pi/5.
        # Test functions:
        self.test_function = {'Casual monome': 0, 'Causal exponential': 1, 'sin': 2, 'cos': 3}
        
        #self.funct returns the string of above that is chosen. I'll use it later for the type of interpolation
        self.funct = list(self.test_function.keys())[2]
        self.funct_idx = self.test_function[self.funct]
        
        # Properties:
        self.properties = {'Shift':0, 'Modulation':1}
        
        #self.funct returns the string of above that is chosen. I'll use it later for the type of interpolation
        self.prop = list(self.properties.keys())[0]
        self.prop_idx = self.properties[self.prop]
        
        # Dropdown menu to select the test function 
        self.funct_menu = widgets.Dropdown(options=self.test_function.keys(), value=self.funct, 
                                            description="Function:", layout=Layout(width='160px'), style={'description_width': 'initial'})
        self.funct_menu.observe(self.menu_callback, names='value')
        
        # Dropdown menu to select the property 
        self.property_menu = widgets.Dropdown(options=self.properties.keys(), value=self.prop, 
                                            description="Property:", layout=Layout(width='160px'), style={'description_width': 'initial'})
        self.property_menu.observe(self.property_callback, names='value')
        
        self.init_function = [self.causal_monome, self.causal_expo, self.sin, self.cos ]
        
        #Text to write the parameter to apply the shift
        self.shift = widgets.IntText(
            value=self.shift_int, description="Shift:", layout=Layout(width='160px'), style={'description_width': 'initial'})
        self.shift.observe(self.shift_callback, names='value')
        
        #Text to write the parameter to apply the modulation
        self.modulation = widgets.FloatText(
            value=self.modulation_float, description="Modulation:", layout=Layout(width='160px'), style={'description_width': 'initial'})
        self.modulation.observe(self.modulation_callback, names='value')
        
        
        self.__init__figure()
        box_layout = Layout(display='flex',
                    flex_flow='column',justify_content='space-around',margin='solid', width='200px')
        display(HBox([VBox([self.funct_menu, self.property_menu, self.shift, self.modulation], layout=box_layout), self.out]))
        plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)
        
        
    def __init__figure(self):
        with self.out:
            
            self.fig = plt.figure('Properties of DTFT demo', figsize=(7,7), constrained_layout=False)
            self.gs = self.fig.add_gridspec(4,4)
    
            # Plot the original function
            self.axs.append(self.fig.add_subplot(self.gs[:2,:2]))
            self.axs[0].set_title("f[n]")
            self.axs[0].set_yticks([1e-4], minor=True)
            self.axs[0].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xticks([1e-4], minor=True)
            self.axs[0].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            #self.axs[0].set_xlabel("t[s]")
            #position = [-2, -1, 0, 1, 2]
            #self.axs[0].set_xticks(position)
            #self.axs[0].set_xlim([-2,2])
            #self.axs[0].set_ylim([0,2])
            #self.axs[0].plot(self.interp_f_x, self.interp_f_y, color='blue', linewidth=0.2)
            
            # Plot the modified function
            self.axs.append(self.fig.add_subplot(self.gs[2:,:2]))
            self.axs[1].set_title("f[n] modified")
            self.axs[1].set_yticks([1e-4], minor=True)
            self.axs[1].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[1].set_xticks([1e-4], minor=True)
            self.axs[1].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            
            # Plot modulus fourier transform
            self.axs.append(self.fig.add_subplot(self.gs[0,2:]))
            self.axs[2].set_title("|F[n]|")
            self.axs[2].set_yticks([1e-4], minor=True)
            self.axs[2].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[2].set_xticks([1e-4], minor=True)
            self.axs[2].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            
            # Plot arg of fourier transform
            self.axs.append(self.fig.add_subplot(self.gs[1,2:]))
            self.axs[3].set_title("Arg(F[n])")
            self.axs[3].set_yticks([1e-4], minor=True)
            self.axs[3].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[3].set_xticks([1e-4], minor=True)
            self.axs[3].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            
            # Plot modulus of the fourier transform modified
            self.axs.append(self.fig.add_subplot(self.gs[2,2:]))
            self.axs[4].set_title("|F[n]| modified")
            self.axs[4].set_yticks([1e-4], minor=True)
            self.axs[4].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[4].set_xticks([1e-4], minor=True)
            self.axs[4].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            
            # Plot arg of fourier transform modified
            self.axs.append(self.fig.add_subplot(self.gs[3,2:]))
            self.axs[5].set_title("Arg(F[n]) modified")
            self.axs[5].set_yticks([1e-4], minor=True)
            self.axs[5].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[5].set_xticks([1e-4], minor=True)
            self.axs[5].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.update_fourier(init=True)
            self.update_fourier_modified(init=True)
            # Plot the signal interpolated
            
    def sin(self, x):
        return np.sin(self.w0*x)
    
    def cos(self, x):
        return np.cos(self.w0*x)
    
    def causal_monome(self, x):
        if x >=0: return (x+1)
        else: return 0
        
    def causal_expo(self, x):
        if np.sign(x)>=0: return np.exp(-2*x)
        else: return 0
    
    def modul_expo(self, x):
        return (np.cos(self.modulation_float*x)+np.sin(self.modulation_float*x)*1j)
    
    
    
    def menu_callback(self, change):
        self.funct = change['new']
        self.funct_idx = self.test_function[self.funct]
        self.update_fourier()
        self.update_fourier_modified()
        
    def property_callback(self, change):
        self.prop = change['new']
        self.prop_idx = self.properties[self.prop]
        self.update_fourier_modified()
        
    def shift_callback(self, change):
        self.shift_int = change['new']
        self.update_fourier_modified()
    
    def modulation_callback(self, change):
        self.modulation_float = change['new']
        self.update_fourier_modified()
        
    def update_fourier(self, init=False):
        
        x = np.linspace(-10, 10, num=21, endpoint=True)
        f = np.vectorize(self.init_function[self.funct_idx], otypes=[complex])
        y = f(x)
        x_fourier = fftshift(fftfreq(x.size, d=1.))
        y_fourier = fftshift(discrete.DTFT(y, x_fourier))
        y_fourier_mod = np.abs(y_fourier)
        y_fourier_arg = np.angle(y_fourier)
        
        if init:
            markerline, stemlines, baseline = self.axs[0].stem(x, y.real, use_line_collection=True)
            baseline.set_visible(False)
            markerline, stemlines, baseline = self.axs[2].stem(x_fourier, y_fourier_mod, linefmt='C2-', use_line_collection=True)
            baseline.set_visible(False)
            markerline, stemlines, baseline = self.axs[3].stem(x_fourier, y_fourier_arg, linefmt='C3-', use_line_collection=True)
            baseline.set_visible(False)
        
        else:
            self.axs[0].clear()
            self.axs[0].set_title("f[n]")
            self.axs[0].set_yticks([1e-4], minor=True)
            self.axs[0].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xticks([1e-4], minor=True)
            self.axs[0].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            markerline, stemlines, baseline = self.axs[0].stem(x, y.real, use_line_collection=True)
            baseline.set_visible(False)
            
            self.axs[2].clear()
            self.axs[2].set_title("|F[n]|")
            self.axs[2].set_yticks([1e-4], minor=True)
            self.axs[2].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[2].set_xticks([1e-4], minor=True)
            self.axs[2].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            markerline, stemlines, baseline = self.axs[2].stem(x_fourier, y_fourier_mod, linefmt='C2-', use_line_collection=True)
            baseline.set_visible(False)
            
            self.axs[3].clear()
            self.axs[3].set_title("Arg(F[n])")
            self.axs[3].set_yticks([1e-4], minor=True)
            self.axs[3].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[3].set_xticks([1e-4], minor=True)
            self.axs[3].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            markerline, stemlines, baseline = self.axs[3].stem(x_fourier, y_fourier_arg, linefmt='C3-', use_line_collection=True)
            baseline.set_visible(False)
        
        self.axs[0].set_xlim(np.min(x)-1, np.max(x)+1)
        self.axs[0].set_ylim(np.min(y.real)-(np.max(y.real)-np.min(y.real))/len(y.real), np.max(y.real)+(np.max(y.real)-np.min(y.real))/len(y.real))
            
        self.axs[2].set_xlim(np.min(x_fourier)-(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier), np.max(x_fourier)+(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier))
        self.axs[2].set_ylim(np.min(y_fourier_mod)-(np.max(y_fourier_mod)-np.min(y_fourier_mod))/len(y_fourier_mod), np.max(y_fourier_mod)+(np.max(y_fourier_mod)-np.min(y_fourier_mod))/len(y_fourier_mod))
            
        self.axs[3].set_xlim(np.min(x_fourier)-(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier), np.max(x_fourier)+(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier))
        self.axs[3].set_ylim(np.min(y_fourier_arg)-(np.max(y_fourier_arg)-np.min(y_fourier_arg))/len(y_fourier_arg), np.max(y_fourier_arg)+(np.max(y_fourier_arg)-np.min(y_fourier_arg))/len(y_fourier_arg))        

    
    def update_fourier_modified(self, init=False):
        
        if self.prop_idx == 0:
            # Shift property
            x = np.linspace(-10-self.shift_int,10-self.shift_int,num=21, endpoint=True)
            f = np.vectorize(self.init_function[self.funct_idx], otypes=[complex])
            y = f(x)
            x_fourier = fftshift(fftfreq(x.size, d=1.))
            y_fourier = fftshift(discrete.DTFT(y, x_fourier))
            y_fourier_mod = np.abs(y_fourier)
            y_fourier_arg = np.angle(y_fourier)
            
        elif self.prop_idx == 1:
            # modulation property
            x = np.linspace(-10, 10, num=21, endpoint=True)
            f = np.vectorize(self.init_function[self.funct_idx], otypes=[complex])
            g = np.vectorize(self.modul_expo, otypes=[complex])
            y = f(x)*g(x)
            x_fourier = fftshift(fftfreq(x.size, d=1.))
            y_fourier = fftshift(discrete.DTFT(y, x_fourier))
            y_fourier_mod = np.abs(y_fourier)
            y_fourier_arg = np.angle(y_fourier)
            
        
            
        if init:
            markerline, stemlines, baseline = self.axs[1].stem(x,y.real, markerfmt='C1o', use_line_collection=True)
            baseline.set_visible(False)
            markerline, stemlines, baseline = self.axs[4].stem(x_fourier, y_fourier_mod, linefmt='C2-', markerfmt='C1o', use_line_collection=True)
            baseline.set_visible(False)
            markerline, stemlines, baseline = self.axs[5].stem(x_fourier, y_fourier_arg, linefmt='C3-', markerfmt='C1o', use_line_collection=True)
            baseline.set_visible(False)
        
        else:
            self.axs[1].clear()
            self.axs[1].set_title("f[n] modified")
            self.axs[1].set_yticks([1e-4], minor=True)
            self.axs[1].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[1].set_xticks([1e-4], minor=True)
            self.axs[1].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            markerline, stemlines, baseline = self.axs[1].stem(x, y.real, markerfmt='C1o', use_line_collection=True)
            baseline.set_visible(False)
            
            self.axs[4].clear()
            self.axs[4].set_title("|F[n]| modified")
            self.axs[4].set_yticks([1e-4], minor=True)
            self.axs[4].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[4].set_xticks([1e-4], minor=True)
            self.axs[4].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            markerline, stemlines, baseline = self.axs[4].stem(x_fourier, y_fourier_mod, linefmt='C2-', markerfmt='C1o', use_line_collection=True)
            baseline.set_visible(False)
            
            self.axs[5].clear()
            self.axs[5].set_title("Arg(F[n]) modified")
            self.axs[5].set_yticks([1e-4], minor=True)
            self.axs[5].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[5].set_xticks([1e-4], minor=True)
            self.axs[5].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            markerline, stemlines, baseline = self.axs[5].stem(x_fourier, y_fourier_arg, linefmt='C3-', markerfmt='C1o', use_line_collection=True)
            baseline.set_visible(False)
            
        self.axs[1].set_xlim(np.min(x)-1, np.max(x)+1)
        self.axs[1].set_ylim(np.min(y.real)-(np.max(y.real)-np.min(y.real))/len(y.real), np.max(y.real)+(np.max(y.real)-np.min(y.real))/len(y.real))
            
        self.axs[4].set_xlim(np.min(x_fourier)-(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier), np.max(x_fourier)+(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier))
        self.axs[4].set_ylim(np.min(y_fourier_mod)-(np.max(y_fourier_mod)-np.min(y_fourier_mod))/len(y_fourier_mod), np.max(y_fourier_mod)+(np.max(y_fourier_mod)-np.min(y_fourier_mod))/len(y_fourier_mod))
            
        self.axs[5].set_xlim(np.min(x_fourier)-(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier), np.max(x_fourier)+(np.max(x_fourier)-np.min(x_fourier))/len(x_fourier))
        self.axs[5].set_ylim(np.min(y_fourier_arg)-(np.max(y_fourier_arg)-np.min(y_fourier_arg))/len(y_fourier_arg), np.max(y_fourier_arg)+(np.max(y_fourier_arg)-np.min(y_fourier_arg))/len(y_fourier_arg))