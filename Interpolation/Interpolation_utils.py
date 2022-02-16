import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt
from ipywidgets import HBox, Layout, Output, VBox
import ipywidgets as widgets
from scipy import signal
from scipy.interpolate import interp1d
from exercise_utils import *

class Interpolation():
    def __init__(self):
        self.Q1 = MCQ(['Rect','Tri', 'Sinc'], 'Sinc').display
        self.Q2 = MCQ(['0.1','0.25', '0.5', 'None of the above'], 'None of the above').display
        self.Q3 = MCQ(['0.1','0.25', '0.5', 'None of the above'], 'None of the above').display
        self.Q4 = MCQ(['0.1','0.25', '0.5', 'None of the above'], '0.5').display
        
        self.out = Output(layout={'width': '980px', 'height': '450px'})

        self.axs = []
        self.period = 0.2
        
        # Interpolation function type
        self.interp_funct_types = {'rect':0, 'tri':1, 'sinc':2}
        
        #self.funct returns the string of above that is chosen. I'll use it later for the type of interpolation
        self.funct = list(self.interp_funct_types.keys())[0]
        self.funct_idx = self.interp_funct_types[self.funct]
        
        #Here I define the array to draw the interpolating function
        self.interp_f_x = np.linspace(-3, 3, num=1001)
        self.interp_f_y = np.zeros(1001)
        #self.interp_f_y = np.cos(self.interp_f_y)
        
        # Dropdown menu to select interpolation function type 
        self.funct_menu = widgets.Dropdown(options=self.interp_funct_types.keys(), 
                                           value=self.funct, 
                                           description="Interpolation function: ", 
                                           layout=Layout(width='200px'),
                                           style = {'description_width': 'initial'})
        
        
        self.funct_menu.observe(self.menu_callback, names='value')
        

        
        
        #self.x_funct = np.linspace(0, 10, num= 1000, endpoint=True)
        #self.y_funct = np.sin(self.x_funct)
        self.x = np.linspace(-2.5, 2.5, num=1001, endpoint=True)
        self.y = np.sinc(self.x)**2
        self.x_val = np.linspace(-2.5, 2.5, num=int(5/self.period)+1, endpoint=True)
        self.y_val = np.sinc(self.x_val)**2
        self.x_interp = np.linspace(-2.5, 2.5, num=1001, endpoint=True)
        self.y_interp = np.zeros(1001)
        #self.f = interp1d(self.x, self.y, kind='nearest')
        #self.y_interp = self.f(self.x_interp)
        
        self.print_err = widgets.Text(
            value=str(np.round(self.error(), 3)), description="Error:", layout=Layout(width='150px'), style={'description_width':'70px'})
        
        self.sampling_period = widgets.SelectionSlider(options=list(np.round(5 / np.linspace(50, 1, 50), 3)),
                                                       value=self.period,
                                                       continuous_update=True, 
                                                       description='Sampling period', 
                                                       style={'description_width':'150px'},
                                                       layout=Layout(width='400px'))
        self.sampling_period.observe(self.sampling_period_callback, names='value')
        #For the interpolation function
        #self.functx = np.linspace(-1, 1, num=100, endpoint= True)
        #self.functy = np.full(shape=100, fill_value=1, dtype=np.int)
        
        self.init_figure()
        
        display(VBox([self.out, HBox([self.funct_menu, self.sampling_period, self.print_err])]))
        plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)
    
    
    def init_figure(self):
        with self.out:
            self.fig = plt.figure('Interpolation Demo', figsize=(8, 4), constrained_layout=False)
            self.gs = self.fig.add_gridspec(1, 3)
    
            #Plot the interpolating function
            self.axs.append(self.fig.add_subplot(self.gs[0, 0]))
            self.axs[0].set_title("Rect function")
            self.axs[0].set_yticks([1e-4], minor=True)
            self.axs[0].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xticks([1e-4], minor=True)
            self.axs[0].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xlabel("t[s]")
            
            #position = [-2, -1, 0, 1, 2]
            #self.axs[0].set_xticks(position)
            self.axs[0].set_xlim([-2,2])
            self.axs[0].set_ylim([0,2])
            #self.axs[0].plot(self.interp_f_x, self.interp_f_y, color='blue', linewidth=0.2)
            self.update_interp_f(init=True)
            #Plot the signal interpolated
            
            self.axs.append(self.fig.add_subplot(self.gs[0, 1:]))
            self.axs[1].set_title("Interpolation")
            self.axs[1].plot(self.x, self.y, '--')
            self.axs[1].plot(self.x_interp, self.interpolation(), '-')
            self.axs[1].plot(self.x_val, self.y_val, 'or')
            self.axs[1].legend(['data', 'interpolated data ', 'sampled data'], loc='upper left')
            plt.show()
        
    def interpolation(self):
        if(self.funct_idx==0):
            f = interp1d(self.x_val, self.y_val, kind="nearest")
            return f(self.x_interp)
        if(self.funct_idx==1): 
            f = interp1d(self.x_val, self.y_val, kind="linear")
            return f(self.x_interp)
        if(self.funct_idx==2): 
            return self.sinc_interp(self.y_val, self.x_val, self.x_interp)
        
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
        
    
    def menu_callback(self, value):
        self.funct = value['new']
        self.funct_idx = self.interp_funct_types[self.funct]
        self.update_interp_f()
        self.y_interp = self.interpolation()
        self.axs[1].lines[1].set_data(self.x_interp, self.y_interp)
        self.print_err_callback()
        
    def sampling_period_callback(self, value):
        self.period = value['new']
        self.x_val = np.linspace(-2.5, 2.5, num=int(5/self.period)+1, endpoint=True)
        self.y_val = np.sinc(self.x_val)**2
        self.axs[1].lines[2].set_data(self.x_val, self.y_val)
        self.y_interp = self.interpolation()
        self.axs[1].lines[1].set_data(self.x_interp, self.y_interp)
        self.print_err_callback()
        
    
    def print_err_callback(self):
        self.print_err.value = str(np.round(self.error(), 3))
        
        
    #Definition of the rect function
    def rect(val):
        return np.where(np.abs(val)<=0.5, 1, 0)
    
    #Definition of the linear spline function
    def tri(self,x):
        if(x>=-1. and x<0.): return(x+1.)
        elif(x>=0. and x<=1.): return(-x+1.)
        else: return 0.
        

        
    #the function is going to be sinc (t/T), with T the sampling period: space between the discrete points
    def sinc(self, x):
        return np.sinc(x/0.2)
    
    #Funtion to calculate the aproximation error
    def error(self):
        err = 0.
        for i in range(len(self.x_interp)):
            err += (self.y_interp[i] - self.y[i])**2
            
        return err
    
    #Function to update the interpolation function, so it changes the interpolation function plot and the
    #interpolation plot
    def update_interp_f(self, init=False):
       
        if(self.funct_idx==0):
            y = np.where(np.abs(self.interp_f_x)<=0.5, 1, 0)
            self.axs[0].set_xlim([-0.75,0.75])
            self.axs[0].set_ylim([0,1.5])
            self.axs[0].set_title("Rect")
        
        if(self.funct_idx==1):
            y = list(map(self.tri,self.interp_f_x))
            self.axs[0].set_xlim([-1.25,1.25])
            self.axs[0].set_ylim([0,1.5])
            self.axs[0].set_title("Tri")
            
        if(self.funct_idx==2):
            y = list(map(self.sinc,self.interp_f_x))
            self.axs[0].set_xlim([-1,1])
            self.axs[0].set_ylim([-0.5,1.5])
            self.axs[0].set_title("sinc(t/T)")
            
            
        if init:
            self.axs[0].plot(self.interp_f_x, y, color='blue')
        else:
            self.axs[0].lines[0].set_data(self.interp_f_x, y)
