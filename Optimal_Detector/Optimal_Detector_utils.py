import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from ipywidgets import FloatSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets
from exercise_utils import *

class ODEx():
    
    Q1 = FloatCheck(minval=0.33, maxval=0.334, description='Value:', width='120px').display
    
    Q2 = MCQ(['4/5 tri(3y)', '4/15 tri(3y)', '1/3 tri(y/3)', '4/15 tri(y/3)'], 
                  correct='1/3 tri(y/3)').display
    
    Q3 = MCQ(['1/15 tri((y-4) / 3)', '1/3 tri((y-4) / 3)', '1/15 tri(y/3) + 4', '1/3 tri(y/3) + 4'], 
                  correct='1/3 tri((y-4) / 3)').display
    
    Q4 = MCQ(['1/3 tri(y/3) + 4/5', '2/3 tri(y/3) + 4', '1/3 tri(y/3) + 1/3 tri((y-4) / 3)', '4/15 tri(y/3) + 1/15 tri((y-4) / 3)'], 
                  correct='4/15 tri(y/3) + 1/15 tri((y-4) / 3)').display
    
    Q5 = FloatCheck(minval=1.5, maxval=2.5, description='Value:', width='100px').display
    
    Q6 = FloatCheck(minval=0, maxval=50, description='Value:', width='100px').display
    

def get_x(s=10000, p_4=0.2):
    x = np.zeros(s)
    x[int(len(x)*(1-p_4)):] = 4
    return np.random.permutation(x)

def tri(x):
    return np.where(np.abs(x) <= 1, 1-np.abs(x), 0)

def get_pdf(n=np.linspace(-3, 3, 10000)):
    return PDF(n, 1/3 * tri(n/3))

class PDF():
    def __init__(self, n, pdf):
        self.pdf = pdf
        self.n = n
    
    def draw_sample(self, s=1):
        pdf = self.pdf / np.sum(self.pdf)
        return np.random.choice(self.n, p=pdf, size=None if s==1 else s)
    
    def disp(self, yaxis=False):
        plt.close('all')
        fig, ax = plt.subplots(1, 1, num='Probability density function')
        ax.set_title('g(n)')
        ax.plot(self.n, self.pdf)
        ax.set_xlabel('n')
        if not yaxis:
            ax.yaxis.set_visible(False)
        plt.show()
    
class OptimalDetecor():
    def __init__(self, x, pdf):
        self.x = x
        self.pdf = pdf
        self.n = self.pdf.draw_sample(len(self.x))
        self.fig = None
        self.ax = None
        
    def hist(self):
        plt.close('all')
        fig, ax = plt.subplots(1,1, num='Histogram')
        
        y_0 = self.x[self.x==0] + self.n[self.x==0]
        y_4 = self.x[self.x==4] + self.n[self.x==4]

        assert len(y_0)!=0 or len(y_4)!=0, "x does not contain any values that are 0 or 4."
        
        if len(y_0)==0:
            y_max = np.ceil(np.max(y_4))
            y_min = np.floor(np.min(y_4))
        elif len(y_4)==0:
            y_max = np.ceil(np.max(y_0))
            y_min = np.floor(np.min(y_0))
        else:
            y_max = np.ceil(max(np.max(y_0), np.max(y_4)))
            y_min = np.floor(min(np.min(y_0), np.min(y_4)))
            
        ax.set_title('Histogram of Y=X+N')
        if len(y_0) == 0 or len(y_4) == 0:
            if len(y_0) == 0:
                ax.hist(y_4, bins=np.linspace(y_min, y_max, int(y_max-y_min)*2+1), color='m', label='X=4')
            if len(y_4) == 0:
                ax.hist(y_0, bins=np.linspace(y_min, y_max, int(y_max-y_min)*2+1), color='b', label='X=0')
        else:
            ax.hist([y_0, y_4], bins=np.linspace(y_min, y_max, int(y_max-y_min)*2+1), histtype='barstacked', color=['b', 'm'], label=['X=0', 'X=4'])
        ax.legend()
        # Set ticks
        # Major
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter('{x:.0f}')
        # Minor
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        
        ax.set_xlabel('Y')
        ax.set_ylabel('Number of samples')

        ax.xaxis.grid(which='both')
        plt.show()
    
    
    def hist_beta(self, beta, update=False):
        if update:
            self.ax.clear()
        else:
            plt.close('all')
            
            self.out = Output(layout={'width': '980px', 'height': '500px'})
            with self.out:
                self.fig, self.ax = plt.subplots(1,1, num='Histogram with beta')
        self.ax.set_title('Histogram of Y=X+N')

        n = self.pdf.draw_sample(len(self.x))
        y = self.x + self.n

        ys = [y[np.logical_and(self.x==0, y<=beta)], 
              y[np.logical_and(self.x==0, y>beta)], 
              y[np.logical_and(self.x==4, y>beta)], 
              y[np.logical_and(self.x==4, y<=beta)]]
        colors = ['blue', 'lightskyblue', 'm', 'plum']
#         hatches = ['/', '|', '\\', '-']
        hatches = ['', '', '', '']
        labels = ['X=0 correct', 'X=0 incorrect', 'X=4 correct', 'X=4 incorrect']

        # Calculate y max and min
        y_max = 0
        for i in range(len(ys)):
            if len(ys[i]) > 0:
                max_ = np.ceil(np.max(ys[i]))
                y_max = max_ if max_ > y_max else y_max
        y_min = y_max
        for i in range(len(ys)):
            if len(ys[i]) > 0:
                min_ = np.floor(np.min(ys[i]))
                y_min = min_ if min_ < y_min else y_min

        bins = np.linspace(y_min, y_max, int(y_max-y_min)*2+1)
        bottom = np.zeros(len(bins) - 1)

        for i in range(len(ys)):    
            if i == 0:
                self.ax.hist(ys[i], bins=bins, color=colors[i], label=labels[i], hatch=hatches[i])
            else:
                h, _ = np.histogram(ys[i-1], bins)
                bottom += h

                self.ax.hist(ys[i], bins=bins, bottom=bottom, color=colors[i], label=labels[i], hatch=hatches[i])
                
        # Add line for beta
        self.ax.plot([beta, beta], [self.ax.get_ylim()[0], self.ax.get_ylim()[1]], 'r--', label=r'$\beta$', linewidth=0.8)
        
        self.ax.legend()
        # Set ticks
        # Major
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        self.ax.xaxis.set_major_formatter('{x:.0f}')
        # Minor
        self.ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        self.ax.set_xlabel('Y')
        self.ax.set_ylabel('Number of samples')
        self.ax.xaxis.grid(which='both')
        
        if not update:
            self.beta_slider = widgets.FloatSlider(value=beta, min=y_min, max=y_max, description=r'$\beta$:', 
                                                   continuous_update=False, step=0.01, layout={'width': '680px'},
                                                   style = {'description_width': '130px'})
            self.beta_slider.observe(self.on_beta_change, names='value')
            
            with self.out:
                plt.show()
            
            display(VBox([self.out, self.beta_slider]))
    
    def on_beta_change(self, change):
        self.hist_beta(change['new'], update=True)