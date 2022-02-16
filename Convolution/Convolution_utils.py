import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets

class Convolution_demo():
    def __init__(self, n=10, a_h=1.3, a_g=1.1, delay=2000):
        self.out = Output(layout={'width': '1000px', 'height': '600px'})
        self.out_static = Output(layout={'width': '1000px', 'height': '300px'})
        self.axs = []
        
        self.n = n
        self.a_h = a_h
        self.a_g = a_g
        self.k = np.linspace(0, self.n-1, self.n)
        self.k_m = np.linspace(-self.n, self.n-1, 2*self.n)
        
        self.formula = r'$(h \ast g)[n] = \sum_{m=-\infty}^{\infty}h[m]g[n-m]$'
        
        self.update_signals(-1)
        
        # Inizializate the figure
        self.init_figure()

        # Play widget for animation
        self.play = widgets.Play(value=-1, min=-1, max=self.n-1, step=1, interval=delay, description="Play")
        # Slider widget for manual change
        self.slider = widgets.IntSlider(min=-1, max=self.n-1, description='n', style={'description_width':'50px'})
        # Link the two widgets
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))
        # Add callback function
        self.play.observe(self.update, names='value')
        # Float widgets for a_h and a_g
        self.a_h_widget = widgets.BoundedFloatText(value=self.a_h, min=0, max=10.0, step=0.01, description=r'$a_h:$', style={'description_width':'initial'}, layout={'width':'100px'})
        self.a_g_widget = widgets.BoundedFloatText(value=self.a_g, min=0, max=10.0, step=0.01, description=r'$a_g:$', style={'description_width':'initial'}, layout={'width':'100px'})
        self.a_h_widget.observe(self.a_h_callback, names='value')
        self.a_g_widget.observe(self.a_g_callback, names='value')
        # +1 button
        self.p1_button = widgets.Button(description='+1', layout={'width':'40px'})
        self.p1_button.on_click(self.p1_callback)
        
        # Display
        display(VBox([self.out_static, self.out, HBox([self.play, self.p1_button, self.slider, self.a_h_widget, self.a_g_widget])]))
    
    def p1_callback(self, value):
        self.slider.value = self.slider.value + 1
    
    def a_h_callback(self, value):
        self.a_h = value['new']
        self.update_signals(self.slider.value)
        # Update h
        self.axs[0].lines[0].set_data(self.k, self.h)
        for i, k in enumerate(self.k):
            self.axs[0].lines[i+1].set_data([k, k], [0, self.h[i]])
        # Update static h
        self.axs_static[0].lines[0].set_data(self.k, self.h)
        for i, k in enumerate(self.k):
            self.axs_static[0].lines[i+1].set_data([k, k], [0, self.h[i]])
        self.axs_static[0].set_ylim([-0.05*np.max(self.h), 1.05*np.max(self.h)])
        self.axs[0].set_ylim([-0.05*np.max(self.h), 1.05*np.max(self.h)])
        max_val_mult = self.get_max_val_mult()
        self.axs[2].set_ylim([-0.05*max_val_mult, 1.05*max_val_mult])
        self.axs[3].set_ylim([-0.05*np.max(self.conv), 1.05*np.max(self.conv)])
        self.update({'new':self.slider.value}, update_mult=False)
        
    def a_g_callback(self, value):
        self.a_g = value['new']
        self.update_signals(self.slider.value)
        # Update static h
        self.axs_static[1].lines[0].set_data(self.k, self.g)
        for i, k in enumerate(self.k):
            self.axs_static[1].lines[i+1].set_data([k, k], [0, self.g[i]])
        self.axs_static[1].set_ylim([-0.05*np.max(self.g), 1.05*np.max(self.g)])
        self.axs[1].set_ylim([-0.05*np.max(self.g), 1.05*np.max(self.g)])
        max_val_mult = self.get_max_val_mult()
        self.axs[2].set_ylim([-0.05*max_val_mult, 1.05*max_val_mult])
        self.axs[3].set_ylim([-0.05*np.max(self.conv), 1.05*np.max(self.conv)])
        self.update({'new':self.slider.value}, update_mult=False)
    
    def get_max_val_mult(self):
        max_val = 0
        for k_curr in range(-1, self.n):
            max_val = max(np.max(np.concatenate([np.zeros(self.n), self.h]) * np.concatenate([np.zeros(k_curr+1), np.flip(self.g), np.zeros(self.n-k_curr-1)])), max_val)
        return max_val if max_val > 0 else 1
    
    def init_figure(self):
        # Plot static plots of h and g
        with self.out_static:
            self.fig_static, self.axs_static = plt.subplots(1, 2, figsize=(8.5, 2.5), num='Static Plots')
            # Plot h
            self.axs_static[0].plot(self.k, self.h, 'o', color='red')
            for i, k in enumerate(self.k):
                self.axs_static[0].plot([k, k], [0, self.h[i]], color='red', linewidth=2)
            self.axs_static[0].set_title('h[k]')
            self.axs_static[0].set_xlabel('k')
            # Plot g
            self.axs_static[1].plot(self.k, self.g, 'o', color='green')
            for i, k in enumerate(self.k):
                self.axs_static[1].plot([k, k], [0, self.g[i]], color='green', linewidth=2)
            self.axs_static[1].set_title('g[k]')
            self.axs_static[1].set_xlabel('k')
            plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)
            plt.show()
        
        with self.out:
            self.fig = plt.figure('Convolution Demo', figsize=(8.5, 5.5))
            self.gs = self.fig.add_gridspec(4, 1)

            # Plot h
            ax_ind = 0
            self.axs.append(self.fig.add_subplot(self.gs[ax_ind, 0]))
            self.axs[ax_ind].plot(self.k, self.h, 'o', color='red')
            for i, k in enumerate(self.k):
                self.axs[ax_ind].plot([k, k], [0, self.h[i]], color='red', linewidth=2)
            self.axs[ax_ind].set_ylabel(r'$h[k]$')
            self.axs[ax_ind].set_title(r'$n=-1$')
            self.axs[ax_ind].set_xlabel(r'$k$')
            # Plot zeros
            self.axs[ax_ind].plot(np.linspace(-self.n, -1, self.n), np.zeros(self.n), 'o', color='red')
            
            # Plot g
            ax_ind=1
            self.axs.append(self.fig.add_subplot(self.gs[ax_ind, 0]))
            self.axs[ax_ind].plot(-self.k-1, self.g, 'o', color='green')
            for i, k in enumerate(-self.k-1):
                self.axs[ax_ind].plot([k, k], [0, self.g[i]], color='green', linewidth=2)
            self.axs[ax_ind].set_ylabel(r'$g^{\prime}[k]$')
            self.axs[ax_ind].set_xlabel(r'$k$')
            # Plot zeros
            self.axs[ax_ind].plot(np.linspace(0, self.n-1, self.n), np.zeros(self.n), 'o', color='green')
            
            # Plot mult
            ax_ind=2
            self.axs.append(self.fig.add_subplot(self.gs[ax_ind, 0]))
            self.axs[ax_ind].plot(self.k_m, self.mult, 'o', color='blue')
            for i, k in enumerate(self.k_m):
                self.axs[ax_ind].plot([k, k], [0, self.mult[i]], color='blue', linewidth=2)
            self.axs[ax_ind].set_ylabel(r'$(h \times g^{\prime})[k]$')
            max_val_mult = self.get_max_val_mult()
            self.axs[ax_ind].set_ylim([-0.05*max_val_mult, 1.05*max_val_mult])
            self.axs[ax_ind].set_xlabel(r'$k$')
            
            # Plot conv
            ax_ind=3
            self.axs.append(self.fig.add_subplot(self.gs[ax_ind, 0]))
            self.axs[ax_ind].set_ylabel(r'$(h \ast g)[-1]$')
            self.axs[ax_ind].set_ylim([-0.05*np.max(self.conv), 1.05*np.max(self.conv)])
            self.axs[ax_ind].set_xlabel(r'$n$')
            for i, k in enumerate(self.k):
                self.axs[ax_ind].plot(k, self.conv[i], 'o', color='gray')
                self.axs[ax_ind].plot([k, k], [0, self.conv[i]], color='gray', linewidth=2)
            # Plot zeros
            self.axs[ax_ind].plot(np.linspace(-self.n, -1, self.n), np.zeros(self.n), 'o', color='gray')
            
            # Text field for printing the formula
            self.txt1 = self.axs[ax_ind].text(-self.n, 3*self.axs[ax_ind].get_ylim()[1]/6, '', size=15)
            self.txt1.set_text(self.formula)
            
            for ax_ind in range(4):
                self.axs[ax_ind].set_xlim([-self.n-0.5, self.n-0.5])
                self.axs[ax_ind].set_xticks(np.linspace(-self.n, self.n-1, 2*self.n))
            plt.tight_layout(pad=0.1, w_pad=1.5, h_pad=0.1)
            plt.show()
            
    def update_signals(self, k_curr):
        self.h = self.a_h**self.k
        self.g = self.a_g**self.k
        self.mult = np.concatenate([np.zeros(self.n), self.h]) * np.concatenate([np.zeros(k_curr+1), np.flip(self.g), np.zeros(self.n-k_curr-1)])
        self.conv = np.convolve(self.h, self.g, mode='full')[:self.n]
        # Update conv lines
        if len(self.axs) > 3:
            for k in range(len(self.k)):
                # Two lines for each k
                self.axs[3].lines[2*k].set_data(k, self.conv[k])
                self.axs[3].lines[2*k+1].set_data([k, k], [0, self.conv[k]])

    def update(self, value, update_mult=True):
        k_curr = value['new']
        # Update title
        self.axs[0].set_title(r'$n=' + str(k_curr) + r'$')
        
        # Update g
        self.axs[1].lines[0].set_data(-self.k+k_curr, self.g)
        for i, k in enumerate(-self.k+k_curr):
            self.axs[1].lines[i+1].set_data([k, k], [0, self.g[i]])
        # Update zeros of g
        self.axs[1].lines[-1].set_data(np.concatenate([np.linspace(-self.n, -self.n+k_curr, k_curr+1), np.linspace(k_curr+1, self.n-1, self.n-k_curr-1)]), np.zeros(self.n))
        # Update mult
        if update_mult:
            self.mult = np.concatenate([np.zeros(self.n), self.h]) * np.concatenate([np.zeros(k_curr+1), np.flip(self.g), np.zeros(self.n-k_curr-1)])
        self.axs[2].lines[0].set_data(self.k_m, self.mult)
        for i, k in enumerate(self.k_m):
            self.axs[2].lines[i+1].set_data([k, k], [0, self.mult[i]])
        
        # Update conv
        self.axs[3].set_ylabel(r'$(h \ast g)[' + str(k_curr) + ']$')
        # Initialize all lines to invisible and gray
        for l in self.axs[3].lines:
            l.set_color('gray')
        if k_curr >=0:
            # Set the current line to black
            self.axs[3].lines[2*k_curr].set_color('red')
            self.axs[3].lines[2*k_curr+1].set_color('red')
        
        self.txt1.set_position((-self.n, 3*self.axs[3].get_ylim()[1]/6))