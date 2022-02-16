import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets

class Recursive_filtering_demo():
    def __init__(self, q=0.8, a=0.6, r=0.6, noto=True):
        self.out = Output(layout={'width': '1000px', 'height': '600px'})
        self.axs = []
        self.noto = noto
        
        self.step_stage = 0
        
        self.x_color = (0.8, 0, 0, 1)
        self.x_color_light = (0.8, 0, 0, 0.3)
        self.y_color = (0, 0.5, 0, 1)
        self.y_color_light = (0, 0.5, 0, 0.3)
        self.y_highlight_color = (0.3, 1, 0.1, 1)
        
        self.m = 0
        
        self.q = q
        self.a = a
        self.r = r
        
        self.t = np.linspace(0, 15, 16)
        
        self.update_x()
        self.update_y()
        
        self.init_figure()
        
        self.button_left = widgets.Button(description='', icon='chevron-left', layout=Layout(width='40px'))
        self.button_left.on_click(self.button_left_callback)
        self.button_right = widgets.Button(description='', icon='chevron-right', layout=Layout(width='40px'))
        self.button_right.on_click(self.button_right_callback)
        self.spacing = widgets.HTMLMath('', layout=Layout(width='30px'))
        if self.noto:
            self.formula_text = widgets.HTMLMath('<h3>$x[k]=q^ku[k], \;\;\;y[m]=a_0x[m]+r_0y[m-1]$</h3>', layout=Layout(width='390px'))
            self.title_text = widgets.HTMLMath('<h2>Recursive Filtering: $m=0$</h2>', layout=Layout(width='400px'))
        else:
            self.formula_text = widgets.HTMLMath('<h5>$x[k]=q^ku[k], \;\;\;y[m]=a_0x[m]+r_0y[m-1]$</h5>', layout=Layout(width='390px'))
            self.title_text = widgets.HTMLMath('<h4>Recursive Filtering: $m=0$</h4>', layout=Layout(width='400px'))
        self.spacing2 = widgets.HTMLMath('', layout=Layout(width='464px'))
        self.button_reset = widgets.Button(description='Reset', icon='history', layout=Layout(width='85px'))
        self.button_reset.on_click(self.button_reset_callback)

        
        # Float widgets for q, a and r
        self.q_widget = widgets.BoundedFloatText(value=self.q, min=0, max=10.0, step=0.01, description=r'$q:$', layout=Layout(width='80px'), style = {'description_width': 'initial'})
        self.a_widget = widgets.BoundedFloatText(value=self.a, min=0, max=10.0, step=0.01, description=r'$a_0:$', layout=Layout(width='85px'), style = {'description_width': 'initial'})
        self.r_widget = widgets.BoundedFloatText(value=self.r, min=0, max=10.0, step=0.01, description=r'$r_0:$', layout=Layout(width='85px'), style = {'description_width': 'initial'})
        self.q_widget.observe(self.q_callback, names='value')
        self.a_widget.observe(self.a_callback, names='value')
        self.r_widget.observe(self.r_callback, names='value')
        
        self.update_plot()
        
        # Display
        display(VBox([HBox([self.formula_text, self.title_text, self.button_left, self.button_right]), HBox([self.q_widget, self.spacing, self.a_widget, self.spacing, self.r_widget, self.spacing2, self.button_reset]), self.out]))
        plt.tight_layout()
    
    def init_figure(self):
        with self.out:
            self.fig = plt.figure('Recursive filtering demo', figsize=(8.5, 4.5))
            self.gs = self.fig.add_gridspec(2, 1)

            # Plot x
            self.axs.append(self.fig.add_subplot(self.gs[0, 0]))
            self.axs[0].plot([-100, 100], [0, 0], linewidth=0.5, color='black')
            self.axs[0].plot([-2, -1], [0, 0], 'o', color=self.x_color_light)
            self.markerline_x, self.stemlines_x, self.baseline_x = self.axs[0].stem(self.x, use_line_collection=True)
            self.active_marker_x = self.axs[0].plot(0, 0, 'o', color=self.x_color)
            self.active_line_x = self.axs[0].plot([0, 0], [0, 1], color=self.x_color)
            self.axs[0].set_xlim([-3, 16])
            self.axs[0].set_xticks(np.linspace(-2, 15, 18))
            self.axs[0].set_title('x[k]')
            self.axs[0].set_xlabel('k')
            
            self.stemlines_x.set_color(self.x_color_light)
            self.markerline_x.set_color(self.x_color_light)
            self.baseline_x.set_visible(False)
            
            # Plot y
            self.axs.append(self.fig.add_subplot(self.gs[1, 0]))
            self.axs[1].plot([-100, 100], [0, 0], linewidth=0.5, color='black')
            self.axs[1].plot([-2, -1], [0, 0], 'o', color=self.y_color_light)
            self.markerline_y, self.stemlines_y, self.baseline_y = self.axs[1].stem(self.y, use_line_collection=True)
            self.active_marker_y = self.axs[1].plot(0, 0, 'o', color=self.y_color)
            self.active_line_y = self.axs[1].plot([0, 0], [0, 1], color=self.y_color)
            self.highlight_marker_y = self.axs[1].plot(0, 0, 'o', color=self.y_highlight_color)
            self.highlight_line_y = self.axs[1].plot([0, 0], [0, 1], color=self.y_highlight_color)
            self.txt = self.axs[1].text(len(self.y)-1, self.y[-1], '?', fontdict={'color': 'red', 'size': 20})
            self.axs[1].set_xlim([-3, 16])
            self.axs[1].set_xticks(np.linspace(-2, 15, 18))
            self.axs[1].set_title(f'y[{self.m}] = {self.a}x[{self.m}]+{self.r}y[{self.m-1}]')
            self.axs[1].set_xlabel('m')
            
            self.stemlines_y.set_color(self.y_color_light)
            self.markerline_y.set_color(self.y_color_light)
            self.baseline_y.set_visible(False)
            self.highlight_marker_y[0].set_visible(False)
            self.highlight_line_y[0].set_visible(False)
            plt.show()
            
            
    def update_x(self):
        self.x = self.q**self.t
    
    def update_y(self):
        self.y = np.zeros(len(self.x))
        self.y[0] = self.a*self.x[0]
        for i in range(1, len(self.x)):
            self.y[i] = self.a*self.x[i] + self.r*self.y[i-1]
    
    def set_active_pos_x(self, x, y):
        self.active_marker_x[0].set_data([x, y])
        self.active_line_x[0].set_data([x]*2, [0, y])
        
    def set_active_x_visible(self, visible):
        self.active_marker_x[0].set_visible(visible)
        self.active_line_x[0].set_visible(visible)
        
    def set_active_pos_y(self, x, y):
        self.active_marker_y[0].set_data([x, y])
        self.active_line_y[0].set_data([x]*2, [0, y])
    
    def set_active_y_visible(self, visible):
        self.active_marker_y[0].set_visible(visible)
        self.active_line_y[0].set_visible(visible)
    
    def set_highlight_pos_y(self, x, y):
        self.highlight_marker_y[0].set_data([x, y])
        self.highlight_line_y[0].set_data([x]*2, [0, y])
    
    def set_highlight_y_visible(self, visible):
        self.highlight_marker_y[0].set_visible(visible)
        self.highlight_line_y[0].set_visible(visible)
        
    def set_questionmark_pos(self, x, y):
        self.txt.set_position([x-0.2, y])
        
    def set_questionmark_visible(self, visible):
        self.txt.set_visible(visible)
        
    def update_plot(self):
        # Update title
        if self.noto:
            self.title_text.value = f'<h2>Recursive Filtering: $m={self.m}$</h2>'
        else:
            self.title_text.value = f'<h4>Recursive Filtering: $m={self.m}$</h4>'
        self.axs[1].set_title(f'y[{self.m}] = {self.a}x[{self.m}]+{self.r}y[{self.m-1}]')
        # Update x plot
        if self.m > 0:
            self.stemlines_x.set_segments([[[i, 0], [i, self.x[i]]] for i in range(len(self.x))])
            self.stemlines_x.set_visible(True)
            self.markerline_x.set_visible(True)
            colors = [self.x_color_light if i < self.m else (0, 0, 0, 0) for i in range(len(self.x))]
            self.stemlines_x.set_color(colors)
            self.markerline_x.set_data([list(range(self.m)), self.x[:self.m]])
        else:
            self.stemlines_x.set_visible(False)
            self.markerline_x.set_visible(False)
        self.set_active_pos_x(self.m, self.x[self.m])
        maxval = np.max(self.x[:self.m+1])
        self.axs[0].set_ylim([-0.05*maxval, maxval + 0.05*maxval])
        
        # Update y plot
        if self.m > 1:
            self.stemlines_y.set_segments([[[i, 0], [i, self.y[i]]] for i in range(len(self.y))])
            self.stemlines_y.set_visible(True)
            self.markerline_y.set_visible(True)
            colors = [self.y_color_light if i < self.m-1 else (0, 0, 0, 0) for i in range(len(self.y))]
            self.stemlines_y.set_color(colors)
            self.markerline_y.set_data([list(range(self.m-1)), self.y[:self.m-1]])
        else:
            self.stemlines_y.set_visible(False)
            self.markerline_y.set_visible(False)
        if self.m > 0:
            self.set_active_pos_y(self.m-1, self.y[self.m-1])
        else:
            self.set_active_pos_y(-1, 0)
            self.axs[1].set_ylim([-0.05, 1.05])
        self.set_highlight_pos_y(self.m, self.y[self.m])
        ylim = self.axs[1].get_ylim()
        self.set_questionmark_pos(self.m, (ylim[1] - ylim[0]) / 10)
        if self.step_stage == 0:
            self.set_questionmark_visible(True)
            self.set_highlight_y_visible(False)
        else:
            self.set_questionmark_visible(False)
            self.set_highlight_y_visible(True)
            maxval = np.max(self.y[:self.m+1])
            self.axs[1].set_ylim([-0.05*maxval, maxval + 0.05*maxval])
        
            
    def button_left_callback(self, value):
        if self.m > 0 :
            self.step_stage = 1
            self.m = self.m - 1
        else:
            self.step_stage = 0
        self.update_plot()
        
    
    def button_right_callback(self, value):
        if self.m < 15 or self.step_stage == 0:
            if self.step_stage == 1:
                self.step_stage = 0
                self.m = self.m + 1
            else:
                self.step_stage = 1
            self.update_plot()
            
    def button_reset_callback(self, value):
        self.m = 0
        self.step_stage = 0
        self.update_plot()
    
    def q_callback(self, value):
        self.q = value['new']
        self.update_x()
        self.update_y()
        self.update_plot()
        
    def a_callback(self, value):
        self.a = value['new']
        self.update_x()
        self.update_y()
        self.update_plot()
        
    def r_callback(self, value):
        self.r = value['new']
        self.update_x()
        self.update_y()
        self.update_plot()