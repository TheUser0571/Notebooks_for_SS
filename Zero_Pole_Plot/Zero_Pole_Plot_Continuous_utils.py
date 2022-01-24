import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets
from scipy import signal

class DraggableMarker():
    def __init__(self, ax=None, lines=None, update_func=None, update_conj=None):
        if ax == None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        if lines == None:
            self.lines = self.ax.lines
        else:
            self.lines = lines
        self.lines = self.lines[:]
        for line in self.lines:
            x, y = line.get_data()
            line.set_data(x.astype(np.float64), y.astype(np.float64))
        
        
        self.update_conj = update_conj
        self.update_func = update_func
        self.tx = self.ax.text(0, 0, "")
        self.active_point = 0
        self.active_line = 0
        self.draggable = False
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)

    def get_active_line(self):
        return self.active_line

    def get_active_point(self):
        return self.active_point

    def click(self, event):
        # Check for correct axes
        if event.inaxes == self.ax:
            if event.button == 1:
                # leftclick
                self.draggable = True
            elif event.button == 3:
                # rightclick
                self.draggable = False
                self.tx.set_visible(False)
                self.ax.figure.canvas.draw_idle()
                return

            self.active_point, self.active_line = self.get_closest(
                event.xdata, event.ydata)
            if self.active_point is None or self.active_line is None:
                self.draggable = False
                return

            self.tx.set_visible(True)
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self, event):
        self.draggable = False

    def update(self, event):
        self.tx.set_position((event.xdata, event.ydata))
        self.tx.set_text(f"Re: {event.xdata:.3f}\nIm: {event.ydata:.3f}")
        data_x, data_y = self.lines[self.active_line].get_data()
        data_x[self.active_point] = event.xdata
        data_y[self.active_point] = event.ydata
        self.lines[self.active_line].set_data(data_x, data_y)
        if self.update_func is not None:
            if self.real_filter and self.update_conj is not None:
                self.update_conj()
            else: 
                self.update_func()
            

        # Update transfer function

    def get_closest(self, mx, my):
        min_dist = np.iinfo(np.int64).max
        line_idx = None
        min_idx = None
        for i, line in enumerate(self.lines):
            x, y = line.get_data()
            # Check for empty lines
            if x.size == 0 or y.size == 0:
                continue
            dist = (x-mx)**2+(y-my)**2
            new_min_dist = np.min(dist)
            if new_min_dist < min_dist and new_min_dist < 0.0625:
                min_dist = new_min_dist
                min_idx = np.argmin(dist)
                line_idx = i

        return min_idx, line_idx


class DraggableZeroPolePlot(DraggableMarker):
    def __init__(self, zeros=1, poles=1, show_phase=None, show_dB=False, real_filter=True):
        self.out = Output(layout={'width': '980px', 'height': '450px'})
        self.axs = []
        #I change this because it is going to be the discrete zero pole plot demo
        self.discrete_mode = False#not continuous
        self.show_phase = True #not self.discrete_mode if show_phase is None else show_phase
        self.actual_change = True
        self.real_filter=real_filter
        self.show_dB = show_dB
        # Initialize zeros and poles
        z_x = []
        z_y = []
        p_x = []
        p_y = []

        for i in range(zeros):
            z_x.append(0.5)
            z_y.append(0)

        for i in range(poles):
            p_x.append(-0.5)
            p_y.append(0)

        self.collapsed_points = []

        # Initialize the figure with the initial number of zeros and poles
        self.init_figure(z_x, z_y, p_x, p_y)
        # Call the super class with the zero pole axis to enable the draggable markers
        super().__init__(ax=self.axs[0], lines=self.axs[0].lines[1:], update_func=self.change_freq_res, update_conj=self.update_conjugate)


        # Non causal text field
        self.tx_deb = self.axs[0].text(-1.75, 1.5, '', fontdict={'color': 'red', 'size': 12})
    
        # Debug text field
        self.tx_debug = self.axs[0].text(-1.75, 1, '', fontdict={'color': 'red', 'size': 15, 'font':'Arial'})

        # 'Calculation not possible' text fields
        self.cnp_gain = None
        self.cnp_ph = None

        # Text field numbers
        self.lastzeroRe = 0
        self.lastzeroIm = 0

        # Init frequency response plot
        self.change_freq_res(init=True)

        # Widgets
        # Zeros
        self.zero_range = widgets.IntSlider(value=zeros, min=0, max=zeros+10, step=1, description='Zeros:', continuous_update=False)

        self.zero_range.observe(self.on_zero_change, names='value')
        # Poles
        self.pole_range = widgets.IntSlider(value=poles, min=0, max=poles+10, step=1, description='Poles:', continuous_update=False)

        self.pole_range.observe(self.on_pole_change, names='value')

        # Check box to show phase plot
        self.phase_check = widgets.Checkbox(value=self.show_phase, description='Show phase')
        self.phase_check.observe(self.show_phase_callback, names='value')

        # Check box to show gain in dB
        self.dB_check = widgets.Checkbox(value=self.show_dB, description='dB')
        self.dB_check.observe(self.show_dB_callback, names='value')

        # Button to switch between continuous and discrete mode
        #self.mode_button = widgets.Button(description='Changer au cas continue' if self.discrete_mode else 'Changer au cas discret',
        #                                  layout=Layout(width='20%'))
        #self.mode_button.on_click(self.mode_button_callback)

        # Button to change to real filter
        self.real_button=widgets.Checkbox(value=self.real_filter, description = "Real filter", layout=Layout(width='50%'))
        
        self.real_button.observe(self.real_filter_callback, names='value')

        # Float text widgets
        self.input_Zero_RE = widgets.FloatText(value=self.lastzeroRe, description='Re:')
        self.input_Zero_RE.observe(self.Zero_RE_Caller, names='value')

        self.input_Zero_IM = widgets.FloatText(value=self.lastzeroIm, description='Im:')
        self.input_Zero_IM.observe(self.Zero_IM_Caller, names='value')


        self.collapsing = False
        # Display widgets and plot
        display(VBox([self.out,
                      HBox([self.zero_range, self.pole_range, self.phase_check]),
                      HBox([self.input_Zero_RE, self.input_Zero_IM, self.dB_check, self.real_button])]))
        plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)
        
        if self.real_filter:
            self.change_conjugate()

    def init_figure(self, z_x, z_y, p_x, p_y):
        with self.out:
            # Create the zero pole plot
            self.fig = plt.figure('Zero Pole Plot', figsize=(8, 4))
            self.gs = self.fig.add_gridspec(2, 2)
            self.axs.append(self.fig.add_subplot(self.gs[:, 0]))
            uc = self.unit_circle()
            # Draw unit circle
            self.axs[0].plot(uc[0], uc[1], color='black', linewidth='0.5')
            labels = ['-2j', '-1.5j', '-j', '-0.5j','0', '0.5j', 'j', '1.5j', '2j']
            position = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            self.axs[0].set_yticks(position)
            self.axs[0].set_yticklabels(labels)
            if not self.discrete_mode:
                self.axs[0].lines[0].set_visible(False)
            # Add zeros and poles
            self.axs[0].plot(z_x, z_y, 'ob', fillstyle='none', label='Zeros')
            self.axs[0].plot(p_x, p_y, 'xr', label='Poles')
            self.axs[0].set_xlim([-2, 2])
            self.axs[0].set_ylim([-2, 2])
            # Display the real and imaginary axes
            self.axs[0].set_yticks([1e-4], minor=True)
            self.axs[0].yaxis.grid(True, which='minor')
            self.axs[0].set_xticks([1e-4], minor=True)
            self.axs[0].xaxis.grid(True, which='minor')
            self.axs[0].set_title('Continuous zero-pole plot')
            self.axs[0].set_xlabel('Re')
            self.axs[0].set_ylabel('Im')
            # Enable the legend
            self.axs[0].legend()
            plt.show()

    #Callback function for the real filter
    def real_filter_callback(self, value):
        self.real_filter = value["new"]
        self.change_conjugate()
        self.change_freq_res()
        if self.real_filter: 
            self.pole_range.value = self.pole_range.value*2
            self.zero_range.value = self.zero_range.value*2
        elif not self.real_filter:
            self.pole_range.value = self.pole_range.value//2
            self.zero_range.value = self.zero_range.value//2

    def show_phase_callback(self, value):
        self.show_phase = value['new']
        self.change_freq_res(init=True, redraw=True)

    def show_dB_callback(self, value):
        self.show_dB = value['new']
        self.change_freq_res(init=True, redraw=True)

    def Zero_RE_Caller(self, change):
        if self.actual_change:
            x_min, x_max = self.axs[0].get_xlim()
            self.lastzeroRe = np.clip(change['new'], x_min, x_max)
            self.ChangeZero()
            if self.real_filter: self.update_conjugate()

    def Zero_IM_Caller(self, change):
        if self.actual_change:
            y_min, y_max = self.axs[0].get_ylim()
            self.lastzeroIm = np.clip(change['new'], y_min, y_max)
            self.ChangeZero()
            if self.real_filter: self.update_conjugate()

    def ChangeZero(self):
        l_x, l_y = self.axs[0].lines[self.active_line+1].get_data()
        l_x[self.active_point] = self.lastzeroRe
        l_y[self.active_point] = self.lastzeroIm
        self.axs[0].lines[self.active_line+1].set_data(l_x, l_y)
        self.tx.set_position((self.lastzeroRe, self.lastzeroIm))
        self.tx.set_text(f"Re: {self.lastzeroRe:.3f}\nIm: {self.lastzeroIm:.3f}")
        self.change_freq_res()

    def on_zero_change(self, change):
        self.active_line = 0                    
        if change['new'] < 0:
            change['new'] = 0
            self.zero_range.min = 0
            self.zero_range.value = 0
        num_zeros = change['new'] #if not self.real_filter else 2*change['new']

        if change['new'] < change['old']:
            while len(self.axs[0].lines[1].get_data()[0]) > num_zeros:
                if self.real_filter:
                    x, y = self.axs[0].lines[1].get_data()
                    n_tot = len(self.collapsed_points[self.active_line])
                    # Add all the conjugates
                    x = np.concatenate((x[:n_tot//2], np.flip(x[:n_tot//2])))
                    y = np.concatenate((y[:n_tot//2], np.flip(-y[:n_tot//2])))
                    # Remove the last pole and its conjugate
                    x = x[1:-1]
                    y = y[1:-1]
                    # Remove the two last positions of the collapsed points mask
                    self.collapsed_points[0] = self.collapsed_points[0][1:-1]
                    # Only draw the non-collapsed points
                    x = x[self.collapsed_points[self.active_line]==0]
                    y = y[self.collapsed_points[self.active_line]==0]
                    self.axs[0].lines[1].set_data(x, y)
                else:
                    x, y = self.axs[0].lines[1].get_data()
                    x = x[:-1]
                    y = y[:-1]
                    self.axs[0].lines[1].set_data(x, y)
        else:
            while len(self.axs[0].lines[1].get_data()[0]) < num_zeros:
                x, y = self.axs[0].lines[1].get_data()
                x = np.append(0.5, x)
                y = np.append(0, y)
                if self.real_filter:
                    self.collapsed_points[0] = np.append(0, self.collapsed_points[0])
                    self.collapsed_points[0] = np.append(self.collapsed_points[0], 1)

                self.axs[0].lines[1].set_data(x, y)

        # Set the correct number of poles in the slider
        self.zero_range.value = len(self.axs[0].lines[1].get_data()[0])                                          
        
        # Make sure to remove all collapsed points
        if self.real_filter and num_zeros == 0:
            self.collapsed_points[0] = []
        # Update frequency response plot
        self.change_freq_res()

    def on_pole_change(self, change):
        self.active_line = 1                             
        if change['new'] < 0:
            change['new'] = 0
            self.pole_range.min = 0
            self.pole_range.value = 0

        num_poles = change['new'] #if not self.real_filter else 2*change['new']

        if change['new'] < change['old']:
            while len(self.axs[0].lines[2].get_data()[0]) > num_poles:
                if self.real_filter:
                    x, y = self.axs[0].lines[2].get_data()
                    n_tot = len(self.collapsed_points[self.active_line])
                    # Add all the conjugates
                    x = np.concatenate((x[:n_tot//2], np.flip(x[:n_tot//2])))
                    y = np.concatenate((y[:n_tot//2], np.flip(-y[:n_tot//2])))
                    # Remove the last pole and its conjugate
                    x = x[1:-1]
                    y = y[1:-1]
                    # Remove the two last positions of the collapsed points mask
                    self.collapsed_points[1] = self.collapsed_points[1][1:-1]
                    # Only draw the non-collapsed points
                    x = x[self.collapsed_points[self.active_line]==0]
                    y = y[self.collapsed_points[self.active_line]==0]
                    self.axs[0].lines[2].set_data(x, y)
                else:
                    x, y = self.axs[0].lines[2].get_data()
                    x = x[:-1]
                    y = y[:-1]
                    self.axs[0].lines[2].set_data(x, y)
        else:
            while len(self.axs[0].lines[2].get_data()[0]) < num_poles:
                x, y = self.axs[0].lines[2].get_data()
                x = np.append(-0.5, x)
                y = np.append(0, y)
                if self.real_filter:
                    self.collapsed_points[1] = np.append(0, self.collapsed_points[1])
                    self.collapsed_points[1] = np.append(self.collapsed_points[1], 1)


                self.axs[0].lines[2].set_data(x, y)

        # Set the correct number of poles in the slider
        self.pole_range.value = len(self.axs[0].lines[2].get_data()[0])                                                                   
        # Make sure to remove all collapsed points
        if self.real_filter and num_poles == 0:
            self.collapsed_points[1] = []

        # Update frequency response plot
        self.change_freq_res()

    #Function to insert the conjugates when the real filter is switch on or remove the conjugates when it
    #is switch off
    def change_conjugate(self):
        if self.real_filter:
            for i in range(2):
                x_aux, y_aux = self.axs[0].lines[i+1].get_data()
                x_aux_aux = np.flip(x_aux)
                y_aux_aux = np.flip(y_aux)
                x_aux = np.concatenate((x_aux_aux, x_aux))
                y_aux = np.concatenate((y_aux_aux, -y_aux))
                self.axs[0].lines[i+1].set_data(x_aux, y_aux)
                self.collapsed_points.append(np.zeros(len(x_aux))) 

        else:
            for i in range (2):
                x_aux, y_aux = self.axs[0].lines[i+1].get_data()
                nl=len(x_aux)
                x_aux = x_aux[int(nl/2):nl]
                y_aux = y_aux[int(nl/2):nl]
                self.axs[0].lines[i+1].set_data(x_aux,y_aux)
            self.collapsed_points = [] 


    #Function that updates the conjugate of the point that is dragged
    def update_conjugate(self):
        x_aux, y_aux = self.axs[0].lines[self.active_line+1].get_data()
        nl = len(x_aux)

        n_collapsed_points = sum(self.collapsed_points[self.active_line])
        n_tot = len(self.collapsed_points[self.active_line])
        adjusted_active_point = 0
        zero_count = -1
        # Adjust the active point based on the number of collapsed points
        for i, value in enumerate(self.collapsed_points[self.active_line]):
            if value == 0:
                zero_count += 1
            if zero_count == self.active_point:
                adjusted_active_point = i
                break

        # Offset the adjusted active point to the upper half
        # We always want to remove the points from the upper half
        if adjusted_active_point < n_tot//2:
            adjusted_active_point += 2*(n_tot//2 - adjusted_active_point - 1) + 1
        else:
            # Swap the active point to the lower half
            new_active_point = adjusted_active_point - 2*(adjusted_active_point - n_tot//2) - 1
            x_aux[new_active_point] = x_aux[self.active_point]
            y_aux[new_active_point] = y_aux[self.active_point]
            self.active_point = new_active_point

        # Set the mask value
        if np.abs(y_aux[self.active_point]) < 0.05:
            # Bind the point to the x axis if close to 0
            y_aux[self.active_point] = 0
            # Update the text field correctly
            self.tx.set_position((x_aux[self.active_point], y_aux[self.active_point]))
            self.tx.set_text(f"Re: {x_aux[self.active_point]:.3f}\nIm: {y_aux[self.active_point]:.3f}")                                          
            if self.collapsed_points[self.active_line][adjusted_active_point] != 1:
                # Collapse points
                self.collapsed_points[self.active_line][adjusted_active_point] = 1
        else:
            if self.collapsed_points[self.active_line][adjusted_active_point] == 1:
                # Un-collapse points
                self.collapsed_points[self.active_line][adjusted_active_point] = 0

        x_aux = np.concatenate((x_aux[:n_tot//2], np.flip(x_aux[:n_tot//2])))[self.collapsed_points[self.active_line]==0]
        y_aux = np.concatenate((y_aux[:n_tot//2], np.flip(-y_aux[:n_tot//2])))[self.collapsed_points[self.active_line]==0]

        self.axs[0].lines[self.active_line+1].set_data(x_aux, y_aux)

        # Remove zero / pole from the slider
        if self.active_line == 0 and self.zero_range.value != len(self.axs[0].lines[self.active_line+1].get_data()[0]):
            if len(self.axs[0].lines[self.active_line+1].get_data()[0]) % 2 == 0:
                min_val = 0
            else:
                min_val = -1
            self.zero_range.min = min_val
            self.zero_range.value = len(self.axs[0].lines[self.active_line+1].get_data()[0])
        if self.active_line == 1 and self.pole_range.value != len(self.axs[0].lines[self.active_line+1].get_data()[0]):
            if len(self.axs[0].lines[self.active_line+1].get_data()[0]) % 2 == 0:
                min_val = 0
            else:
                min_val = -1
            self.pole_range.min = min_val
            self.pole_range.value = len(self.axs[0].lines[self.active_line+1].get_data()[0])

        self.change_freq_res()

    def change_freq_res(self, init=False, redraw=False):
        if self.cnp_gain is not None:
            self.cnp_gain.remove()
            self.cnp_gain = None
        if self.cnp_ph is not None:
            self.cnp_ph.remove()
            self.cnp_ph = None

        if init == True:
            # Generate the plots from scratch and name the axes
            with self.out:
                if redraw == True:
                    # Remove the gain and phase plots
                    for i in range(1, len(self.axs)):
                        self.axs[1].remove()
                        self.axs.pop(1)
                self.axs[0].lines[0].set_visible(self.discrete_mode)
                # Add gain (and phase) plot
                if self.show_phase:
                    self.axs.append(self.fig.add_subplot(self.gs[:1, 1]))
                    self.axs.append(self.fig.add_subplot(self.gs[1:, 1]))
                else:
                    self.axs.append(self.fig.add_subplot(self.gs[:, 1]))

        # Get zeros and poles from the zero pole plot
        z_re, z_im = self.axs[0].lines[1].get_data()
        z = z_re + 1j*z_im
        p_re, p_im = self.axs[0].lines[2].get_data()
        p = p_re + 1j*p_im
        # Calculate the gain (C)
        gaind = np.prod(1.+0.j-z) / np.prod(1.+0.j-p)
        gainc = np.prod(-z) / np.prod(-p)

        try:
            if self.discrete_mode:
                # Generate the transfer function
                H = signal.ZerosPolesGain(z, p, gaind, dt=0.1).to_tf()
                # Generate dicrete frequency response
                w, h = signal.freqz(H.num, H.den, whole=True)
                # Shift the angles to [-pi, pi]
                w = w-np.pi
                # Shift the gain and phase accordingly
                h_ph = np.fft.fftshift(np.angle(h, deg=True))
                h = np.abs(np.fft.fftshift(h))
                if self.show_dB:
                    h = 20*np.log10(h)
            else:
                # Generate the transfer function
                H = signal.ZerosPolesGain(z, p, gainc)
                # Generate the continuous frequency response
                w1, h11 = signal.freqresp(H, w=None, n=1000)
                h_ph1 = np.angle(h11, deg=True)
                h1 = np.abs(h11)
                w2 = np.flip(-w1)
                w2, h21 = signal.freqresp(H,w2, n=1000)
                h_ph2 = np.angle(h21, deg=True)
                h2 = np.abs(h21)
                w = np.concatenate((w2, w1))
                h = np.concatenate((h2, h1))


                if self.show_dB:
                    h = 20*np.log10(h)
                h_ph = np.concatenate((h_ph2, h_ph1))
        except ValueError:
            w = 1
            h = 1
            h_ph = 1
            self.calc_not_possible()

        if np.any(np.isinf(h)) or np.any(np.isnan(h)) or np.any(np.isinf(h_ph)) or np.any(np.isnan(h_ph)):
            w = 1
            h = 1
            h_ph = 1
            self.calc_not_possible()

        #This is to check if any pole is on/outside of the unit circle   
        if np.any(np.real(p) > 0):
            self.tx_deb.set_text("Filter is non causal")
        else:
            self.tx_deb.set_text("")

        if init == True:
            with self.out:
                # Gain
                self.axs[1].set_title('Frequency response')
                if self.discrete_mode:
                    self.axs[1].plot(w, h)
                else:
                    self.axs[1].plot(w, h)
                self.axs[1].set_xlabel('$\omega$ [rad]')
                self.axs[1].set_ylabel('|H($e^{j\omega}$)| [dB]' if self.show_dB else '|$H(e^{-j\omega})$|')
                # Phase
                if self.show_phase:
                    if self.discrete_mode:
                        self.axs[2].plot(w, h_ph)
                    else:
                        self.axs[2].plot(w, h_ph)
                    self.axs[2].set_xlabel('$\omega$ [rad]')
                    self.axs[2].set_ylabel('$\phi$($H(e^{-j\omega})$) [deg]')
                if self.discrete_mode:
                    positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
                    labels = ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$']
                    self.axs[1].set_xticks(positions)
                    self.axs[1].set_xticklabels(labels)
                    if self.show_phase:
                        self.axs[1].xaxis.set_visible(False)
                        positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
                        labels = ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$']
                        self.axs[2].set_xticks(positions)
                        self.axs[2].set_xticklabels(labels)
                else:
                    w_max = np.round(np.max(w)/np.pi)*np.pi
                    w_min = np.round(np.min(w)/np.pi)*np.pi
                    t = np.linspace(w_min, w_max, int((w_max - w_min)//np.pi) + 1)
                    t_labels = [f'{int(np.round(tick/np.pi))}$\pi$' if abs(int(np.round(tick/np.pi))) > 1 
                                else '$\pi$' if tick > 0 
                                else '$-\pi$' if int(np.round(tick/np.pi)) != 0 
                                else '0' for tick in t]
                    self.axs[1].set_xticks(t)
                    self.axs[1].set_xticklabels(t_labels)
                    if self.show_phase:
                        self.axs[1].xaxis.set_visible(False)
                        self.axs[2].set_xticks(t)
                        self.axs[2].set_xticklabels(t_labels)
        else:
            # Only change the values of the plots
            # Gain
            self.axs[1].lines[0].set_data(w, h)
            h_min = np.min(h)
            h_max = np.max(h)
            h_range = abs(h_max - h_min)
            self.axs[1].set_ylim(
                [h_min-0.05*h_range, h_max+0.05*h_range] if h_min != h_max else [h_min-1, h_max+1])
#             w_min = np.min(w)
#             w_max = np.max(w)
#             w_range = abs(w_max - w_min)
#             check = False if not self.discrete_mode and w_min < 0 else True
#             self.axs[1].set_xlim([w_min-0.05*w_range, w_max+0.05*w_range] if w_min != w_max and check else [w_min, w_max+1])
            # Phase
            if self.show_phase:
                h_ph_min = np.min(h_ph)
                h_ph_max = np.max(h_ph)
                h_ph_range = abs(h_ph_max - h_ph_min)
                self.axs[2].lines[0].set_data(w, h_ph)
                self.axs[2].set_ylim([h_ph_min-0.05*h_ph_range, h_ph_max+0.05*h_ph_range] if h_ph_min != h_ph_max else [h_ph_min-1, h_ph_max+1])
#                 self.axs[2].set_xlim([w_min-0.05*w_range, w_max+0.05*w_range] if w_min != w_max and check else [w_min, w_max+1])

            if self.active_line is not None:
                l_x, l_y = self.axs[0].lines[self.active_line + 1].get_data()
                if len(l_y) > self.active_point and len(l_x) > self.active_point:
                    self.actual_change = False
                    self.lastzeroRe = round(l_x[self.active_point], 3)
                    self.input_Zero_RE.value = self.lastzeroRe
                    self.lastzeroIm = round(l_y[self.active_point], 3)
                    self.input_Zero_IM.value = self.lastzeroIm
                    self.actual_change = True

    def calc_not_possible(self):
        self.cnp_gain = self.axs[1].text(0.5, 0.5, "Calculation not possible", fontdict={'color': 'red', 'size': 17},
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         transform=self.axs[1].transAxes)
        if self.show_phase:
            self.cnp_ph = self.axs[2].text(0.5, 0.5, "Calculation not possible", fontdict={'color': 'red', 'size': 17},
                                           horizontalalignment='center',
                                           verticalalignment='center',
                                           transform=self.axs[2].transAxes)


    def unit_circle(self):
        # Generate a unit circle plot
        x1 = np.linspace(-1, 1, 1000)
        y1 = np.sqrt(1-x1**2)
        x2 = np.linspace(0.999, -1, 1000)
        y2 = np.sqrt(1-x2**2)
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, -y2])
        return x, y