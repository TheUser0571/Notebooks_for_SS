import matplotlib.pyplot as plt
import numpy as np
import IPython
from scipy import signal
from scipy.io import wavfile
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets


class LPF():
    def __init__(self, filename='piano.wav'):
        self.SF, self.s = wavfile.read(filename)
        self.s = self.s/np.max(np.abs(self.s))
        print(f'Sampling frequency = {self.SF} Hz\nNumber of samples = {len(self.s)}')
        
        self.s_FT = np.abs(np.fft.fft(self.s))
        self.s_FT = self.s_FT / self.s_FT.max()
        self.w_FT = np.linspace(-self.SF//2, self.SF//2, len(self.s))
    
    def explore_signal(self):
        axs = []

        plt.close('all')

        fig = plt.figure('Signal exploration', figsize=(8, 6))
        gs = fig.add_gridspec(3, 1)
        axs.append(fig.add_subplot(gs[0, 0]))
        axs[0].plot(np.linspace(0, len(self.s)/self.SF, len(self.s)), self.s)
        axs[0].set_title('Sound wave')
        axs[0].set_xlabel('Time [s]')

        axs.append(fig.add_subplot(gs[1:, 0]))
        axs[1].plot(self.w_FT, np.fft.fftshift(self.s_FT))
        axs[1].set_title('Frequency spectrum')
        axs[1].set_ylabel('Normalized magnitude')
        axs[1].set_xlabel('f [Hz]')

        plt.tight_layout()
        plt.show()

        display(IPython.display.Audio(self.s, rate=self.SF))
    
    def remove_phase(self):
        # Remove phase information
        s_phaseless = np.fft.ifft(np.abs(np.fft.fft(self.s))).real

        plt.close('all')
        plt.figure('Removed phase', figsize=(8, 2.5))
        plt.plot(np.linspace(0, len(self.s)/self.SF, len(self.s)), self.s, label='Original')
        plt.plot(np.linspace(0, len(s_phaseless)/self.SF, len(s_phaseless)), s_phaseless, label='Removed phase', alpha=0.7)
        plt.title('Sound wave')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.legend()
        plt.show()

        # Create audio widgets
        orig_sound = Output(layout={'width': '320px', 'height': '60px'})
        orig_sound.append_display_data(IPython.display.Audio(self.s, rate=self.SF))
        filt_sound = Output(layout={'width': '320px', 'height': '60px'})
        filt_sound.append_display_data(IPython.display.Audio(s_phaseless, rate=self.SF))

        display(HBox([widgets.HTML('<h3>Original: </h3>'), orig_sound, widgets.HTML('<h3>Removed phase: </h3>'), filt_sound]))
        
    def modify_phase(self):
        # Create phase modification
        # ph = np.random.rand(len(self.s) // 2) * 2*np.pi * 1j
        # ph = np.concatenate(([0], ph, -ph[-2::-1]))
        ph = np.sin(np.linspace(0, 25000*np.pi, len(self.s)))

        # Add the phase offset
        s_random_phase = np.fft.ifft(np.fft.fft(self.s) * np.exp(ph)).real

        plt.close('all')
        plt.figure('Modified phase', figsize=(8, 2.5))
        plt.plot(np.linspace(0, len(self.s)/self.SF, len(self.s)), self.s, label='Original')
        plt.plot(np.linspace(0, len(s_random_phase)/self.SF, len(s_random_phase)), s_random_phase, label='Random phase', alpha=0.7)

        plt.title('Sound wave')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.legend()
        plt.show()

        # Create audio widgets
        orig_sound = Output(layout={'width': '320px', 'height': '60px'})
        orig_sound.append_display_data(IPython.display.Audio(self.s, rate=self.SF))
        filt_sound = Output(layout={'width': '320px', 'height': '60px'})
        filt_sound.append_display_data(IPython.display.Audio(s_random_phase, rate=self.SF))

        display(HBox([widgets.HTML('<h3>Original: </h3>'), orig_sound, widgets.HTML('<h3>Random phase: </h3>'), filt_sound]))


class DraggableMarker():
    def __init__(self, ax=None, lines=None, update_func=None):
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
        
        self.update_func = update_func
        self.tx = self.ax.text(0, 0, "")
        self.active_point = 0
        self.active_line = 0
        self.draggable = False
        self.c1 = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect(
            "button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect(
            "motion_notify_event", self.drag)

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
            
            # Don't move the original points but display their location
            if (self.active_line == 0 and self.active_point == 0) or (self.active_line == 1 and self.active_point < 2):
                self.draggable = False
                data_x, data_y = self.lines[self.active_line].get_data()
                event.xdata = data_x[self.active_point]
                event.ydata = data_y[self.active_point]
            
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
            self.update_func()

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


class LinearPhaseFilterExercise(DraggableMarker):
    def __init__(self):
        self.out = Output(layout={'width': '2000px', 'height': '450px'})
        self.axs = []
        #I change this because it is going to be the discrete zero pole plot demo
        self.discrete_mode = True#not continuous
        self.show_phase = True #not self.discrete_mode if show_phase is None else show_phase
        self.actual_change = True
        self.real_filter=False
        self.show_dB = False
        # Initialize zeros and poles
        z_x = [1/3]
        z_y = [-1/3]
        p_x = [4, 0]
        p_y = [0, 4/5]
            
        self.collapsed_points = []
            
        # Initialize the figure with the initial number of zeros and poles
        self.init_figure(z_x, z_y, p_x, p_y)
        # Call the super class with the zero pole axis to enable the draggable markers
        super().__init__(
            ax=self.axs[0], lines=self.axs[0].lines[1:], update_func=self.change_freq_res)

        # Non causal text field
        self.tx_deb = self.axs[0].text(-4.75, 4.2, '', fontdict={'color': 'red', 'size': 12})
        
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
        self.zero_range = widgets.IntSlider(value=1, min=1, max=2, step=1, description='Zeros:')

        self.zero_range.observe(self.on_zero_change, names='value')
        # Poles
        self.pole_range = widgets.IntSlider(value=2, min=2, max=4, step=1, description='Poles:')

        self.pole_range.observe(self.on_pole_change, names='value')

        # Check box to show phase plot
        self.phase_check = widgets.Checkbox(
            value=self.show_phase, description='Show phase')
        self.phase_check.observe(self.show_phase_callback, names='value')
               
        

        # Float text widgets
        self.input_Zero_RE = widgets.FloatText(
            value=self.lastzeroRe, description='Re:')
        self.input_Zero_RE.observe(self.Zero_RE_Caller, names='value')

        self.input_Zero_IM = widgets.FloatText(
            value=self.lastzeroIm, description='Im:')
        self.input_Zero_IM.observe(self.Zero_IM_Caller, names='value')

        self.collapsing = False
        
        # Display widgets and plot
        display(VBox([self.out,
                      HBox([self.zero_range, self.pole_range,
                             self.phase_check]),
                      HBox([self.input_Zero_RE, self.input_Zero_IM])]))
        plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)

    def init_figure(self, z_x, z_y, p_x, p_y):
        with self.out:
            # Create the zero pole plot
            self.fig = plt.figure('Zero Pole Plot', figsize=(8, 4))
            self.gs = self.fig.add_gridspec(2, 2)
            self.axs.append(self.fig.add_subplot(self.gs[:, 0]))
            uc = self.unit_circle()
            # Draw unit circle
            self.axs[0].plot(uc[0], uc[1], color='black', linewidth='0.5')
            labels = ['-5j', '-4j', '-3j', '-2j', '-j', '0', 'j', '2j', '3j', '4j', '5j']
            position = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
            self.axs[0].set_yticks(position)
            self.axs[0].set_yticklabels(labels)
            if not self.discrete_mode:
                self.axs[0].lines[0].set_visible(False)
            # Add zeros and poles
            self.axs[0].plot(z_x, z_y, 'ob', fillstyle='none', label='Zeros')
            self.axs[0].plot(p_x, p_y, 'xr', label='Poles')
            self.axs[0].set_xlim([-5, 5])
            self.axs[0].set_ylim([-5, 5])
            # Display the real and imaginary axes
            self.axs[0].set_yticks([1e-3], minor=True)
            self.axs[0].yaxis.grid(True, which='minor')
            self.axs[0].set_xticks([1e-3], minor=True)
            self.axs[0].xaxis.grid(True, which='minor')
            self.axs[0].set_title('Discrete zero-pole plot')
            self.axs[0].set_xlabel('Re')
            self.axs[0].set_ylabel('Im')
            # Enable the legend
            self.axs[0].legend()
            plt.show()

    def show_phase_callback(self, value):
        self.show_phase = value['new']
        self.change_freq_res(init=True, redraw=True)

    def Zero_RE_Caller(self, change):
        # Don't move the original points
        if (self.active_line == 0 and self.active_point == 0) or (self.active_line == 1 and self.active_point < 2):
            change['new'] = change['old']
        if self.actual_change:
            x_min, x_max = self.axs[0].get_xlim()
            self.lastzeroRe = np.clip(change['new'], x_min, x_max)
            self.ChangeZero()

    def Zero_IM_Caller(self, change):
        # Don't move the original points
        if (self.active_line == 0 and self.active_point == 0) or (self.active_line == 1 and self.active_point < 2):
            change['new'] = change['old']
        if self.actual_change:
            y_min, y_max = self.axs[0].get_ylim()
            self.lastzeroIm = np.clip(change['new'], y_min, y_max)
            self.ChangeZero()

    def ChangeZero(self):
        l_x, l_y = self.axs[0].lines[self.active_line+1].get_data()
        l_x[self.active_point] = self.lastzeroRe
        l_y[self.active_point] = self.lastzeroIm
        self.axs[0].lines[self.active_line+1].set_data(l_x, l_y)
        self.tx.set_position((self.lastzeroRe, self.lastzeroIm))
        self.tx.set_text(f"Re: {self.lastzeroRe:.3f}\nIm: {self.lastzeroIm:.3f}")
        self.change_freq_res()

    def on_zero_change(self, change):
        if change['new'] < 0:
            change['new'] = 0
            self.zero_range.min = 0
            self.zero_range.value = 0
        num_zeros = change['new'] #if not self.real_filter else 2*change['new']
        
        if change['new'] < change['old']:
            while len(self.axs[0].lines[1].get_data()[0]) > num_zeros:
                if self.real_filter:
                    x, y = self.axs[0].lines[1].get_data()
                    x = x[1:-1]
                    y = y[1:-1]
                    self.axs[0].lines[1].set_data(x, y)
                    self.collapsed_points[0] = self.collapsed_points[0][1:-1]
                else:
                    x, y = self.axs[0].lines[1].get_data()
                    x = x[:-1]
                    y = y[:-1]
                    self.axs[0].lines[1].set_data(x, y)
        else:
            while len(self.axs[0].lines[1].get_data()[0]) < num_zeros:
                x, y = self.axs[0].lines[1].get_data()
                x = np.append(x, 0.5)
                y = np.append(y, 0.5)
                if self.real_filter:
                    x = np.append(0.5, x)
                    y = np.append(-0.5, y)
                    self.collapsed_points[0] = np.append(0, self.collapsed_points[0])
                    self.collapsed_points[0] = np.append(self.collapsed_points[0], 0)

                self.axs[0].lines[1].set_data(x, y)
        
        # Make sure to remove all collapsed points
        if self.real_filter and num_zeros == 0:
            self.collapsed_points[0] = []
        
        # Update frequency response plot
        self.change_freq_res()

    def on_pole_change(self, change):
        if change['new'] < 0:
            change['new'] = 0
            self.pole_range.min = 0
            self.pole_range.value = 0
        num_poles = change['new'] #if not self.real_filter else 2*change['new']
        # Don't update if we are collapsing a point
        if change['new'] < change['old']:
            while len(self.axs[0].lines[2].get_data()[0]) > num_poles:
                if self.real_filter:
                    x, y = self.axs[0].lines[2].get_data()
                    x = x[1:-1]
                    y = y[1:-1]
                    self.axs[0].lines[2].set_data(x, y)
                    self.collapsed_points[1] = self.collapsed_points[1][1:-1]
                else:
                    x, y = self.axs[0].lines[2].get_data()
                    x = x[:-1]
                    y = y[:-1]
                    self.axs[0].lines[2].set_data(x, y)
        else:
            while len(self.axs[0].lines[2].get_data()[0]) < num_poles:
                x, y = self.axs[0].lines[2].get_data()
                nl = len(x)
                nlhalf = int(nl/2)
                x = np.append(x, 0.5)
                y = np.append(y, -0.5)
                if self.real_filter:
                    x = np.append(0.5, x)
                    y = np.append(0.5, y)
                    self.collapsed_points[1] = np.append(0, self.collapsed_points[1])
                    self.collapsed_points[1] = np.append(self.collapsed_points[1], 0)

                self.axs[0].lines[2].set_data(x, y)
                
        # Make sure to remove all collapsed points
        if self.real_filter and num_poles == 0:
            self.collapsed_points[1] = []
            
        # Update frequency response plot
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
        if np.any(np.abs(p)>=1):
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
                self.axs[1].set_ylabel('|H(z)| [dB]' if self.show_dB else '|H(z)|')
                # Phase
                if self.show_phase:
                    if self.discrete_mode:
                        self.axs[2].plot(w, h_ph)
                    else:
                        self.axs[2].plot(w, h_ph)
                    self.axs[2].set_xlabel('$\omega$ [rad]')
                    self.axs[2].set_ylabel('$\phi$(H(z)) [deg]')
                if self.discrete_mode:
                    positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
                    labels = ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$']
                    self.axs[1].set_xticks(positions)
                    self.axs[1].set_xticklabels(labels)
                    # Move y axis to the right
                    self.axs[1].yaxis.set_label_position("right")
                    self.axs[1].yaxis.tick_right()
                    if self.show_phase:
                        self.axs[1].xaxis.set_visible(False)
                        positions = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
                        labels = ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$']
                        self.axs[2].set_xticks(positions)
                        self.axs[2].set_xticklabels(labels)
                        # Move y axis to the right
                        self.axs[2].yaxis.set_label_position("right")
                        self.axs[2].yaxis.tick_right()
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
        self.cnp_gain = self.axs[1].text(0.5, 0.5, "Le calcul n'est pas possible", fontdict={'color': 'red', 'size': 17},
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         transform=self.axs[1].transAxes)
        if self.show_phase:
            self.cnp_ph = self.axs[2].text(0.5, 0.5, "Le calcul n'est pas possible", fontdict={'color': 'red', 'size': 17},
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