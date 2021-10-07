import numpy as np
import matplotlib.gridspec
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, IntSlider, HBox, Layout, Output, VBox
import ipywidgets as widgets
from scipy import signal

class Region_of_Convergence():
    def __init__(self, n=31):
        self.out = Output(layout={'width': '980px', 'height': '450px'})
        self.axs = []

        # Defining the boolean casual and the region(1:inside, 2:between and 3:outside)
        self.causal = True
        self.region = 1
        self.hover_region = None

        # Defining the ns, always has to be an odd number
        self.n = np.linspace(-n//2+1, n//2, n)

        # Defining the poles
        p_x = [0.25, 2/3]
        p_y = [0, 0]

        # Inizializate the figure
        self.init_figure(p_x, p_y)
        # Connect axes 0 to the mouse click event
        self.c1 = self.axs[0].figure.canvas.mpl_connect(
            "button_press_event", self.click)
        # Mouse motion (hover) event
        self.c2 = self.axs[0].figure.canvas.mpl_connect(
            "motion_notify_event", self.hover)
        # Text field
        self.tx = self.axs[0].text(-1.2, 1.2, '',
                                   fontdict={'color': 'red', 'size': 15, 'name': 'Arial'})

    def click(self, event):
        if event.button == 3:
            # rightclick
            self.tx.set_visible(False)
            self.axs[2].remove()
            self.axs[2] = self.fig.add_subplot(self.gs[1, -1])
            self.axs[2].set_title("h[n]")
            self.axs[2].set_xlabel('n')
            self.axs[2].set_xlim([np.min(self.n)-1, np.max(self.n)+1])
            self.axs[2].set_ylim([-1, 1])
            return
        if event.inaxes == self.axs[0]:
            if event.button == 1:
                # leftclick
                r = np.sqrt(event.xdata**2 + event.ydata**2)
                self.tx.set_visible(True)
                self.axs[2].remove()
                self.axs[2] = self.fig.add_subplot(self.gs[1, -1])
                if r < 0.25:
                    self.region = 1
                    self.axs[2].set_title("h[n] (Anticausal)")
                    self.tx.set_text('Unstable')
                elif r < 2/3:
                    self.region = 2
                    self.axs[2].set_title("h[n] (Neither)")
                    self.tx.set_text('Unstable')
                else:
                    self.region = 3
                    self.axs[2].set_title("h[n] (Causal)")
                    self.tx.set_text('Stable')
                # Update impulse response
                h = self.impulse_resp(self.n)
                self.axs[2].set_xlabel('n')
                self.axs[2].scatter(self.n, h, color='blue')
                if self.region == 3:
                    self.axs[2].set_ylim(np.array(self.axs[2].get_ylim())*1.5)
                for i, p in enumerate(h):
                    self.axs[2].plot([self.n[i], self.n[i]], [0, p], 'b')

                self.ax.figure.canvas.draw_idle()

    def init_figure(self, p_x, p_y):
        with self.out:
            # Create the Complex plane plot and small plots for H and h
            self.fig = plt.figure('Region of convergence demo', figsize=(8, 4))
            self.gs = self.fig.add_gridspec(2, 2)

            self.axs.append(self.fig.add_subplot(self.gs[:, 0]))
            self.axs[0].set_title("Zero Pole Plot")
            
            # Add ROC texts
            self.tx0 = self.axs[0].text(-1.2, 0.8, 'ROC', color='green', fontsize='large', visible=False)
            self.tx1 = self.axs[0].text(-0.4, 0.33, 'ROC', color='blue', fontsize='large', visible=False)
            self.tx2 = self.axs[0].text(-0.15, 0.05, 'ROC', color='red', fontsize='large', visible=False)

            self.axs.append(self.fig.add_subplot(self.gs[0, -1]))
            self.axs[1].set_title(r'$|H(e^{j\omega})|$')

            self.axs.append(self.fig.add_subplot(self.gs[1, -1]))
            self.axs[2].set_title("h[n]")
            self.axs[2].set_xlabel('n')
            self.axs[2].set_xlim([np.min(self.n)-1, np.max(self.n)+1])
            self.axs[2].set_ylim([-1, 1])

            # Defining the unit circle and regions
            self.uc = self.draw_circle(1)
            self.fc = self.draw_circle(0.25)
            self.sc = self.draw_circle(2/3)
            self.inf_circ = self.draw_circle(5)

            # Draw the unit circle
            self.axs[0].plot(self.uc[0], self.uc[1],
                             color='black', linewidth='0.5')

            # Draw ROC overlay
            self.axs[0].plot(self.fc[0], self.fc[1], ':', color='grey')
            self.axs[0].plot(self.sc[0], self.sc[1], ':', color='grey')

            labels = ['-2j', '-1.5j', '-j', '-0.5j',
                      '0', '0.5j', 'j', '1.5j', '2j']
            position = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            self.axs[0].set_yticks(position)
            self.axs[0].set_yticklabels(labels)

            # Add zeros and poles
            #self.axs[0].plot(z_x, z_y, 'xr', label='Zeros')
            self.axs[0].plot(p_x, p_y, 'xr', label='Poles')
            self.axs[0].set_xlim([-1.5, 1.5])
            self.axs[0].set_ylim([-1.5, 1.5])

            # Display the real and imaginary axes
            self.axs[0].set_yticks([1e-4], minor=True)
            self.axs[0].yaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xticks([1e-4], minor=True)
            self.axs[0].xaxis.grid(True, which='minor',
                                   color='black', linewidth='0.5')
            self.axs[0].set_xlabel('Re')
            self.axs[0].set_ylabel('Im')

            # Enable the legend
            self.axs[0].legend()

            # H
            w, h = self.H()
            self.axs[1].plot(w, h, color='blue')
            self.axs[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            self.axs[1].set_xticklabels(
                ['-$\pi$', '-$\dfrac{\pi}{2}$', '0', '$\dfrac{\pi}{2}$', '$\pi$'])
            self.axs[1].set_xlabel(r'$\omega$')
            plt.show()

        display(self.out)
        plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)

    def hover(self, event):
        if event.inaxes == self.axs[0]:
            r = np.sqrt(event.xdata**2 + event.ydata**2)
            if r < 0.25 and self.hover_region != 1:
                self.hover_region = 1
                self.axs[0].fill(self.inf_circ[0],
                                 self.inf_circ[1], facecolor='white')
                self.axs[0].fill(self.fc[0], self.fc[1],
                                 facecolor='lightsalmon')
                self.tx0.set_visible(False)
                self.tx1.set_visible(False)
                self.tx2.set_visible(True)
            elif r < 2/3 and r >= 0.25 and self.hover_region != 2:
                self.hover_region = 2
                self.axs[0].fill(self.inf_circ[0],
                                 self.inf_circ[1], facecolor='white')
                self.axs[0].fill(self.sc[0], self.sc[1],
                                 facecolor='lightskyblue')
                self.axs[0].fill(self.fc[0], self.fc[1], facecolor='white')
                self.tx0.set_visible(False)
                self.tx1.set_visible(True)
                self.tx2.set_visible(False)
            elif r >= 2/3 and abs(event.xdata) < 1.5 and abs(event.ydata) < 1.5 and self.hover_region != 3:
                self.hover_region = 3
                self.axs[0].fill(self.inf_circ[0],
                                 self.inf_circ[1], facecolor='white')
                self.axs[0].fill(self.inf_circ[0],
                                 self.inf_circ[1], facecolor='lightgreen')
                self.axs[0].fill(self.sc[0], self.sc[1], facecolor='white')
                self.tx0.set_visible(True)
                self.tx1.set_visible(False)
                self.tx2.set_visible(False)
        elif self.hover_region != 4:
            self.hover_region = 4
            self.axs[0].fill(self.inf_circ[0],
                             self.inf_circ[1], facecolor='white')
            self.tx0.set_visible(False)
            self.tx1.set_visible(False)
            self.tx2.set_visible(False)

    def H(self):
        # Generate the transfer function
        H = signal.ZerosPolesGain([], [0.25, 2/3], 0.25, dt=0.1).to_tf()
        # Generate dicrete frequency response
        w, h = signal.freqz(H.num, H.den, whole=True)
        # Shift the angles to [-pi, pi]
        w = w-np.pi
        # Shift the gain and phase accordingly
#         h_ph = np.fft.fftshift(np.angle(h, deg=True))
        h = np.abs(np.fft.fftshift(h))
        return w, h

    def step_f(self, positive, n):
        if positive:
            return np.where(n >= 0, 1, 0)
        else:
            return np.where(n < 0, 1, 0)

    def impulse_resp(self, n):
        if self.region == 1:
            # Anticausal
            return (-2/5*(2/3)**n+3/20*(1/4)**n)*self.step_f(False, n)
        elif self.region == 2:
            # Neither
            return -4/3*(2/3)**n*self.step_f(False, n)-3/20*(1/4)**n*self.step_f(True, n)
        else:
            # Causal
            return (2/5*(2/3)**n-3/20*(1/4)**n)*self.step_f(True, n)

    def draw_circle(self, r):
        x1 = np.linspace(-r, r, 1000)
        y1 = np.sqrt(r**2-x1**2)
        x2 = np.linspace(r-r/1000, -r, 1000)
        y2 = np.sqrt(r**2-x2**2)
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, -y2])
        return x, y