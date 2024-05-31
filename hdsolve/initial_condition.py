import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

class InitConds:
    '''
    Create initial conditions for the HD equations.

    Parameters
    ----------
    nx : `int`
        Points in the x direction.

    ny : `int`, default=1
        Points in the y direction.

    x0, xf, y0, yf : `int` or `float`, optional
        Corners of the simulation box. Default=[0, 1, 0, 1].

    gamma : `float`, default=5/3
        Ratio of specific heats.
    '''
    def __init__(self, nx: int, ny=1, x0=0, xf=1, y0=0, yf=1, gamma=5/3):
        self.nx = nx 
        self.ny = ny
        self.dx = (xf - x0) / nx
        self.dy = (yf - y0) / ny
        self.g = gamma 
        self.rotation = False
        self.contours = False

        x = np.linspace(x0, xf, nx)
        y = np.linspace(y0, yf, ny)
        self.x, self.y = np.meshgrid(x, y)
        self.r = np.sqrt(self.x**2 + self.y**2)

        self.extent = [x0, xf, y0, yf]

        self.figtitle = f'Initial condition\n$\\Delta x={self.dx:.4f}$'
        
        if ny > 1:
            self.is2D = True
            self.figtitle += f', $\\Delta y={self.dy:.4f}$'

        else:
            self.is2D = False

        self.mu = 0.61 
        self.m_u = 1.66e-27
        self.kB = 1.38e-23
        self.G = 6.67e-11

        self._create_box()

    def _create_box(self):
        nx, ny = self.nx, self.ny

        self.rho = np.zeros((ny, nx))
        self.ux = np.zeros((ny, nx))
        self.uy = np.zeros((ny, nx))
        self.E = np.zeros((ny, nx))
        self.Pg = np.zeros((ny, nx))
        self.T = np.zeros((ny, nx))

    def _shear(self, A, HW=1, **kwargs):
        nx, ny = self.nx, self.ny
        X = np.arange(nx)
        Y = np.arange(ny)
        x, y = np.meshgrid(X, Y)

        arg1 = y - ny/4
        arg2 = 3 * ny/4 - y

        shear = A * 0.5 * np.where(y < ny/2, 1 + np.tanh(arg1/HW),
                                   1 + np.tanh(arg2/HW))

        return shear
    
    def _gaussian(self, x0, y0, A, sigma):
        x, y = self.x, self.y 
        arg1 = ((x - x0) / sigma)**2
        arg2 = ((y - y0) / sigma)**2

        return A * np.exp(-0.5 * (arg1 + arg2))

    def shocktube(self, rhoL: float, rhoR: float,
                  PgL: float, PgR: float):
        '''
        Initiate the Sod shock tube.

        Parameters
        ----------
        rhoL, rhoR, PgL, PgR : `float`
            Left and right values for density and gas pressure.
        '''
        if self.is2D:
            raise ValueError(
                f"Only ny=1 is valid, not ny={self.ny}."
                )
        
        nx = self.nx 
        x = np.arange(nx)
        mid = nx // 2

        self.rho[:,:] = np.where(x < mid, rhoL, rhoR)
        self.Pg[:, :] = np.where(x < mid, PgL, PgR)
        self.E[:, :] = self.Pg / ((self.g - 1) * self.rho)

    def solar(self, Pg0: float, **kwargs):
        '''
        Initiate conditions similar to the solar corona and stratosphere.

        Parameters
        ----------
        Pg0 : `float`
            Gas pressure.
        '''
        mu, m_u, kB = self.mu, self.m_u, self.kB

        rho_corona = 1e-11
        rho_chromo = 1e-9
        drho = rho_chromo - rho_corona

        self.rho += rho_chromo - self._shear(drho, **kwargs)
        self.ux += self._shear(1e5, **kwargs)
        self.uy += self._gaussian(0.5, 0.5, 2.5e5, 0.05)
        self.Pg += Pg0 
        self.T += self.Pg / self.rho * mu * m_u / kB
        
        e = kB * self.T / (mu * m_u)
        V2 = self.ux**2 + self.uy**2

        self.E += e + 0.5 * V2

    def coll_discs(self, ux0=0, uy0=0, rho0=1, Pg0=1, T0=1, rot=False,
                   scale=0.1, gauss=False, A1=1, A2=1, sigma1=0.05,
                   sigma2=0.05, **kwargs):
        
        self.rho += rho0
        self.Pg += Pg0 

        if gauss:
            mu, m_u, kB = self.mu, self.m_u, self.kB
            x0, xf, y0, yf = self.extent

            x1 = x0 +(xf - x0) / 5
            x2 = x0 + 4*(xf - x0) / 5
            y1 = y0 + (yf - y0) / 2
            y2 = y1
            # y1 = y0 + (yf - y0) / 4 
            # y2 = y0 + 3*(yf - y0) / 4

            g1 = self._gaussian(x1, y1, A1, sigma1)
            g2 = self._gaussian(x2, y2, A2, sigma2)

            self.rho += g1 + g2
            self.T += self.Pg / self.rho * mu * m_u / kB
            e = kB * self.T / (mu * m_u)

        else:
            self.T += T0
            e = Pg0 / ((self.g - 1) * rho0) 


        if rot:
            self.rotation = True
            r, x, y = self.r, self.x, self.y
            Vx = -scale * y / r**2
            Vy = scale * x / r**2

            self.ux += Vx 
            self.uy += Vy 

        else:
            self.ux += ux0
            self.uy += uy0

        V2 = self.ux**2 + self.uy**2

        self.E += e + 0.5 * V2

    def riemann2D(self, Pgs: list, rhos: list, uxs: list, uys: list,
                  T0=1):
        
        self.contours = True
        x0, xf, y0, yf = self.extent

        x = np.linspace(x0, xf, self.nx)
        y = np.linspace(y0, yf, self.ny)

        xmid = x0 + (xf - x0) / 2
        ymid = y0 + (yf - y0) / 2

        bound_x = np.argwhere(x >= xmid)[0][0]
        bound_y = np.argwhere(y >= ymid)[0][0]

        self.Pg[bound_y:, bound_x:] = Pgs[0]
        self.rho[bound_y:, bound_x:] = rhos[0]
        self.ux[bound_y:, bound_x:] = uxs[0]
        self.uy[bound_y:, bound_x:] = uys[0]

        self.Pg[bound_y:, :bound_x] = Pgs[1]
        self.rho[bound_y:, :bound_x] = rhos[1]
        self.ux[bound_y:, :bound_x] = uxs[1]
        self.uy[bound_y:, :bound_x] = uys[1]

        self.Pg[:bound_y, :bound_x] = Pgs[2]
        self.rho[:bound_y, :bound_x] = rhos[2]
        self.ux[:bound_y, :bound_x] = uxs[2]
        self.uy[:bound_y, :bound_x] = uys[2]

        self.Pg[:bound_y, bound_x:] = Pgs[3]
        self.rho[:bound_y, bound_x:] = rhos[3]
        self.ux[:bound_y, bound_x:] = uxs[3]
        self.uy[:bound_y, bound_x:] = uys[3]

        self.T += T0
        
        e = self.Pg / ((self.g - 1) * self.rho)
        V2 = self.ux**2 + self.uy**2
        self.E += e + 0.5 * V2

    def gaussian_density(self, rho0=1, Pg0=1, ux0=1, uy0=1, x0=0.5,
                         y0=0.5, A=1, sigma=0.05):
        '''
        Initiate density with a gaussian.

        Parameters
        ----------
        rho0, Pg0, ux0, uy0 : `int` or `float`, optional
            Initial values for density, gas pressure, horizontal and
            vertical velocity. Default=[1, 1, 1, 1].

        x0, y0 : `int` or `float`, optional
            Center for the gaussian. Default=[0.5, 0.5].

        A : `int` or `float`, default=1
            Amplitude of the gaussian.

        sigma : `float`, default=0.05
            Standard deviation of the gaussian.
        '''
        mu, m_u, kB = self.mu, self.m_u, self.kB

        self.rho += rho0
        self.ux += ux0 
        self.uy += uy0
        self.Pg += Pg0 

        self.rho += self._gaussian(x0, y0, A, sigma)
        self.T += self.Pg / self.rho * mu * m_u / kB
        e = kB * self.T / (mu * m_u)
        V2 = self.ux**2 + self.uy**2

        self.E += e + 0.5 * V2

    def show_initial(self, fname=None, show=True, vel_cmap='bwr'):
        '''
        Show the initial condition.

        Parameters
        ----------
        fname : `str`, optional
            If given, the figure is saved as `fname`.

        show : `bool`, default=`True`
            Wheather to show the figure.

        vel_cmap : `str`, default='bwg'
            Color map to use on the velocity plots.
        '''
        if self.is2D:
            self._show2D(fname, show, vel_cmap)

        else:
            self._show1D(fname, show)

    def _show1D(self, fname, show):
        vars = [self.rho, self.ux, self.Pg, self.E]
        ylabs = [
            'Density', 'Horizontal velocity', 'Gas pressure', 
            'Total energy'
            ]
        xlabs = ['', '', r'$x$', r'$x$']
        x = np.linspace(0, 1, self.nx)

        fig, axes = plt.subplots(2, 2, figsize=(10.6, 6), sharex=True,
                                 dpi=200)

        for var, ax, xlab, ylab in zip( vars, axes.flat, xlabs, ylabs):
            ax.plot(x, var.T)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_xlim(0, 1)

        fig.suptitle(self.figtitle)

        if not fname is None:
            fig.savefig(fname, bbox_inches='tight')

        if show:
            plt.show()

    def _show2D(self, fname, show, vel_cmap):
        maps = ['plasma', vel_cmap, vel_cmap, 'plasma', 'plasma', 'plasma']
        titles = [
            'Density', 'Horizontal velocity', 'Vertical velocity',
            'Total energy', 'Gas pressure', 'Gas temperature'
        ]
        if self.rotation:
            vars = [np.log10(self.rho), self.ux, self.uy, np.log10(self.E), self.Pg, self.T]
            titles[0] += r' ($\log_{10}$)'
            titles[3] += r' ($\log_{10}$)'
        
        else:
            vars = [self.rho, self.ux, self.uy, self.E, self.Pg, self.T]
        labels = [
            ('', r'$y$'), ('', ''), ('', ''),
            (r'$x$', r'$y$'), (r'$x$', ''), (r'$x$', '')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(10.6, 6), sharex=True,
                                 sharey=True)
        
        for var, ax, title, lab, map in zip(vars, axes.flat, titles,
                                            labels, maps):
            
            xlab, ylab = lab
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_aspect('equal')
            ax.set_xlim(self.extent[:2])
            ax.set_ylim(self.extent[2:])

            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')

            im = ax.imshow(var, origin='lower', cmap=map,
                           extent=self.extent)
            
            fig.colorbar(im, cax=cax)

        if self.rotation:
            axes[0, 0].streamplot(self.x, self.y, self.ux, self.uy,
                                  density=0.5, linewidth=0.5,
                                  color='white')

        fig.suptitle(self.figtitle)

        if not fname is None:
            fig.savefig(fname, bbox_inches='tight')

        if show:
            plt.show()

    def get_ICs(self) -> tuple:
        '''
        Return the arrays for the initial condition.

        Returns
        -------
        dx, dy : `float`
            Spatial step length. `dy` is only returned if the simulation
            is 2D.

        rho, ux, uy, E, Pg, T : `ndarray`
            Arrays for density, horizontal and vertical (only if 2D) 
            velocity, total energy, gas pressure, and temperature (only
            if 2D).
        '''
        if self.is2D:
            return self.dx, self.dy, self.rho, self.ux, self.uy, self.E, self.Pg, self.T

        else:
            return self.dx, self.rho, self.ux, self.E, self.Pg

if __name__ == '__main__':
    plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})
    
    gamma = 5/3
    rhoL = 0.125
    rhoR = 1.0
    PgL = 0.125 / gamma
    PgR = 1.0 / gamma

    rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL
    nx = 500

    ic = InitConds(nx, nx)
    ic.solar(0.25)
    x = ic.get_ICs()