import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 
from time import perf_counter_ns
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    from schemes.schemes import *

else:
    from .schemes.schemes import *

SOLVERS = {'lf': LaxFriedrich, 'lw': LaxWendroff, 'roe': Roe,
           'mc': MacCormack, 'flic': FLIC, 'muscl': MUSCL}

class HDSolver:
    def __init__(self, rho0, ux0, Pg0, E0, dx, x0, xf, nt, cfl_cut,
                 gamma, verbose):
        
        self.x0 = x0; self.xf = xf
        self.cfl = cfl_cut
        self.g = gamma 
        self.dx = dx 
        self.N = nt 
        self.t = np.zeros(nt)
        self.verbose = verbose

        ny, nx = rho0.shape

        self.rho = np.zeros((ny, nx, nt))
        self.ux = np.zeros((ny, nx, nt))
        self.Pg = np.zeros((ny, nx, nt))
        self.E = np.zeros((ny, nx, nt))

        self.rho[..., 0] = rho0 
        self.ux[..., 0] = ux0 
        self.Pg[..., 0] = Pg0 
        self.E[..., 0] = E0

class HDSolver2D(HDSolver):
    '''
    Solver for the 2D HD equations.

    Parameters
    ----------
    rho0, ux0, uy0, Pg0, E0, T0 : `ndarray`
        Initial conditions for density, horizontal and vertical
        velocity, gas pressure, total energy, and temperature. 

    dx, dy : `float`
        Spatial step lengths.

    x0, xf, y0, yf : `int` or `float`, optional 
        Corners of the simulation box. Default=[0, 1, 0, 1].

    force : `bool`, default=`False`
        Weather to add gravity force to the system.

    nt : `int`, default=100
        Number of time steps to simulate.

    cfl_cut : `float`, default=0.9
        CFL condition to limit the timestep.

    gamma : `float`, default=5/3
        Ratio of specific heats.

    solver : {'roe', 'mc', 'lf', 'lw', 'muscl', 'flic'}, default='roe'
        Solver/scheme to use. 

            * 'roe' : Roe-Pike
            * 'mc' : Mac Cormack
            * 'lf' : Lax-Friedrich
            * 'lw' : Lax-Wendroff
            * 'muscl' : MUSCL
            * 'flic' : Flux-limiter central 

    bc : {'periodic', 'constant', 'noslip'}, default='periodic'
        Boundary conditions.

    verbose : `bool`, default=`True`
        Output progression.
    '''
    def __init__(self, rho0: np.ndarray, ux0: np.ndarray,
                 uy0: np.ndarray, Pg0: np.ndarray, E0: np.ndarray,
                 T0: np.ndarray, dx: float, dy: float, x0=0, xf=1, y0=0,
                 yf=1, gravity=False, nt=100, cfl_cut=0.9, gamma=5/3,
                 solver='roe', bc='periodic', verbose=True, **kwargs):
        
        self.y0 = y0; self.yf = yf 
        self.dy = dy 
        self.solver = SOLVERS[solver](gamma, x0, xf, y0, yf, dx, dy,
                                      bc, gravity, **kwargs)
        
        self.scheme_name = self.solver.method

        ny, nx = rho0.shape
        x = np.linspace(x0, xf, nx)
        y = np.linspace(y0, yf, ny)
        self.x, self.y = np.meshgrid(x, y)
        self.r = np.sqrt(self.x**2 + self.y**2)

        super().__init__(rho0, ux0, Pg0, E0, dx, x0, xf, nt, cfl_cut,
                         gamma, verbose)
        
        self.uy = np.zeros_like(self.ux)
        self.T = np.zeros_like(self.ux)

        self.uy[..., 0] = uy0
        self.T[..., 0] = T0

        self._evolve()

    def _gravity(self, rho, M=1e5):
        G = 6.67e-11
        r, x, y = self.r, self.x, self.y
        F = -rho * G * M / r**2
        Fx = F * x / np.linalg.norm(r)
        Fy = F * y / np.linalg.norm(r)

        return Fx, Fy

    def _timestep(self, rho, ux, uy, Pg):
        dx, dy = self.dx, self.dy 
        cs = np.sqrt(self.g * Pg / rho)
        xterm = np.max(np.abs(ux) + cs) / dx 
        yterm = np.max(np.abs(uy) + cs) / dy 
        dt = self.cfl / (xterm + yterm)

        return dt 

    def _evolve(self):
        t1 = perf_counter_ns()
        for i in range(self.N - 1):
            r = self.rho[..., i]
            ux = self.ux[..., i]
            uy = self.uy[..., i]
            Pg = self.Pg[..., i]
            E = self.E[..., i]
            dt = self._timestep(r, ux, uy, Pg)
            rn, uxn, uyn, En, Pgn = self.solver.update(
                r, ux, uy, E, Pg, dt
            )
            
            self.rho[..., i+1] = rn 
            self.ux[..., i+1] = uxn 
            self.uy[..., i+1] = uyn 
            self.E[..., i+1] = En
            self.Pg[..., i+1] = Pgn
            self.T[..., i+1] = Pgn / (rn * self.g)
            self.t[i+1] = self.t[i] + dt 

            if np.isnan(dt):
                break

            if self.verbose:
                progr = (i+1) / (self.N-1) * 100 
                print(f' Progress: {progr:6.2f} % completed', end='\r')

        t2 = perf_counter_ns()
        if self.verbose:
            print('')
            time = (t2 - t1) * 1e-9
            print(f' Time elapsed: {time:.3f} seconds')
            
    def get_arrays(self) -> tuple:
        '''
        Return the arrays of the simulation. Note that if the simulation
        has generated negative pressure, the returns are cut to exclude
        `nan` values in the time array.

        Returns
        -------
        t : `array_like`
            Time array.

        rho, ux, uy, E, Pg, T : `ndarray`
            Density, horizontal and vertical velocity, total energy,
            gas pressure, and temperature arrays.
        '''
        if np.any(np.isnan(self.t)):
            idx = np.argwhere(np.isnan(self.t))[0][0]
            t = self.t[:idx]
            n = self.N - len(t)
            print(f'Warning: Excluding {n} NaN values out of {self.N} total.')

        else:
            idx = self.N
            t = self.t

        rho = self.rho[..., :idx]
        ux = self.ux[..., :idx]
        uy = self.uy[..., :idx]
        E = self.E[..., :idx]
        Pg = self.Pg[..., :idx]
        T = self.T[..., :idx]

        return t, rho, ux, uy, E, Pg, T

class HDSolver1D(HDSolver):
    def __init__(self, rho0, ux0, Pg0, E0, dx, x0=0, xf=1, nt=100,
                 cfl_cut=0.9, gamma=5/3, solver='roe',
                 bc='constant', verbose=False, **kwargs):
        
        kwargs['c'] = cfl_cut
        self.solver = SOLVERS[solver](gamma, x0, xf, 0, 1, dx, 1, bc,
                                      **kwargs)
        
        self.scheme_name = self.solver.method
        
        super().__init__(rho0, ux0, Pg0, E0, dx, x0, xf, nt, cfl_cut,
                         gamma, verbose)
        
        self._evolve()

    def _timestep(self, rho, ux, Pg):
        dx = self.dx 
        cs = np.sqrt(self.g * Pg / rho)
        dt = self.cfl * dx / np.max(np.abs(ux) + cs)

        return dt 

    def _evolve(self):
        for i in range(self.N - 1):
            r = self.rho[..., i]
            ux = self.ux[..., i]
            Pg = self.Pg[..., i]
            E = self.E[..., i]
            dt = self._timestep(r, ux, Pg)

            rn, uxn, _, En, Pgn = self.solver.update(
                r, ux, 0, E, Pg, dt
            )

            self.rho[..., i+1] = rn 
            self.ux[..., i+1] = uxn 
            self.Pg[..., i+1] = Pgn 
            self.E[..., i+1] = En 
            self.t[i+1] = self.t[i] + dt

    def get_arrays(self):
        if np.any(np.isnan(self.t)):
            idx = np.argwhere(np.isnan(self.t))[0][0]
            t = self.t[:idx]
            n = self.N - len(t)
            print(f'Warning: Excluding {n} NaN values out of {self.N} total.')

        else:
            idx = self.N
            t = self.t

        rho = self.rho[..., :idx]
        ux = self.ux[..., :idx]
        E = self.E[..., :idx]
        Pg = self.Pg[..., :idx]

        return t, rho, ux, E, Pg

def Pg4_func(Pg4, Pg1, Pg5, rho1, rho5, gamma):
    g = gamma 
    c1 = np.sqrt(g*Pg1 / rho1)
    c5 = np.sqrt(g*Pg5 / rho5)

    z = Pg4/Pg5 - 1
    a1 = (g-1) / (2*g) * c5/c1 * z / np.sqrt(1 + (g+1) / (2 * g) * z)
    a2 = (1 - a1)**(2*g / (g - 1))

    return Pg1 * a2 - Pg4

def sod_analytical(rhoL, rhoR, PgL, PgR, nx, t, uL=0, uR=0, gamma=5/3):
    g = gamma

    if PgL > PgR:
        rho1 = rhoL
        Pg1 = PgL 
        u1 = uL
        rho5 = rhoR 
        Pg5 = PgR
        u5 = uR 

    else:
        rho1 = rhoR
        Pg1 = PgR 
        u1 = uR
        rho5 = rhoL 
        Pg5 = PgL
        u5 = uL

    Pg4 = fsolve(Pg4_func, Pg1, (Pg1, Pg5, rho1, rho5, g))[0]

    z = Pg4/Pg5 - 1 
    c1 = np.sqrt(g*Pg1 / rho1)
    c5 = np.sqrt(g*Pg5 / rho5)

    b1 = 0.5 * (g-1) / g 
    b2 = 0.5 * (g+1) / g
    theta = np.sqrt(1 + b2*z)

    u4 = z * c5 / (g * theta)
    rho4 = rho5 * theta**2 / (1 + b1*z)

    w = c5 * theta 
    Pg3 = Pg4 
    u3 = u4 
    rho3 = rho1 * (Pg3/Pg1)**(1/g)
    c3 = np.sqrt(g*Pg3 / rho3)

    x = np.linspace(0, 1, nx)
    xmid = x[-1] / 2
    rho = np.zeros(nx)
    u = np.zeros(nx)
    Pg = np.zeros(nx)

    if PgL > PgR:
        x_ft = xmid + (u3 - c3) * t 
        x_hd = xmid - c1 * t 
        x_cd = xmid + u3 * t 
        x_sh = xmid + w * t 

        for i, xi in enumerate(x):
            if xi < x_hd:
                rho[i] = rho1
                Pg[i] = Pg1
                u[i] = u1
            elif xi < x_ft:
                u[i] = 2. / (g+1) * (c1 + (xi - xmid) / t)
                d = 1. - 0.5 * (g-1) * u[i] / c1
                rho[i] = rho1 * d ** (2. / (g-1))
                Pg[i] = Pg1 * d ** (2. * gamma / (g-1))
            elif xi < x_cd:
                rho[i] = rho3
                Pg[i] = Pg3
                u[i] = u3
            elif xi < x_sh:
                rho[i] = rho4
                Pg[i] = Pg4
                u[i] = u4
            else:
                rho[i] = rho5
                Pg[i] = Pg5
                u[i] = u5

    else:
        x_ft = xmid - (u3 - c3) * t 
        x_hd = xmid + c1 * t 
        x_cd = xmid - u3 * t 
        x_sh = xmid - w * t

        for i, xi in enumerate(x):
            if xi < x_sh:
                rho[i] = rho5
                Pg[i] = Pg5
                u[i] = -u1
            elif xi < x_cd:
                rho[i] = rho4
                Pg[i] = Pg4
                u[i] = -u4
            elif xi < x_ft:
                rho[i] = rho3
                Pg[i] = Pg3
                u[i] = -u3
            elif xi < x_hd:
                u[i] = -2. / (g+1) * (c1 + (xmid - xi) / t)
                d = 1. + 0.5 * (g-1) * u[i] / c1
                rho[i] = rho1 * d ** (2. / (g-1))
                Pg[i] = Pg1 * d ** (2. * gamma / (g-1))
            else:
                rho[i] = rho1
                Pg[i] = Pg1
                u[i] = -u1

    e = Pg / ((g-1) * rho) + 0.5 * u**2

    return rho, u, e, Pg

if __name__ == '__main__':
    from initial_condition import InitConds
    gamma = 5/3
    rhoL = 0.125
    rhoR = 1.0
    PgL = 0.125 / gamma
    PgR = 1.0 / gamma

    rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL

    nx = 500
    N = 250

    ic = InitConds(nx)
    ic.shocktube(rhoL, rhoR, PgL, PgR)

    dx, rho, ux, Pg, E = ic.get_ICs()

    HD = HDSolver1D(rho, ux, Pg, E, dx, nt=N)






