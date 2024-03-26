import numpy as np
from .schemes.schemes import *
from scipy.optimize import fsolve
import matplotlib.pyplot as plt 

SOLVERS = {'lf': LaxFriedrich, 'lw': LaxWendroff, 'roe': Roe,
           'mc': MacCormack, 'FLIC': FLIC, 'muscl': MUSCL}

class HDSolver:

    def __init__(self, nx, ny=1, lx=1, ly=1, gamma=5/3, nt=100,
                 cfl_cut=0.9, PgL=None, PgR=None, rhoL=None, rhoR=None,
                 rho0=1, Pg0=1, ux0=1, uy0=1, rho1=1e-11, rho2=1e-9,
                 KHI=False, n_regions=3, **kwargs):
        
        self.g = gamma 
        self.N = nt 
        self.cfl = cfl_cut
        self.nx = nx 
        self.ny = ny 
        self.dx = lx / nx 
        self.dy = ly / ny 

        self.t = np.zeros(nt)
        self.is2D = False
        self.scheme_name = None
        self._create_box()

        self.mu = 0.61
        self.m_u = 1.66e-27
        self.k_B = 1.38e-23

        if ny > 1:
            self.is2D = True

        if not self.is2D:
            self._set_initial_1D(rhoL, rhoR, PgL, PgR)

        elif not KHI:
            self._set_initial_2D(rho0, Pg0, ux0, uy0, **kwargs)

        else:
            self._set_initial_KHI(Pg0, rho0, rho1, rho2, n_regions)

    def _create_box(self):
            nx, ny, N = self.nx, self.ny, self.N

            self.rho = np.zeros((ny, nx, N))
            self.ux = np.zeros((ny, nx, N))
            self.uy = np.zeros((ny, nx, N))
            self.E = np.zeros((ny, nx, N))
            self.Pg = np.zeros((ny, nx, N))
            self.T = np.zeros((ny, nx, N))

    @staticmethod
    def _gauss(x, x0, y, y0, A=1, sigma=5, **kwargs):
        P1 = ((x - x0) / sigma)**2
        P2 = ((y - y0) / sigma)**2

        return A * np.exp(-0.5 * (P1 + P2))

    def _hyper_tan(self, y, A, HW=1, n_regions=3, **kwargs):
        if n_regions == 3:
            arg1 = y - self.ny/4
            arg2 = 3 * self.ny/4 - y 

            if y < self.ny/2:
                return A * 0.5 * (1 + np.tanh(arg1 / HW))
            
            else:
                return A * 0.5 * (1 + np.tanh(arg2 / HW))
            
        else:
            arg = y - self.ny/2

            return A * 0.5 * (1 + np.tanh(arg / HW))

    def _set_initial_1D(self, rhoL, rhoR, PgL, PgR):
        nx = self.nx
        inds = np.arange(self.nx)

        self.rho[..., 0] = np.where(inds < nx//2, rhoL, rhoR)
        self.Pg[..., 0] = np.where(inds < nx//2, PgL, PgR)
        self.E[..., 0] = self.Pg[..., 0] / ((self.g - 1) * self.rho[..., 0])

    def _set_initial_2D(self, rho0, Pg0, ux0, uy0, **kwargs):        
        nx, ny = self.nx, self.ny
        g, mu, m_u, k_B = self.g, self.mu, self.m_u, self.k_B
        x0, y0 = nx/2, ny/2

        if rho0 is None:
            rho0 = np.zeros((ny, nx)) + Pg0 * self.g

        else:
            rho0 = np.zeros((ny, nx)) + rho0

        for y in range(ny):
            for x in range(nx):
                rho0[y, x] += self._gauss(x, x0, y, y0, **kwargs)
        
        T0 = Pg0 / rho0 * mu * m_u / k_B
        e0 = k_B * T0 / (mu * m_u)
        
        self.rho[..., 0] = rho0
        self.ux[..., 0] = ux0
        self.uy[..., 0] = uy0
        self.Pg[..., 0] = Pg0 
        self.T[..., 0] = T0 
        self.E[..., 0] = e0 + 0.5 * (ux0**2 + uy0**2)

    def _set_initial_KHI(self, Pg0, rho0, rho1, rho2, n_regions):
        nx, ny = self.nx, self.ny
        g, mu, m_u, k_B = self.g, self.mu, self.m_u, self.k_B

        ux0 = np.zeros((ny, nx))
        uy0 = np.zeros((ny, nx))

        if n_regions == 2:
            rho0 = np.zeros((ny, nx)) + rho0
            Pg0 = np.zeros((ny, nx)) + Pg0

            for y in range(ny):
                ux0[y, :] += self._hyper_tan(y, 1, n_regions=n_regions)
                Pg0[y, :] -= self._hyper_tan(y, 0.1, n_regions=n_regions)

                for x in range(nx):
                    uy0[y, x] += self._gauss(x, nx/4, y, ny/2, 1, 5) \
                               + self._gauss(x, 3*nx/4, y, ny/2, 0.1, 10)

        elif n_regions == 3:
            dRho = rho2-rho1
            rho0 = np.zeros((ny, nx)) + rho2

            for y in range(ny):
                rho0[y, :] -= self._hyper_tan(y, dRho)
                ux0[y, :] += self._hyper_tan(y, 1e5)
                
                for x in range(nx):
                    uy0[y, x] += self._gauss(x, nx/2, y, ny/2, 2e5, 10)
        
        T0 = Pg0 / rho0 * mu * m_u / k_B
        e0 = k_B * T0/ (mu * m_u)

        self.rho[..., 0] = rho0
        self.ux[..., 0] = ux0
        self.uy[..., 0] = uy0
        self.Pg[..., 0] = Pg0
        self.T[..., 0] = T0 
        self.E[..., 0] = e0 + 0.5 * (ux0**2 + uy0**2)
    
    def _timestep(self, Pg, rho, ux, uy):
        dx, dy = self.dx, self.dy 

        cs = np.sqrt(self.g * Pg / rho)
        tmp_x = np.max(np.abs(ux) + cs) / dx

        if not self.is2D:
            dt = self.cfl / tmp_x

        else:
            tmp_y = np.max(np.abs(uy) + cs) / dy 
            dt = self.cfl / (tmp_x + tmp_y)

        return dt
    
    def evolve(self, method='lw', bc='constant', verbose=False, **kwargs):
        solver = SOLVERS[method](self.g, self.dx, self.dy, bc, **kwargs)

        self.scheme_name = solver.method

        for i in range(self.N - 1):
            r = self.rho[..., i]
            ux = self.ux[..., i]
            uy = self.uy[..., i]
            Pg = self.Pg[..., i]
            E = self.E[..., i]
            dt = self._timestep(Pg, r, ux, uy)

            rn, uxn, uyn, En, Pgn = solver.update(r, ux, uy, E, Pg,
                                                  dt)
            
            if self.is2D:
                self.T[..., i+1] = Pgn / (rn * self.g)
                
            self.rho[..., i+1] = rn
            self.ux[..., i+1] = uxn
            self.uy[..., i+1] = uyn
            self.E[..., i+1] = En
            self.Pg[..., i+1] = Pgn
            self.t[i+1] = self.t[i] + dt

            if np.isnan(dt):
                self.rho = self.rho[..., :i+2]
                self.ux = self.ux[..., :i+2]
                self.uy = self.uy[..., :i+2]
                self.E = self.E[..., :i+2]
                self.Pg = self.Pg[..., :i+2]
                self.T = self.T[..., :i+2]
                self.t = self.t[..., :i+2]
                break

            if verbose:
                perc = (i+1) / (self.N-1) * 100
                print(f'  Progress: {perc:6.2f} % completed', end='\r')

        if verbose:
            print('')

    def get_arrays(self):
        idx = np.isnan(self.t)
        t = self.t[~idx]

        if np.any(np.isnan(self.t)):
            n = self.N - len(t)
            print(f'Warning: Excluding {n} NaN values out of {self.N} total.')

        rho = self.rho[..., ~idx]
        ux = self.ux[..., ~idx]
        uy = self.uy[..., ~idx]
        E = self.E[..., ~idx]
        Pg = self.Pg[..., ~idx]
        T = self.T[..., ~idx]
        
        if self.is2D:
            return t, rho, ux, uy, E, Pg, T 
        
        else:
            return t, rho, ux, E, Pg
    
    def get_errors(self, metric='maxabs'):
        METRICS = {'maxabs': lambda x, y: np.max(np.abs(x - y)),
                   'square': lambda x, y: np.sum(x - y)**2 / y.size}
        
        err_metric = METRICS[metric]

        t, rho, u, e, Pg = self.get_arrays()
        rhoerr = np.zeros_like(t)
        uerr = np.zeros_like(t)
        eerr = np.zeros_like(t)
        Pgerr = np.zeros_like(t)

        for i in range(len(t)):
            rhoa, ua, ea, Pga = sod_analytical(self.rhoL, self.rhoR,
                                             self.PgL, self.PgR,
                                             self.Nx, t[i],
                                             gamma=self.g)
            
            rhoerr[i] = err_metric(rho[:, i], rhoa)
            uerr[i] = err_metric(u[:, i], ua)
            eerr[i] = err_metric(e[:, i], ea)
            Pgerr[i] = err_metric(Pg[:, i], Pga)

        return t, rhoerr, uerr, eerr, Pgerr

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

    


