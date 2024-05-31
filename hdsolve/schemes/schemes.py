import numpy as np 
from numba import njit
from .roe_flux import roe_flux
from .muscl import reconstruct1D, build_flux1D, reconstruct2D, build_flux2D

class Schemes:
    r'''
    Parent class for the numerical schemes.

    Available schemes are:

        * Lax-Friedrich
        * Lax-Wendroff
        * Mac Cormack
        * Roe
        * Flux limiter central
        * MUSCL

    Parameters
    ----------
    gamma : `float`
        Ratio of specific heats.

    dx, dy : `float`
        Spatial resolution in the x and y directions.

    boundary_condition : `{'constant', 'periodic', 'noslip', 'transmissive'}`, default=`'constant'`
        Boundary condition for the simulation box.

            * `'constant'` : Set the boundaries to be constant from the
            initial condition.

            * `'periodic'` : Periodic boundary conditions.

            * `'noslip'` : Reflective walls.

            * `'transmissive'` : Transmissive walls.

    gravity : `bool`, default=`False`
        Add "gravity" to the system. 
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False):
        
        self.g = gamma 
        self.dx = dx
        self.dy = dy
        self.bc = boundary_condition
        self.gravity = gravity
        self.two_bc = False

        self.nx = int((xf - x0) / dx)
        self.ny = int((yf - y0) / dy)
        
        if dy < 1:
            self.is2D = True
            self.frac = 0.25
            self._step = self._step2D

            x = np.linspace(x0, xf, self.nx)
            y = np.linspace(y0, yf, self.ny)

            self.x, self.y = np.meshgrid(x, y)
            self.r = np.sqrt(self.x**2 + self.y**2)

        else:
            self.is2D = False 
            self.frac = 0.5
            self._step = self._step1D
            self.r, self.x, self.y = 1, 1, 1

    def _gravity(self, rho, axis, M=1, **kwargs):
        if self.gravity:
            G = 6.67e-11
            r, x, y = self.r, self.x, self.y
            
            F = -rho * M / r**2
            # F = -G * M / r**2
            Fx = np.zeros((4, self.ny, self.nx))
            Fy = np.zeros_like(Fx)

            Fx[1] = F * x / np.linalg.norm(r)
            # Fx[1] = F * x / np.abs(x)
            Fy[2] = F * y / np.linalg.norm(r)
            # Fy[2] = F * y / np.abs(x)

            if axis == -1:
                return Fx
            
            elif axis == -2:
                return Fy
            
        else:
            return 0

    def _step1D(self):
        raise NotImplementedError
    
    def _step2D(self):
        raise NotImplementedError

    def _get_variables(self, U):
        r'''
        Return the primitive variables.

        Parameters
        ----------
        U : `ndarray`
            Conserved variables.

        Returns
        -------
        rho, ux, uy, E : `ndarray`
            The primitive variabels for density, horizontal and vertical
            velocity, and internal energy.
        '''
        rho = U[0]
        ux = U[1] / rho
        uy = U[2] / rho 
        E = U[3] / rho

        return rho, ux, uy, E

    def _flux(self, U, Pg=None, axis='all'):
        r'''
        Create the flux vector.

        Parameters
        ----------
        U : `ndarray`
            Conserved variables.

        Pg : `ndarray` or `None`, default=`None`
            Gas pressure.

        axis : `int` or `str`, optional
            Return either the flux in the x direction or the y direction,
            or both (default).

        Returns
        -------
        Fx, Fy : `ndarray`
            The flux in the x and y directions.
        '''
        if Pg is None:
            Pg = self._EOS(U)

        rho, ux, uy, E = self._get_variables(U)

        # x-direction 
        F1x = rho * ux 
        F2x = rho * ux**2 + Pg 
        F3x = rho * ux * uy 
        F4x = (rho * E + Pg) * ux 
        Fx = np.array([F1x, F2x, F3x, F4x])

        # y-direction 
        F1y = rho * uy 
        F2y = rho * ux * uy 
        F3y = rho * uy**2 + Pg 
        F4y = (rho * E + Pg) * uy 
        Fy = np.array([F1y, F2y, F3y, F4y])

        if axis == 'all':
            return Fx, Fy
        
        elif axis == -2 or axis == 'y':
            return Fy 
        
        elif axis == -1 or axis == 'x':
            return Fx
    
    def _U(self, rho, ux, uy, E, set_initial=True):
        U = np.array([rho, rho * ux, rho * uy, rho * E])

        if set_initial:
            self.U0 = U[..., 0]
            self.U1 = U[..., -1]

        return U 
    
    def _set_bc(self, U):
        Utmp = U.copy()

        if self.bc == 'constant':
            if self.two_bc:
                Utmp[..., 1] = self.U0
                Utmp[..., -2] = self.U1 

            Utmp[..., 0] = self.U0
            Utmp[..., -1] = self.U1

        elif self.bc == 'transmissive':
            if self.is2D:
                Utmp[:, 0, :] = Utmp[:, 1, :]
                Utmp[:, -1, :] = Utmp[:, -2, :]
                
            Utmp[..., 0] = Utmp[..., 1]
            Utmp[..., -1] = Utmp[..., -2]

        elif self.bc == 'noslip':
            for i in [1, 2]:
                Utmp[i, :, 0] = 0
                Utmp[i, :, -1] = 0
                Utmp[i, 0, :] = 0
                Utmp[i, -1, :] = 0

            for i in [0, 3]:
                Utmp[i, :, 0] = (4 * Utmp[i, :, 1] - Utmp[i, :, 2]) / 3
                Utmp[i, :, -1] = (4 * Utmp[i, :, -2] - Utmp[i, :, -3]) / 3
                Utmp[i, 0, :] = (4 * Utmp[i, 1, :] - Utmp[i, 2, :]) / 3
                Utmp[i, -1, :] = (4 * Utmp[i, -2, :] - Utmp[i, -3, :]) / 3

        elif self.bc == 'periodic':
            for i in range(Utmp.shape[0]):
                pad = Utmp[i, 1:-1, 1:-1]
                Utmp[i] = np.pad(pad, [1, 1], mode='wrap')

        return Utmp
    
    @staticmethod
    def _diff3(phi, j):
        fwd = (4 * phi[j+1, :] - phi[j+2, :]) / 3
        bck = (4 * phi[j-1, :] - phi[j-2, :]) / 3

        if j == 0:
            return fwd 
        
        elif j == -1:
            return bck
    
    def _EOS(self, U):
        rho, ux, uy, E = self._get_variables(U)

        e = rho * (E - 0.5 * (ux**2 + uy**2))
        Pg = (self.g - 1) * e 

        return Pg 
    
    def _enthalpy(self, U):
        rho, ux, uy, E = self._get_variables(U)
        Pg = self._EOS(U)
        V2 = ux**2 + uy**2
        H = self.g / (self.g - 1) * Pg/rho + 0.5 * V2

        return H 
    
    def _step(self):
        raise NotImplementedError
    
    def update(self, rho, ux, uy, E, Pg, dt, **kwargs):
        Utmp = self._step(rho, ux, uy, E, Pg, dt, **kwargs)
        Unew = self._set_bc(Utmp)

        rhon, uxn, uyn, En = self._get_variables(Unew)
        Pgn = self._EOS(Unew)

        return rhon, uxn, uyn, En, Pgn

    def _Roe_variables(self, UL, axis=None, UR=None):
        if UR is None:
            UR = np.roll(UL, -1, axis=axis)

        rhoL, uxL, uyL, _ = self._get_variables(UL)
        rhoR, uxR, uyR, _ = self._get_variables(UR)
        HL = self._enthalpy(UL)
        HR = self._enthalpy(UR)

        drho = rhoR - rhoL 
        dux = uxR - uxL
        duy = uyR - uyL 

        den = np.sqrt(rhoL) + np.sqrt(rhoR)
        rho = np.sqrt(rhoL * rhoR)
        ux = (np.sqrt(rhoL) * uxL + np.sqrt(rhoR) * uxR) / den 
        uy = (np.sqrt(rhoL) * uyL + np.sqrt(rhoR) * uyR) / den
        V2 = ux**2 + uy**2
        H = (np.sqrt(rhoL) * HL + np.sqrt(rhoR) * HR) / den 
        c = np.sqrt((self.g - 1) * (H - 0.5 * V2))

        return rho, drho, ux, dux, uy, duy, H, c, V2

    def _Roe_flux(self, UL, PgL, axis, UR=None, PgR=None):
        nx, ny = UL.shape[2], UL.shape[1]

        if UR is None and PgR is None:
            UR = np.roll(UL, -1, axis=axis)
            PgR = np.roll(PgL, -1, axis=axis)

        FL = self._flux(UL, PgL, axis=axis)
        FR = self._flux(UR, PgR, axis=axis)
        dF = 0.5 * (FL + FR)

        R, dR, Ux, dUx, Uy, dUy, H, C, V2 = self._Roe_variables(UL, axis, UR)
        dPg = PgR - PgL

        FRoe = roe_flux(nx, ny, dF, R, dR, Ux, dUx, Uy, dUy, dPg,
                        H, C, V2, axis)

        return FRoe
    
class Roe(Schemes):
    r'''
    The Roe-Pike scheme. 
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False, M=1,
                 **kwargs):
        
        self.M = M
        self.method = 'Roe'
        super().__init__(gamma, x0, xf, y0, yf, dx, dy,
                         boundary_condition, gravity)

    def _step1D(self, rho, ux, uy, E, Pg, dt):
        dx = self.dx 
        U = self._U(rho, ux, uy, E)
        FRoexP = self._Roe_flux(U, Pg, -1)

        Uxm = np.roll(U, 1, axis=-1)
        Pgxm = self._EOS(Uxm)
        FRoexM = self._Roe_flux(Uxm, Pgxm, -1)
        Force_x = self._gravity(rho, axis=-1, M=self.M)

        dFx = FRoexP - FRoexM
        Un = U - dt * (dFx/dx - Force_x)

        return Un 
    
    def _step2D(self, rho, ux, uy, E, Pg, dt):
        dy = self.dy
        U = self._U(rho, ux, uy, E)
        GRoeyP = self._Roe_flux(U, Pg, -2)

        Uym = np.roll(U, 1, -2)
        Pgym = self._EOS(Uym)
        GRoeyM = self._Roe_flux(Uym, Pgym, -2)
        Force_y = self._gravity(rho, axis=-2, M=self.M)

        dFy = GRoeyP - GRoeyM
        Un = self._step1D(rho, ux, uy, E, Pg, dt)
        Unn = Un - dt * (dFy/dy - Force_y)

        return Unn
    
class LaxFriedrich(Schemes):
    r'''
    The Lax-Friedrich scheme.
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False, **kwargs):
        
        self.method = 'Lax-Friedrich'
        super().__init__(gamma, x0, xf, y0, yf, dx, dy,
                         boundary_condition, gravity)

    def _step1D(self, rho, ux, uy, E, Pg, dt):
        dx = self.dx 
        U = self._U(rho, ux, uy, E)

        Umin = np.roll(U, 1, axis=-1)
        Uplus = np.roll(U, -1, axis=-1)
        Fmin = self._flux(Umin, axis=-1)
        Fplus = self._flux(Uplus, axis=-1)

        Un = self.frac * (Umin + Uplus) - 0.5 * dt/dx * (Fplus - Fmin)

        return Un 
    
    def _step2D(self, rho, ux, uy, E, Pg, dt):
        dy = self.dy
        U = self._U(rho, ux, uy, E)

        Umin = np.roll(U, 1, axis=-2)
        Uplus = np.roll(U, -1, axis=-2)
        Fmin = self._flux(Umin, axis=-2)
        Fplus = self._flux(Uplus, axis=-2)

        Un = self._step1D(rho, ux, uy, E, Pg, dt)
        Unn = Un + 0.25 * (Umin + Uplus) - 0.5 * dt/dy * (Fplus - Fmin)

        return Unn
    
class LaxWendroff(Schemes):
    r'''
    The Lax-Wendroff scheme.
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False, **kwargs):
        
        self.method = 'Lax-Wendroff'
        super().__init__(gamma, x0, xf, y0, yf, dx, dy,
                         boundary_condition, gravity)

    def _step1D(self, rho, ux, uy, E, Pg, dt):
        dx = self.dx 
        U = self._U(rho, ux, uy, E)
        Fx = self._flux(U, axis=-1)

        UL = np.roll(U, 1, axis=-1)
        UR = np.roll(U, -1, axis=-1)
        FL = np.roll(Fx, 1, axis=-1)
        FR = np.roll(Fx, -1, axis=-1)

        UW = self.frac * (UL + U) - 0.5 * dt/dx * (Fx - FL)
        UE = self.frac * (UR + U) - 0.5 * dt/dx * (FR - Fx)

        FW = self._flux(UW, axis=-1)
        FE = self._flux(UE, axis=-1)

        Un = U - dt/dx * (FE - FW)

        return Un
    
    def _step2D(self, rho, ux, uy, E, Pg, dt):
        dy = self.dy
        U = self._U(rho, ux, uy, E)
        Fy = self._flux(U, axis=-2)

        UD = np.roll(U, 1, axis=-2)
        UU = np.roll(U, -1, axis=-2)
        FD = np.roll(Fy, 1, axis=-2)
        FU = np.roll(Fy, -1, axis=-2)

        US = self.frac * (UD + U) - 0.5 * dt/dy * (Fy - FD)
        UN = self.frac * (UU + U) - 0.5 * dt/dy * (FU - Fy)

        FS = self._flux(US, axis=-2)
        FN = self._flux(UN, axis=-2)

        Un = self._step1D(rho, ux, uy, E, Pg, dt)
        Unn = Un - dt/dy * (FN - FS)

        return Unn

class MacCormack(Schemes):
    r'''
    The Mac Cormack scheme.
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False, **kwargs):
        
        self.method = 'MacCormack'
        super().__init__(gamma, x0, xf, y0, yf, dx, dy,
                         boundary_condition, gravity)

    def _step1D(self, rho, ux, uy, E, Pg, dt):
        dx = self.dx 
        pos_x = np.all(ux >= 0)

        U = self._U(rho, ux, uy, E)
        Fx = self._flux(U, axis=-1)

        FL = np.roll(Fx, 1, axis=-1)
        FR = np.roll(Fx, -1, axis=-1)

        fwd_x = FR - Fx 
        bck_x = Fx - FL

        dFx1 = pos_x * fwd_x + (not pos_x) * bck_x
        Uxn = U - dt/dx * dFx1
        Fxn = self._flux(Uxn, axis=-1)

        FLn = np.roll(Fxn, 1, axis=-1)
        FRn = np.roll(Fxn, -1, axis=-1)

        fwd_xn = FRn - Fxn 
        bck_xn = Fxn - FLn 

        dFx = pos_x * bck_xn + (not pos_x) * fwd_xn
        Un = self.frac * (Uxn + U - dt/dx * dFx)

        return Un
    
    def _step2D(self, rho, ux, uy, E, Pg, dt):
        dy = self.dy
        pos_y = np.all(uy >= 0)

        U = self._U(rho, ux, uy, E)
        Fy = self._flux(U, axis=-2)

        FS = np.roll(Fy, 1, axis=-2)
        FN = np.roll(Fy, -1, axis=-2)

        fwd_y = FN - Fy 
        bck_y = Fy - FS 

        dFy1 = pos_y * fwd_y + (not pos_y) * bck_y
        Uyn = U - dt/dy * dFy1
        Fyn = self._flux(Uyn, axis=-2)
        
        FSn = np.roll(Fyn, 1, axis=-2)
        FNn = np.roll(Fyn, -1, axis=-2)

        fwd_yn = FNn - Fyn 
        bck_yn = Fyn - FSn 

        dFy = pos_y * bck_yn + (not pos_y) * fwd_yn
        Un = self._step1D(rho, ux, uy, E, Pg, dt)
        Unn = Un + self.frac * (U + Uyn - dt/dy * dFy)

        return Unn
        
class FLIC(Schemes):
    '''
    The Flux LImiter Center scheme.
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False,
                 limit_func='superbee', epsilon=1e-8, c=0.9, **kwargs):
        
        self.method = 'FLIC+' + limit_func
        self.type = limit_func
        self.eps = epsilon
        self.phi_g = (1 - c) / (1 + c)
        super().__init__(gamma, x0, xf, y0, yf, dx, dy,
                         boundary_condition, gravity)
    
    def _HO_flux(self, Uprev, U, Unext, Fprev, F, Fnext, dt, axis):
        '''
        Hig-order flux
        '''
        if axis == -1:
            dxy = self.dx

        elif axis == -2:
            dxy = self.dy

        UL = 0.5 * (U + Uprev - dt/dxy * (F - Fprev))
        UR = 0.5 * (U + Unext - dt/dxy * (Fnext - F))

        FL = self._flux(UL, axis=axis)
        FR = self._flux(UR, axis=axis)

        return FL, FR
    
    def _LO_flux(self, Uprev, U, Unext, Fprev, F, Fnext, dt, axis):
        '''
        Low-order flux
        '''
        if axis == -1:
            dxy = self.dx

        elif axis == -2:
            dxy = self.dy

        LF_L = 0.5 * (F + Fprev - dxy/dt * (U - Uprev))
        LF_R = 0.5 * (F + Fnext - dxy/dt * (Unext - U))

        # Richtmyer flux
        Ri_L, Ri_R = self._HO_flux(Uprev, U, Unext, Fprev, F, Fnext,
                                   dt, axis)
        
        # First-order central flux
        FL = 0.5 * (LF_L + Ri_L)
        FR = 0.5 * (LF_R + Ri_R)

        return FL, FR
    
    def _phi(self, U, axis):
        phi_g = self.phi_g

        Up = np.roll(U, -1, axis=axis)
        Um = np.roll(U, 1, axis=axis)

        num = U - Um 
        den = np.where(np.abs(Up - U) < self.eps, self.eps, Up - U)
        r = num / den

        def superbee(r):
            phi = np.where(r>0, 2*r, 0)
            phi[r>0.5] = np.where(
                r[r>0.5]<=1, 1,
                np.minimum(2, phi_g*(1-phi_g)*r[r>0.5])
            )

            return phi
        
        def vanleer(r):
            phi = np.where(r>0, 2*r/(1+r), 0)
            phi[r>1] = phi_g+(2*(1-phi_g)*r[r>1])/(1+r[r>1])

            return phi
        
        def vanalbada(r):
            phi = np.where(r>0, r*(1+r)/(1+r**2), 0)
            phi[r>1] = phi_g + (1-phi_g)*r[r>1]*(1+r[r>1])/(1+r[r>1]**2)

            return phi
        
        def minbee(r):
            phi = np.zeros_like(r)
            phi[r>0] = np.where(r[r>0]<=1, r[r>0], 1)

            return phi

        
        LIMITERS = {'superbee': superbee, 'vanleer': vanleer,
                    'vanalbada': vanalbada, 'minbee': minbee}

        limiter = LIMITERS[self.type](r)

        return limiter
    
    def _step1D(self, rho, ux, uy, E, Pg, dt):
        dx = self.dx 
        U = self._U(rho, ux, uy, E)
        Fx = self._flux(U, axis=-1)

        Uprev = np.roll(U, 1, axis=-1)
        Unext = np.roll(U, -1, axis=-1)
        Fprev = np.roll(Fx, 1, axis=-1)
        Fnext = np.roll(Fx, -1, axis=-1)

        FLo_L, FLo_R = self._LO_flux(Uprev, U, Unext, Fprev, Fx, Fnext,
                                     dt, axis=-1)
        
        FHi_L, FHi_R = self._HO_flux(Uprev, U, Unext, Fprev, Fx, Fnext,
                                     dt, axis=-1)
        
        phiL = self._phi(Uprev, axis=-1)
        phiR = self._phi(U, axis=-1)

        FL = FLo_L + phiL * (FHi_L - FLo_L)
        FR = FLo_R + phiR * (FHi_R - FLo_R)

        Un = U - dt/dx * (FR - FL)

        return Un 
    
    def _HO_flux2D(self, U, Uxm, Uxp, Uym, Uyp, Fx, Fxm, Fxp, Fy, Fym, Fyp, dt):
        dx, dy = self.dx, self.dy

        UxpH = 0.5 * (U + Uxp) - dt/dx * (Fxp - Fx)
        UxmH = 0.5 * (U + Uxm) - dt/dx * (Fx - Fxm)
        UypH = 0.5 * (U + Uyp) - dt/dy * (Fyp - Fy)
        UymH = 0.5 * (U + Uym) - dt/dy * (Fy - Fym)

        FxHOp = self._flux(UxpH, axis=-1)
        FxHOm = self._flux(UxmH, axis=-1)
        FyHOp = self._flux(UypH, axis=-2)
        FyHOm = self._flux(UymH, axis=-2)

        return FxHOp, FxHOm, FyHOp, FyHOm
    
    def _LO_flux2D(self, U, Uxm, Uxp, Uym, Uyp, Fx, Fxm, Fxp, Fy, Fym, Fyp, dt):
        dx, dy = self.dx, self.dy

        FxLFp = 0.5 * (Fx + Fxp) - 0.25 * dx/dt * (Uxp - U)
        FxLFm = 0.5 * (Fx + Fxm) - 0.25 * dx/dt * (U - Uxm)
        FyLFp = 0.5 * (Fy + Fyp) - 0.25 * dy/dt * (Uyp - U)
        FyLFm = 0.5 * (Fy + Fym) - 0.25 * dy/dt * (U - Uym)

        FxHOp, FxHOm, FyHOp, FyHOm = self._HO_flux2D(U, Uxm, Uxp, Uym, Uyp, Fx, Fxm, Fxp, Fy, Fym, Fyp, dt)

        FxLOp = 0.5 * (FxHOp + FxLFp)
        FxLOm = 0.5 * (FxHOm + FxLFm)
        FyLOp = 0.5 * (FyHOp + FyLFp)
        FyLOm = 0.5 * (FyHOm + FyLFm)

        return FxLOp, FxLOm, FyLOp, FyLOm

    
    def _step2D(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy

        U = self._U(rho, ux, uy, E)
        Uxm = np.roll(U, 1, axis=-1)
        Uxp = np.roll(U, -1, axis=-1)
        Uym = np.roll(U, 1, axis=-2)
        Uyp = np.roll(U, -1, axis=-2)

        Fx, Fy = self._flux(U)

        Fxm = self._flux(Uxm, axis=-1)
        Fxp = self._flux(Uxp, axis=-1)
        Fym = self._flux(Uym, axis=-2)
        Fyp = self._flux(Uyp, axis=-2)

        FxLOp, FxLOm, FyLOp, FyLOm = self._LO_flux2D(U, Uxm, Uxp, Uym, Uyp, Fx, Fxm, Fxp, Fy, Fym, Fyp, dt)
        FxHOp, FxHOm, FyHOp, FyHOm = self._HO_flux2D(U, Uxm, Uxp, Uym, Uyp, Fx, Fxm, Fxp, Fy, Fym, Fyp, dt)

        rxm = self._phi(Uxm, axis=-1)
        rxp = self._phi(U, axis=-1)
        rym = self._phi(Uym, axis=-2)
        ryp = self._phi(U, axis=-2)

        Fxm = FxLOm + rxm * (FxHOm - FxLOm)
        Fxp = FxLOp + rxp * (FxHOp - FxLOp)
        Fym = FyLOm + rym * (FyHOm - FyLOm)
        Fyp = FyLOp + ryp * (FyHOp - FyLOp)

        Unn = U - dt * ((Fxp - Fxm)/dx + (Fyp - Fym)/dy)

        return Unn
        
class MUSCL(Schemes):
    r'''
    The MUSCL scheme.
    '''
    def __init__(self, gamma, x0, xf, y0, yf, dx, dy,
                 boundary_condition='constant', gravity=False,
                 slope_limiter='minmod', **kwargs):
        
        self.method = 'MUSCL-Roe+' + slope_limiter
        self.lim = slope_limiter
        super().__init__(gamma, x0, xf, y0, yf, dx, dy,
                         boundary_condition, gravity)
    
    def _comp_flux1D(self, U):
        nx, ny = self.nx+2, self.ny
        dx = self.dx

        UL, UR = reconstruct1D(U, nx, ny, dx, self.lim)
        
        PgL = self._EOS(UL)
        PgR = self._EOS(UR)
        F = self._Roe_flux(UL, PgL, -1, UR, PgR)
        
        dF = build_flux1D(F, nx, ny, dx)

        return dF
    
    def _comp_flux2D(self, U):
        nx, ny = self.nx+2, self.ny+2
        dx, dy = self.dx, self.dy

        UL, UR, US, UN = reconstruct2D(U, nx, ny, dx, dy, self.lim)
        
        PgL = self._EOS(UL)
        PgR = self._EOS(UR)
        PgS = self._EOS(US)
        PgN = self._EOS(UN)

        Fx = self._Roe_flux(UL, PgL, -1, UR, PgR)
        Fy = self._Roe_flux(US, PgS, -2, UN, PgN)

        dFx, dFy = build_flux2D(Fx, Fy, nx, ny, dx, dy)
        dF = dFx + dFy

        return dF
    
    def _step1D(self, rho, ux, uy, E, Pg, dt):
        nx, ny = self.nx+2, self.ny

        U = np.zeros((4, ny, nx))
        U[..., 1:-1] = self._U(rho, ux, uy, E)
        U = self._set_bc(U)
        dF = self._comp_flux1D(U)

        Utmp = U - dt * dF
        Un = Utmp[..., 1:-1]

        return Un
    
    def _step2D(self, rho, ux, uy, E, Pg, dt):
        nx, ny = self.nx+2, self.ny+2

        U = np.zeros((4, ny, nx))
        U[:, 1:-1, 1:-1] = self._U(rho, ux, uy, E)
        U = self._set_bc(U)
        dF = self._comp_flux2D(U)

        Utmp = U - dt * dF
        Un = Utmp[:, 1:-1, 1:-1]

        return Un
