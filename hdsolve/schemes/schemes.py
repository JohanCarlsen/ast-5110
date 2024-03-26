import numpy as np 
from numba import njit
from .roe_flux import roe_flux

class Schemes:

    def __init__(self, gamma, dx, dy, boundary_condition='constant'):
        self.g = gamma 
        self.dx = dx
        self.dy = dy
        self.bc = boundary_condition
        self.two_bc = False
        self.is2D = False 
        
        if dy < 1:
            self.is2D = True

    def _get_variables(self, U):
        rho = U[0]
        ux = U[1] / rho
        uy = U[2] / rho 
        E = U[3] / rho

        return rho, ux, uy, E

    def _flux(self, U, Pg=None, axis='all'):
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

        elif self.bc == 'periodic' or self.bc == 'noslip':
            for i in range(Utmp.shape[0]):
                pad = Utmp[i, 1:-1, 1:-1]
                Utmp[i] = np.pad(pad, [1, 1], mode='wrap')

                if self.bc == 'noslip':
                    if i == 2:
                        Utmp[i, 0, :] = 0
                        Utmp[i, -1, :] = 0

                    else:    
                        Utmp[i, 0, :] = self._diff3(Utmp[i], 0)
                        Utmp[i, -1, :] = self._diff3(Utmp[i], -1)

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

    def _Roe_variables(self, UL, axis):
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

    @staticmethod
    @njit
    def __RF(nx, ny, dF, R, dR, Ux, dUx, Uy, dUy, dPg, H, C, V2, axis):
        FRoe = np.zeros_like(dF, dtype=np.float64)
        D = np.zeros((FRoe.shape[0], FRoe.shape[0]), dtype=np.float64)
        d = np.zeros(FRoe.shape[0], dtype=np.float64)
        alpha = np.zeros(FRoe.shape[0], dtype=np.float64)

        if axis == -1:
            for i in range(ny):
                for j in range(nx):
                    r, ux, uy = R[i, j], Ux[i, j], Uy[i, j]
                    dr, dux, duy = dR[i, j], dUx[i, j], dUy[i, j]
                    h, c, v2, dp = H[i, j], C[i, j], V2[i, j], dPg[i, j]

                    K = np.array([[  1.   ,  1.   , 0. ,   1.  ],
                                  [ux-c   ,  ux   , 0. ,  ux+c ],
                                  [ uy    ,  uy   , 1. ,   uy  ],
                                  [h-ux*c , v2/2. , uy , h+ux*c]],
                                  dtype=np.float64)
                    
                    a1 = (dp - r*c*dux) / (2*c**2)
                    a2 = dr - dp/c**2
                    a3 = r*duy
                    a4 = (dp + r*c*dux) / (2*c**2)

                    alpha[0] = a1
                    alpha[1] = a2
                    alpha[2] = a3
                    alpha[3] = a4

                    d[0] = abs(ux-c)
                    d[1] = abs(ux)
                    d[2] = abs(ux)
                    d[3] = abs(ux+c)
                    np.fill_diagonal(D, d)

                    FRoe[:, i, j] = dF[:, i, j] - 0.5 * (K @ D @ alpha)

        elif axis == -2:
            for i in range(ny):
                for j in range(nx):
                    r, ux, uy = R[i, j], Ux[i, j], Uy[i, j]
                    dr, dux, duy = dR[i, j], dUx[i, j], dUy[i, j]
                    h, c, v2, dp = H[i, j], C[i, j], V2[i, j], dPg[i, j]

                    K = np.array([[  1.   ,  1.   , 0. ,   1.  ],
                                  [ ux    ,  uy   , 1. ,  ux   ],
                                  [ uy-c  ,  ux   , 0. ,  uy+c ],
                                  [h-uy*c , v2/2. , ux , h+uy*c]],
                                  dtype=np.float64)
                    
                    a1 = (dp - r*c*duy) / (2*c**2)
                    a2 = dr - dp/c**2
                    a3 = r*dux
                    a4 = (dp + r*c*duy) / (2*c**2)

                    alpha[0] = a1
                    alpha[1] = a2
                    alpha[2] = a3
                    alpha[3] = a4

                    d[0] = abs(uy-c)
                    d[1] = abs(uy)
                    d[2] = abs(uy)
                    d[3] = abs(uy+c)
                    np.fill_diagonal(D, d)

                    FRoe[:, i, j] = dF[:, i, j] - 0.5 * (K @ D @ alpha)

        return FRoe
    
    def _limiter(self, U, type, epsilon, axis):
        Up1 = np.roll(U, -1, axis=axis)
        Um1 = np.roll(U, 1, axis=axis)

        num = np.where(np.abs(U - Um1) < epsilon, 0, U-Um1)
        den = np.where(np.abs(Up1 - U) < epsilon, epsilon, Up1-U)

        r = num/den
        zero = np.zeros_like(r)
        one = np.ones_like(r)

        sb = lambda r: np.maximum(zero,
                                  np.minimum(2*r, one),
                                  np.minimum(r, 2*one)
                                  )
        
        mm = lambda r: np.maximum(zero, np.minimum(one, r))

        vl = lambda r: (r + np.abs(r)) / (1 + np.abs(r))

        LIMITERS = {'superbee': sb, 'minmod': mm, 'vanleer': vl}

        phi = LIMITERS[type](r)

        return phi


    def _Roe_flux(self, UL, PgL, axis):
        nx, ny = UL.shape[2], UL.shape[1]

        UR = np.roll(UL, -1, axis=axis)
        PgR = np.roll(PgL, -1, axis=axis)

        FL = self._flux(UL, PgL, axis=axis)
        FR = self._flux(UR, PgR, axis=axis)
        dF = 0.5 * (FL + FR)

        R, dR, Ux, dUx, Uy, dUy, H, C, V2 = self._Roe_variables(UL, axis)
        dPg = PgR - PgL

        FRoe = roe_flux(nx, ny, dF, R, dR, Ux, dUx, Uy, dUy, dPg,
                        H, C, V2, axis)
        # FRoe = self.__RF(nx, ny, dF, R, dR, Ux, dUx, Uy, dUy, dPg,
        #                  H, C, V2, axis)

        return FRoe
    
class Roe(Schemes):

    def __init__(self, gamma, dx, dy, boundary_condition='constant', **kwargs):
        self.method = 'Roe'
        super().__init__(gamma, dx, dy, boundary_condition)

    def _step(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy
        U = self._U(rho, ux, uy, E)
        FRoexP = self._Roe_flux(U, Pg, -1)

        Uxm = np.roll(U, 1, axis=-1)
        Pgxm = self._EOS(Uxm)
        # from time import perf_counter_ns
        # t1 = perf_counter_ns()
        FRoexM = self._Roe_flux(Uxm, Pgxm, -1)
        # t2 = perf_counter_ns()
        # time = (t2 - t1) * 1e-6
        # print(f'Time: {time:.2f} ms')
        # exit()

        dFx = FRoexP - FRoexM
        Un = U - dt/dx * dFx 

        if not self.is2D:
            return Un 
        
        else:
            Pgn = self._EOS(Un)
            GRoeyP = self._Roe_flux(Un, Pgn, -2)

            Uym = np.roll(Un, 1, axis=-2)
            Pgym = self._EOS(Uym)

            GRoeyM = self._Roe_flux(Uym, Pgym, -2)

            dFy = GRoeyP - GRoeyM
            Unn = Un - dt/dy * dFy 

            return Unn
    
class LaxFriedrich(Schemes):

    def __init__(self, gamma, dx, dy, boundary_condition='constant', **kwargs):
        self.method = 'Lax-Friedrich'
        super().__init__(gamma, dx, dy, boundary_condition)

    def _step(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy
        U = self._U(rho, ux, uy, E)
        Fx = self._flux(U, Pg, axis=-1)

        UW = np.roll(U, 1, axis=-1)
        UE = np.roll(U, -1, axis=-1)
        FW = np.roll(Fx, 1, axis=-1)
        FE = np.roll(Fx, -1, axis=-1)

        Un = 0.5 * (UW + UE - dt/dx * (FE - FW))

        if not self.is2D:
            return Un 
        
        else:
            Pgn = self._EOS(Un)
            Fy = self._flux(Un, Pgn, axis=-2)

            US = np.roll(Un, 1, axis=-2)
            UN = np.roll(Un, -1, axis=-2)
            FS = np.roll(Fy, 1, axis=-2)
            FN = np.roll(Fy, -1, axis=-2)

            Unn = 0.5 * (US + UN - dt/dy * (FN - FS))

            return Unn
    
class LaxWendroff(Schemes):

    def __init__(self, gamma, dx, dy, boundary_condition='constant', **kwargs):
        self.method = 'Lax-Wendroff'
        super().__init__(gamma, dx, dy, boundary_condition)

    def _step(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy 

        U = self._U(rho, ux, uy, E)
        Fx = self._flux(U, Pg, axis=-1)

        UL = np.roll(U, 1, axis=-1)
        UR = np.roll(U, -1, axis=-1)
        FL = np.roll(Fx, 1, axis=-1)
        FR = np.roll(Fx, -1, axis=-1)

        UW = 0.5 * (UL + U - dt/dx * (Fx - FL))
        UE = 0.5 * (UR + U - dt/dx * (FR - Fx))

        PgW = self._EOS(UW)
        PgE = self._EOS(UE)
        FW = self._flux(UW, PgW, axis=-1)
        FE = self._flux(UE, PgE, axis=-1)

        Un = U - dt/dx * (FE - FW)

        if not self.is2D:
            return Un 
        
        else:
            Pgn = self._EOS(Un)
            Fy = self._flux(Un, Pgn, axis=-2)

            UD = np.roll(Un, 1, axis=-2)
            UU = np.roll(Un, -1, axis=-2)
            FD = np.roll(Fy, 1, axis=-2)
            FU = np.roll(Fy, -1, axis=-2)

            US = 0.5 * (UD + Un - dt/dy * (Fy - FD))
            UN = 0.5 * (UU + Un - dt/dy * (FU - Fy))

            PgS = self._EOS(US)
            PgN = self._EOS(UN)
            FS = self._flux(US, PgS, axis=-2)
            FN = self._flux(UN, PgN, axis=-2)

            Unn = Un - dt/dy * (FN - FS)

            return Unn

class MacCormack(Schemes):

    def __init__(self, gamma, dx, dy, boundary_condition='constant', **kwargs):
        self.method = 'MacCormack'
        super().__init__(gamma, dx, dy, boundary_condition)

    def _step(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy

        pos_x = np.all(ux >= 0)
        pos_y = np.all(uy >= 0)

        U = self._U(rho, ux, uy, E)
        Fx = self._flux(U, Pg, axis=-1)

        FL = np.roll(Fx, 1, axis=-1)
        FR = np.roll(Fx, -1, axis=-1)

        fwd_x = FR - Fx 
        bck_x = Fx - FL

        dFx1 = pos_x * fwd_x + (not pos_x) * bck_x
        Uxn = U - dt/dx * dFx1
        Pgxn = self._EOS(Uxn)

        Fxn = self._flux(Uxn, Pgxn, axis=-1)
        FLn = np.roll(Fxn, 1, axis=-1)
        FRn = np.roll(Fxn, -1, axis=-1)

        fwd_xn = FRn - Fxn 
        bck_xn = Fxn - FLn 

        dFx = pos_x * bck_xn + (not pos_x) * fwd_xn
        Un = 0.5 * (Uxn + U - dt/dx * dFx)

        if not self.is2D:
            return Un
        
        else:
            Pgn = self._EOS(Un)
            Fy = self._flux(Un, Pgn, axis=-2)
            
            FS = np.roll(Fy, 1, axis=-2)
            FN = np.roll(Fy, -1, axis=-2)

            fwd_y = FN - Fy 
            bck_y = Fy - FS 

            dFy1 = pos_y * fwd_y + (not pos_y) * bck_y
            Uyn = Un - dt/dy * dFy1
            Pgyn = self._EOS(Uyn)

            Fyn = self._flux(Uyn, Pgyn, axis=-2)
            FSn = np.roll(Fyn, 1, axis=-2)
            FNn = np.roll(Fyn, -1, axis=-2)

            fwd_yn = FNn - Fyn 
            bck_yn = Fyn - FSn 

            dFy = pos_y * bck_yn + (not pos_y) * fwd_yn
            Unn = 0.5 * (Un + Uyn - dt/dy * dFy)

            return Unn
        
class FLIC(Schemes):
    '''
    Flux LImiter Center scheme
    '''
    def __init__(self, gamma, dx, dy, boundary_condition='constant',
                 limit_func='minmod', epsilon=1e-8, **kwargs):
        
        self.method = 'FLIC'
        self.type = limit_func
        self.eps = epsilon
        super().__init__(gamma, dx, dy, boundary_condition)
    
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

        # Lax-Friedrich flux
        LF_L = 0.5 * (F + Fprev - dxy/dt * (U - Uprev))
        LF_R = 0.5 * (F + Fnext - dxy/dt * (Unext - U))

        # Richtmyer flux
        Ri_L, Ri_R = self._HO_flux(Uprev, U, Unext, Fprev, F, Fnext,
                                   dt, axis)
        
        # First-order central flux
        FL = 0.5 * (LF_L + Ri_L)
        FR = 0.5 * (LF_R + Ri_R)

        return FL, FR

    def _step(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy 

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
        
        phiL = self._limiter(Uprev, self.type, self.eps, axis=-1)
        phiR = self._limiter(U, self.type, self.eps, axis=-1)

        FL = FLo_L + phiL * (FHi_L - FLo_L)
        FR = FLo_R + phiR * (FHi_R - FLo_R)

        Un = U - dt/dx * (FR - FL)

        if not self.is2D:
            return Un
        
        else:
            Fy = self._flux(Un, axis=-2)

            Uprev = np.roll(Un, 1, axis=-2)
            Unext = np.roll(Un, -1, axis=-2)
            Fprev = np.roll(Fy, 1, axis=-2)
            Fnext = np.roll(Fy, -1, axis=-2)

            FLo_L, FLo_R = self._LO_flux(Uprev, Un, Unext, Fprev, Fy,
                                         Fnext, dt, axis=-2)
            
            FHi_L, FHi_R = self._HO_flux(Uprev, Un, Unext, Fprev, Fy,
                                         Fnext, dt, axis=-2)
            
            phiL = self._limiter(Uprev, self.type, self.eps, axis=-2)
            phiR = self._limiter(Un, self.type, self.eps, axis=-2)

            FL = FLo_L + phiL * (FHi_L - FLo_L)
            FR = FLo_R + phiR * (FHi_R - FLo_R)

            Unn = Un - dt/dy * (FR - FL)

            return Unn
        
class MUSCL(Schemes):
    def __init__(self, gamma, dx, dy,boundary_condition='constant',
                 beta=1/3, limit_func='minmod', epsilon=1e-8, **kwargs):
        
        self.method = 'MUSCL'
        self.type = limit_func
        self.eps = epsilon
        self.beta = beta
        super().__init__(gamma, dx, dy, boundary_condition)
        self.two_bc = True

    def _reconstruct(self, U, axis):
        k = self.beta
        eps = self.eps 
        type = self.type 
        
        Umin2 = np.roll(U, 2, axis=axis)
        Umin1 = np.roll(U, 1, axis=axis)
        Uplus1 = np.roll(U, -1, axis=axis)
        Uplus2 = np.roll(U, -2, axis=axis)

        dUplus1h = Uplus1 - U
        dUmin1h = U - Umin1
        dUplus3h = Uplus2 - Uplus1
        dUmin3h = Umin1 - Umin2

        phi_i = self._limiter(U, type, eps, axis=axis)
        phi_iplus1 = self._limiter(Uplus1, type, eps, axis=axis)
        phi_imin1 = self._limiter(Umin1, type, eps, axis=axis)

        ULplus = U + phi_i/4 * ((1-k) * dUmin1h + (1+k) * dUplus1h)
        URplus = Uplus1 - phi_iplus1/4 * ((1-k) * dUplus3h + (1+k) * dUplus1h)
        ULmin = Umin1 + phi_imin1/4 * ((1-k) * dUmin3h + (1+k) * dUmin1h)
        URmin = U - phi_i/4 * ((1-k) * dUplus1h + (1+k) * dUmin1h)

        return ULplus, URplus, ULmin, URmin
    
    def _RK_step(self, U, dxy, dt, axis):
        ULplus, URplus, ULmin, URmin = self._reconstruct(U, axis=axis)

        FLplus = self._flux(ULplus, axis=axis)
        FRplus = self._flux(URplus, axis=axis)
        FLmin = self._flux(ULmin, axis=axis)
        FRmin = self._flux(URmin, axis=axis)

        FL = 0.5 * (FLmin + FRmin - dxy/dt * (URmin - ULmin))
        FR = 0.5 * (FLplus + FRplus - dxy/dt * (URplus - ULplus))

        dF = -1/dxy * (FR - FL)

        return dF


    def _step(self, rho, ux, uy, E, Pg, dt):
        dx, dy = self.dx, self.dy

        U = self._U(rho, ux, uy, E)
        ULplus, URplus, ULmin, URmin = self._reconstruct(U, axis=-1)

        FLplus = self._flux(ULplus, axis=-1)
        FRplus = self._flux(URplus, axis=-1)
        FLmin = self._flux(ULmin, axis=-1)
        FRmin = self._flux(URmin, axis=-1)

        FL = 0.5 * (FLmin + FRmin - dx/dt * (URmin - ULmin))
        FR = 0.5 * (FLplus + FRplus - dx/dt * (URplus - ULplus))

        # Un = U - dt/dx * (FR - FL)
        k1 = self._RK_step(U, dx, dt, axis=-1)
        U1tmp = U + dt * k1/2
        U1 = self._set_bc(U1tmp)
        
        k2 = self._RK_step(U1, dx, dt, axis=-1)
        U2tmp = U + dt * k2/2
        U2 = self._set_bc(U2tmp)

        k3 = self._RK_step(U2, dx, dt, axis=-1)
        U3tmp = U + dt * k3
        U3 = self._set_bc(U3tmp)

        k4 = self._RK_step(U3, dx, dt, axis=-1)

        Un = U + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        if not self.is2D:
            return Un
        
        else:
            raise NotImplementedError('Not yet available')
