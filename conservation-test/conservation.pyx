cimport cython
import numpy as np 
cimport numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def calc_conserved(double[:, :, :] t_term, double[:, :, :] x_term,
                   double[:, :, :] y_term, double[:] t, double dx,
                   double dy, int nx, int ny, int nt):

    cdef:
        np.ndarray[dtype=DTYPE_t, ndim=1] result = np.zeros(len(t) - 1, dtype=DTYPE)
        double tn, tnn, xn, xnp, yn, ynp, dt, s, res
        int i, j, n 

    for n in range(nt - 1):
        s = 0
        dt = t[n+1] - t[n]

        # Internal points
        for i in range(ny - 1):
            for j in range(nx - 1):
                tn = t_term[i, j, n]; tnn = t_term[i, j, n+1]
                xn = x_term[i, j, n]; xnp = x_term[i, j+1, n]
                yn = y_term[i, j, n]; ynp = y_term[i+1, j, n]

                res = (tnn - tn) + dt/dx * (xnp - xn) + dt/dy * (ynp - yn)
                #res = (tnn - tn) / dt + (xnp - xn) / dx + (ynp - yn) / dy

                if i == 0 or j == 0:
                    s -= abs(res)
                
                else:
                    s += abs(res)

        # x boundary
        for i in range(ny - 1):
            tn = t_term[i, nx-1, n]; tnn = t_term[i, nx-1, n+1]
            xn = x_term[i, nx-1, n]; xnp = x_term[i, 0, n]
            yn = y_term[i, nx-1, n]; ynp = y_term[i+1, nx-1, n]

            res = (tnn - tn) + dt/dx * (xnp - xn) + dt/dy * (ynp - yn)
            #res = (tnn - tn) / dt + (xnp - xn) / dx + (ynp - yn) / dy 
            s -= abs(res)

        # y boundary 
        for j in range(nx - 1):
            tn = t_term[ny-1, j, n]; tnn = t_term[ny-1, j, n+1]
            xn = x_term[ny-1, j, n]; xnp = x_term[ny-1, j+1, n]
            yn = y_term[ny-1, j, n]; ynp = y_term[0, j, n]

            res = (tnn - tn) + dt/dx * (xnp - xn) + dt/dy * (ynp - yn)
            #res = (tnn - tn) / dt + (xnp - xn) / dx + (ynp - yn) / dy
            s -= abs(res)

        # Corner point
        tn = t_term[ny-1, nx-1, n]; tnn = t_term[ny-1, nx-1, n+1]
        xn = x_term[ny-1, nx-1, n]; xnp = x_term[ny-1, 0, n]
        yn = y_term[ny-1, nx-1, n]; ynp = y_term[0, nx-1, n]
        
        res = (tnn - tn) + dt/dx * (xnp - xn) + dt/dy * (ynp - yn)
        #res = (tnn - tn) / dt + (xnp - xn) / dx + (ynp - yn) / dy
        s -= abs(res)

        result[n] = s / (nx * ny)

    return result 


