import numpy as np 
from hdsolve.hydrosolve import HDSolver1D, sod_analytical
from hdsolve.initial_condition import InitConds
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})

basepath = 'figures/shocktube/'

gamma = 5/3
rhoL = 0.125
rhoR = 1.0
PgL = 0.125 / gamma
PgR = 1.0 / gamma

rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL

nx = 750
nt = 500

x = np.linspace(0, 1, nx)
t_snap = 0.2

ic = InitConds(nx)
ic.shocktube(rhoL, rhoR, PgL, PgR)
dx, rho0, ux0, E0, Pg0 = ic.get_ICs()

solvers = ['lf', 'lw', 'roe', 'mc', 'flic', 'muscl']
xlabs = ['', '', r'$x$', r'$x$']
ylabs = ['Density', 'Horizontal velocity', 'Gas pressure',
         'Total energy']

ra, ua, ea, pa = sod_analytical(rhoL, rhoR, PgL, PgR, nx, t_snap)
avars = [ra, ua, pa, ea]

for solver in solvers:
    hd = HDSolver1D(rho0, ux0, Pg0, E0, dx, nt=nt, solver=solver,
                    limiter='minmod')
    
    title = hd.scheme_name + f'\n$\\Delta x={dx:.3f}$, $t={t_snap}$'
    path = basepath + hd.scheme_name + '.png'

    t, rho, ux, e, Pg = hd.get_arrays()
    idx = np.argwhere(t >= t_snap)

    r = rho[0, :, idx][0]
    u = ux[0, :, idx][0]
    e = e[0, :, idx][0]
    p = Pg[0, :, idx][0]

    vars = [r, u, p, e]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

    print('')
    print(f'Scheme: {hd.scheme_name}')
    for var, avar, ax, xlab, ylab in zip(vars, avars, axes.flat,
                                         xlabs, ylabs):
        
        ax.plot(x, avar, color='black', lw=0.75)
        ax.scatter(x, var, s=1, color='tab:blue')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax*1.075)

        mse = np.sum((var - avar)**2) / nx

        ax.text(0.05, 0.95, f'$MSE={mse:.1e}$', transform=ax.transAxes,
                va='top')
        
        print(f'Quantity: {ylab}')
        print(f'MSE: {mse:.1e}')

    fig.suptitle(title)
    fig.savefig(path, bbox_inches='tight')

plt.show()
    
'''
Scheme: Lax-Friedrich
Quantity: Density
MSE: 5.9e-04
Quantity: Horizontal velocity
MSE: 5.4e-04
Quantity: Gas pressure
MSE: 9.4e-05
Quantity: Total energy
MSE: 4.6e-03

Scheme: Lax-Wendroff
Quantity: Density
MSE: 1.3e-04
Quantity: Horizontal velocity
MSE: 9.1e-05
Quantity: Gas pressure
MSE: 7.0e-06
Quantity: Total energy
MSE: 8.6e-04

Scheme: Roe
Quantity: Density
MSE: 2.4e-04
Quantity: Horizontal velocity
MSE: 3.1e-04
Quantity: Gas pressure
MSE: 2.4e-05
Quantity: Total energy
MSE: 2.5e-03

Scheme: MacCormack
Quantity: Density
MSE: 1.2e-04
Quantity: Horizontal velocity
MSE: 1.5e-04
Quantity: Gas pressure
MSE: 6.4e-06
Quantity: Total energy
MSE: 9.1e-04

Scheme: FLIC
Quantity: Density
MSE: 1.3e-04
Quantity: Horizontal velocity
MSE: 1.9e-04
Quantity: Gas pressure
MSE: 8.3e-06
Quantity: Total energy
MSE: 1.6e-03

Scheme: MUSCL
Quantity: Density
MSE: 1.9e-04
Quantity: Horizontal velocity
MSE: 2.8e-04
Quantity: Gas pressure
MSE: 1.3e-05
Quantity: Total energy
MSE: 2.2e-03
'''