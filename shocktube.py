import numpy as np 
from hdsolve.hydrosolve import HDSolver1D, sod_analytical
from hdsolve.initial_condition import InitConds
import matplotlib.pyplot as plt 

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

nx = 500
# nx = 750
nt = 500
# nt = 500
cfl = 0.9

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
    # if not solver in ['muscl']:
    #     continue
    if solver == 'muscl':
        cfl = 0.3
        nt = 1000
    hd = HDSolver1D(rho0, ux0, Pg0, E0, dx, nt=nt, solver=solver,
                    limit_func='superbee', cfl_cut=cfl,
                    bc='transmissive')
    
    title = hd.scheme_name + f' at $t={t_snap}$' \
          + f'\n$\\Delta x={dx:.3f}$, $CFL={hd.cfl}$'
    
    path = basepath + hd.scheme_name + '.png'

    t, rho, ux, e, Pg = hd.get_arrays()
    # idx = np.argwhere(t == t[-1])
    idx = np.argwhere(t >= t_snap)

    r = rho[0, :, idx][0]
    u = ux[0, :, idx][0]
    e = e[0, :, idx][0]
    p = Pg[0, :, idx][0]

    vars = [r, u, p, e]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6), sharex=True)

    for var, avar, ax, xlab, ylab in zip(vars, avars, axes.flat,
                                         xlabs, ylabs):
        
        ax.plot(x, avar, color='black', lw=0.5)
        # ax.plot(x, var.T, color='tab:blue')
        ax.scatter(x, var, s=1, color='tab:blue')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1)

    fig.suptitle(title)
    fig.savefig(path, bbox_inches='tight')

plt.show()