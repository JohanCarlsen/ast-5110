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

basepath = 'figures/muscl-tests/'

gamma = 5/3
rhoL = 0.125
rhoR = 1.0
PgL = 0.125 / gamma
PgR = 1.0 / gamma

rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL

nx = 500
nt = 1000 
cfl = 0.3
t_end = 0.2

x = np.linspace(0, 1, nx)

ic = InitConds(nx)
ic.shocktube(rhoL, rhoR, PgL, PgR)
dx, rho0, u0, e0, p0 = ic.get_ICs()

ra, ua, ea, pa = sod_analytical(rhoL, rhoR, PgL, PgR, nx, t_end)
avars = [ra, ua, pa, ea]

titles = ['Density', 'Horizontal velocity', 'Gas pressure',
          'Total energy']

labels = [('', r'$y$'), ('', ''), (r'$x$', r'$y$'), (r'$x$', '')]

fig, ax = plt.subplots(figsize=(10.6, 6), dpi=200)
ax.set_title(f'MUSCL-Roe\n$\\Delta x={dx:.3f},CFL={cfl},t={t_end}$')
ax.set_xlabel(r'$x$')
ax.set_ylabel('Density')
ax.set_xlim(0, 1)

ax1 = ax.inset_axes([0.65, 0.5, 0.3, 0.4], xlim=(0.6, 0.65),
                    ylim=(0.185, 0.53), xticklabels=[], yticklabels=[])

limiters = ['minmod', 'vanleer', 'vanalbada', 'superbee']
lim_abbs = ['mm', 'vl', 'va', 'sb']

for lim, abb in zip(limiters, lim_abbs):
    hd = HDSolver1D(rho0, u0, p0, e0, dx, nt=nt, cfl_cut=cfl, solver='muscl',
                    slope_limiter=lim)

    t, r, u, e, p = hd.get_arrays()

    if t[-1] >= t_end:
        idx = np.argwhere(t >= t_end)[0][0]

    else:
        print('\nDid not reach t_end\n')
        idx = -1

    t = t[idx]
    r = r[0, :, idx]
    u = u[0, :, idx]
    e = e[0, :, idx]
    p = p[0, :, idx]

    vars = [r, u, p, e]

    ax.plot(x, r, lw=0.75, label=abb)
    ax.plot(x, ra, lw=0.75, color='black')

    ax1.plot(x, r, lw=1, marker='o', ms=2)
    ax1.plot(x, ra, lw=0.75, color='black')

    ax.indicate_inset_zoom(ax1, ec='black')

ax.legend(ncol=4, loc='upper right')
fig.savefig(basepath + 'MUSCL.png', bbox_inches='tight')

plt.show()