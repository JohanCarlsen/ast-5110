import numpy as np 
from hdsolve.hydrosolve import HDSolver, sod_analytical
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
import matplotlib as mpl 


colors = reversed(['#eecc66', '#ee99aa', '#6699cc', '#997700',
                   '#994455', '#004488'])

mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'axes.facecolor': '0.925',
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})


gamma = 5/3
rhoL = 0.125
rhoR = 1.0
PgL = 0.125 / gamma
PgR = 1.0 / gamma

Nx = 500
N = 250
t_snap = 0.23
x, dx = np.linspace(0, 1, Nx, retstep=True)

rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL

ra, ua, ea, pa = sod_analytical(rhoL, rhoR, PgL, PgR, Nx, t_snap)
avars = [ra, ua, pa, ea]

schemes = ['lf', 'lw', 'mc', 'muscl', 'flic', 'roe']
abbrevs = ['LF', 'LW', 'MC', 'ML', 'FC', 'R']

xlabels = ['', '', r'$x$', r'$x$']
ylabels = ['Density', 'Horizontal velocity', 'Gas pressure',
           'Internal energy']

fname_ext = '-all.png'

fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)

for scheme, abbr in zip(schemes, abbrevs):
    # if not abbr in ['FC', 'R']:
    # if not abbr in ['LW', 'MC']:
        # continue

    HD = HDSolver(Nx, nt=N, PgL=PgL, PgR=PgR, rhoL=rhoL, rhoR=rhoR)
    HD.evolve(scheme)

    t, rho, ux, e, Pg = HD.get_arrays()
    idx = np.argwhere(t >= t_snap)[0]
    
    r = rho[0, :, idx][0]
    u = ux[0, :, idx][0]
    e = e[0, :, idx][0]
    p = Pg[0, :, idx][0]
    
    vars = [r, u, p, e]

    for var, xlab, ylab, ax in zip(vars, xlabels, ylabels, axes.flat):
        
        ax.scatter(x, var, s=0.75, label=abbr)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1)

for avar, ax in zip(avars, axes.flat):
    ax.plot(x, avar, ls='dashed', lw=1, color='black', label='Exact',
            alpha=0.5)
    
    ax.legend(ncol=7, loc='upper center', bbox_to_anchor=[0.5, 1.1],
                  markerscale=3)

fig.suptitle(r'Comparison at $t=$' + f'${t_snap:.2f}$\nwith ' + r'$\Delta x=$' + f'${dx:.3f}$')
fig.savefig('figures/comparison' + fname_ext, bbox_inches='tight')
# plt.show()
