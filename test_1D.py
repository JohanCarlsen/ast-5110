import numpy as np 
from hdsolve.hydrosolve import HDSolver, sod_analytical
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

gamma = 5/3
rhoL = 0.125
rhoR = 1.0
PgL = 0.125 / gamma
PgR = 1.0 / gamma

rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL

Nx = 500
N = 250

HD = HDSolver(nx=Nx, nt=N, PgL=PgL, PgR=PgR, rhoL=rhoL, rhoR=rhoR)
HD.evolve('muscl', limiter='minmod')
fname_ext = '-sodtest.gif'

t, rho, ux, e, Pg = HD.get_arrays()
rhoa, uxa, ea, Pga = sod_analytical(rhoL, rhoR, PgL, PgR, HD.nx, 0)

frames = np.arange(0, len(t), 1)
vars = [rho, ux, Pg, e]
anavars = [rhoa, uxa, Pga, ea]

x, dx = np.linspace(0, 1, Nx, retstep=True)

rhopad = 0.05 * (np.max(rho) - np.min(rho))
rholim = [np.min(rho) - rhopad, np.max(rho) + rhopad]

upad = 0.05 * (np.max(ux) - np.min(ux))
ulim = [np.min(ux) - upad, np.max(ux) + upad]

epad = 0.05 * (np.max(e) - np.min(e))
elim = [np.min(e) - epad, np.max(e) + epad]

Pgpad = 0.05 * (np.max(Pg) - np.min(Pg))
Pglim = [np.min(Pg) - Pgpad, np.max(Pg) + Pgpad]

lims = [rholim, ulim, Pglim, elim]

ylabels = ['Density', 'Horizontal velocity', 'Gas pressure', 'Internal energy']
xlabels = ['', '', r'$x$', r'$x$']
lines = []
analines = []


fig, axes = plt.subplots(2, 2, figsize=(12,6), sharex=True)
default_txt = HD.scheme_name + r', $\Delta x=$' + f'${dx:.3f}$'
time = r'$t=$' + f'${t[0]:.3f}$'
txt = fig.suptitle(default_txt + f'\n{time}')

for var, anavar, ax, xlab, ylab, lim in zip(vars, anavars, axes.flat,
                                            xlabels, ylabels, lims):
    
    analine, = ax.plot(x, anavar, color='black', lw=0.75)
    analines.append(analine)

    line = ax.scatter(x, var[..., 0].T, s=2, color='blue', ls='dotted')
    lines.append(line)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 1)
    ax.set_ylim(lim)

def update(i):
    rhoa, uxa, ea, Pga = sod_analytical(rhoL, rhoR, PgL, PgR, Nx, t[i])
    anavars = [rhoa, uxa, Pga, ea]

    for var, anavar, line, analine in zip(vars, anavars, lines, analines):
        
        line.set_offsets(np.c_[x, var[..., i].T])
        analine.set_data(x, anavar)

    time = r'$t=$' + f'${t[i]:.3f}$'
    txt.set_text(default_txt + f'\n{time}')

ani = FuncAnimation(fig, update, frames, interval=50)
ani.save('figures/' + HD.scheme_name + fname_ext, PillowWriter(fps=10))
# plt.show()