import numpy as np 
from hdsolve.hydrosolve import sod_analytical
import matplotlib.pyplot as plt 

plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica',
                     'figure.dpi': 200,
                     'figure.figsize': (10.6, 6),
                     'savefig.bbox': 'tight'})

basepath = 'figures/shocktube/'

gamma = 5/3
rhoL = 0.125
rhoR = 1.0
PgL = 0.125 / gamma
PgR = 1.0 / gamma

rhoL, rhoR, PgL, PgR = rhoR, rhoL, PgR, PgL

nx = 500

x = np.linspace(0, 1, nx)
t = 0.2

r, u, e, p, ft, hd, cd, sh = sod_analytical(rhoL, rhoR, PgL, PgR, nx, t,
                                            ret_pos=True)
vars = [r, e, u, p]
labels = [r'$\rho$', r'$E$', r'$u_x$', r'$P$']
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

regions1 = [1, 2, 3, 4, 5]
regions2 = [1, 2, 3, 4]

pos1 = [hd/2, hd+(ft-hd)/2, ft+(cd-ft)/2, cd+(sh-cd)/2, sh+(1-sh)/2]
pos2 = [hd/2, hd+(ft-hd)/2, ft+(sh-ft)/2, sh+(1-sh)/2]

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0}, sharex=True)
fig.suptitle(f"Sod's shock tube problem with an exact Riemann solver\n$t={t}$")

for i in range(len(vars)):
    var = vars[i]
    lab = labels[i]
    c = colors[i]

    if i < 2:
        ax = ax1 

    else:
        ax = ax2

    ax.plot(x, var, label=lab, color=c)

for i, ax in enumerate([ax1, ax2]):
    if i == 0:
        y = 1.2

    else:
        y = 0.45
    
    for l in [hd, ft, cd, sh]:
        ax.axvline(l, color='black', ls='dashed', lw=1)

    for pos, reg in zip(pos1, regions1):
        ax.text(pos, y, reg, ha='center', va='center', fontsize=12)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(0, 1)
    ax.legend()

fig.savefig(basepath + 'analytical.png')

plt.show()