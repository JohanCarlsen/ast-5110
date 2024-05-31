import numpy as np 
from hdsolve.hydrosolve import HDSolver2D
from hdsolve.initial_condition import InitConds
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from conservation import calc_conserved

plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})

basepath = 'figures/rotation/'

nx = 100
ny = nx
nt = 12000
x0 = -1
xf = 1 
y0 = x0 
yf = xf

X = np.linspace(x0, xf, nx)
Y = np.linspace(y0, yf, ny)
x, y = np.meshgrid(X, Y)

ic = InitConds(nx, ny, x0, xf, y0, yf)

ic.coll_discs(rot=True, scale=0.06, gauss=True, A1=40, A2=40)

# ic.show_initial()
# exit()

dx, dy, rho0, ux0, uy0, E0, Pg0, T0 = ic.get_ICs()

# hd = HDSolver2D(rho0, ux0, uy0, Pg0, E0, T0, dx, dy, x0, xf, y0, yf,
#                 gravity=True, nt=nt, bc='periodic', M=2, cfl_cut=1)

# t, rho, ux, uy, e, Pg, T = hd.get_arrays()
# np.save('time', t)

path = basepath + 'Roe' + '.gif'

t = np.load('time.npy')
rho = np.load('rho.npy')
ux = np.load('ux.npy')
uy = np.load('uy.npy')
e = np.load('e.npy')
Pg = np.load('Pg.npy')
T = np.load('T.npy')

vars = [rho, ux, uy, e, Pg, T]
names = ['rho', 'ux', 'uy', 'e', 'Pg', 'T']

# for var, name in zip(vars, names):
#     np.save(name, var)

mass_err = np.zeros(len(t) - 2)
energy_err = np.zeros_like(mass_err)

M0 = np.sum(rho[..., 1]) * dx*dy
E0 = np.sum(rho[..., 1] * e[..., 1]) * dx*dy

for i in range(len(t) - 2):
    ri = rho[..., i+2]
    ei = e[..., i+2]

    Mi = np.sum(ri) * dx*dy 
    Ei = np.sum(ei * ri) * dx*dy

    mass_err[i] = Mi - np.sum(rho[..., i+1]) * dx*dy
    energy_err[i] = Ei - np.sum(e[..., i+1] * rho[..., i+1]) * dx*dy
    # mass_err[i] = np.abs(Mi - M0) / np.abs(M0)
    # energy_err[i] = np.abs(Ei - E0) / np.abs(E0)
    
nplot = np.arange(1, len(t) - 1)

fig, ax = plt.subplots(figsize=(10.6, 6), dpi=200)
ax.set_title('Conservation check')

ax.plot(nplot, energy_err, lw=1, label=r'$\Delta\,E$')
ax.plot(nplot, mass_err, lw=1, label=r'$\Delta\,M$')
ax.legend()
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 6), dpi=200,
#                                sharex=True, gridspec_kw={'hspace': 0})

# fig.suptitle('Conservation check')

# ax1.plot(nplot, mass_err, color='black', lw=1)
# ax1.set_xlabel('Time steps')
# ax1.set_ylabel(r'$\Delta\,M$')

# ax2.plot(nplot, energy_err, color='black', lw=1)
# ax2.set_xlabel('Time steps')
# ax2.set_ylabel(r'$\Delta\,E$')

fig.savefig(basepath + 'conservation-check.png', bbox_inches='tight')

plt.show()
plt.close()
exit()

titles = [r'Density', 'Horizontal velocity',
          'Vertical velocity', r'Total energy',
          'Gas pressure', 'Gas temperature']

labels = [('', r'$y$'), ('', ''), ('', ''),
          (r'$x$', r'$y$'), (r'$x$', ''), (r'$x$', '')]

maps = ['plasma', 'bwr', 'bwr', 'plasma', 'plasma', 'plasma']
ims = []

nframes = 100
if nt // nframes == 0:
    frames = np.arange(len(t))

else:
    frames = np.arange(0, len(t), nt//nframes)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

time = r'$t=$' + f'${t[0]:.3e}$'
base_text = 'Collision/merger with rotation and gravity'
txt = fig.suptitle(base_text + f'\n{time}')

for var, ax, title, label, map in zip(vars, axes.flat, titles, labels,
                                      maps):
    
    xlab, ylab = label
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    ax.set_xlim(x0, xf)
    ax.set_ylim(y0, yf)

    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    im = ax.imshow(var[..., 0], origin='lower', cmap=map,
                   extent=[x0, xf, y0, yf])
    
    ims.append(im)
    fig.colorbar(im, cax=cax)

def update(i):
    time = r'$t=$' + f'${t[i]:.3e}$'

    for var, im, in zip(vars, ims):
        phi = var[..., i]
        vmin = np.min(phi)
        vmax = np.max(phi)

        im.set_data(phi)
        im.set_clim(vmin, vmax)

    txt.set_text(base_text + f'\n{time}')

ani = FuncAnimation(fig, update, frames, interval=50)
ani.save(path, PillowWriter(fps=20))

