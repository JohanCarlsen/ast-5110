import numpy as np 
from hdsolve.hydrosolve import HDSolver2D
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

basepath = 'figures/rotation/'

nx = 150
ny = nx
nt = 4000
x0 = -1
xf = 1 
y0 = x0 
yf = xf

X = np.linspace(x0, xf, nx)
Y = np.linspace(y0, yf, ny)
x, y = np.meshgrid(X, Y)

ic = InitConds(nx, ny, x0, xf, y0, yf)
ic.coll_discs(rot=True, scale=1e-2, gauss=True, A1=2, A2=2)
# ic.show_initial()
# exit()

dx, dy, rho0, ux0, uy0, E0, Pg0, T0 = ic.get_ICs()

hd = HDSolver2D(rho0, ux0, uy0, Pg0, E0, T0, dx, dy, x0, xf, y0, yf,
                gravity=True, nt=nt, bc='transmissive', M=1)

path = basepath + hd.scheme_name + '.gif'

t, rho, ux, uy, e, Pg, T = hd.get_arrays()
vars = [np.log10(rho), ux, uy, np.log10(e), Pg, T]
titles = [r'Density ($\log{10}$)', 'Horizontal velocity', 'Vertical velocity',
          r'Total energy ($\log_{10}$)', 'Gas pressure', 'Gas temperature']

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
txt = fig.suptitle(hd.scheme_name + f'\n{time}')

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

    txt.set_text(hd.scheme_name + f'\n{time}')

ani = FuncAnimation(fig, update, frames, interval=50)
ani.save(path, PillowWriter(fps=20))

