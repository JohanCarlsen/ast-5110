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

nx = 150 
ny = nx 

muscl_pressure = 0.25
muscl_params = {'nt': 500, 'HW': 1, 'solver': 'muscl', 'cfl_cut': 0.3}

roe_pressure = 0.25
roe_params = {'nt': 7000, 'HW': 10}

lw_pressure = 200
lw_params = {'nt': 7000, 'solver': 'lw', 'HW': 10}

flic_pressure = 1000
flic_params = {'nt': 1000, 'solver': 'flic'}

mc_pressure = 5
mc_params = {'nt': 5000, 'solver': 'mc', 'HW': 25}

pg0, params = muscl_pressure, muscl_params
nt = params['nt']

ic = InitConds(nx, ny)
ic.solar(pg0, **params)
# ic.show_initial()
# exit()

dx, dy, rho0, ux0, uy0, E0, Pg0, T0 = ic.get_ICs()
hd = HDSolver2D(rho0, ux0, uy0, Pg0, E0, T0, dx, dy, **params)

t, rho, ux, uy, e, Pg, T = hd.get_arrays()
vars = [rho, ux, uy, e, Pg, T]
titles = ['Density', 'Horizontal velocity', 'Vertical velocity',
          'Total energy', 'Gas pressure', 'Gas temperature']

labels = [('', r'$y$'), ('', ''), ('', ''),
          (r'$x$', r'$y$'), (r'$x$', ''), (r'$x$', '')]
ims = []

if nt // 200 == 0:
    frames = np.arange(len(t))

else:
    frames = np.arange(0, len(t), nt//200)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
time = r'$t=$' + f'${t[0]:.3e}$'
txt = fig.suptitle(hd.scheme_name + f'\n{time}')

for var, ax, title, label in zip(vars, axes.flat, titles, labels):
    xlab, ylab = label
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    im = ax.imshow(var[..., 0], origin='lower', cmap='plasma')
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
# ani.save(hd.scheme_name + '.gif', PillowWriter(fps=50))
ani.save('figures/KHI/' + hd.scheme_name + '.gif', PillowWriter(fps=50))