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

basepath = 'figures/gravity-test/'

nx = 150
ny = nx
nt = 2000
x0 = -1
xf = 1 
y0 = x0 
yf = xf

X = np.linspace(x0, xf, nx)
Y = np.linspace(y0, yf, ny)
x, y = np.meshgrid(X, Y)

ic = InitConds(nx, ny, x0, xf, y0, yf)
ic.coll_discs(0, 0)
dx, dy, rho0, ux0, uy0, E0, Pg0, T0 = ic.get_ICs()

hd = HDSolver2D(rho0, ux0, uy0, Pg0, E0, T0, dx, dy, x0, xf, y0, yf,
                gravity=True, nt=nt, bc='periodic', M=1e10)

path = basepath + hd.scheme_name + '.gif'

t, rho, ux, uy, e, Pg, T = hd.get_arrays()

step = 9

x = x[::step, ::step]
y = y[::step, ::step]
ux = ux[::step, ::step, :]
uy = uy[::step, ::step, :]

if nt // 100 == 0:
    frames = np.arange(len(t))

else:
    frames = np.arange(0, len(t), nt//100)

fig, ax = plt.subplots()

time = r'$t=$' + f'${t[0]:.3e}$'
txt = fig.suptitle(hd.scheme_name + f'\n{time}')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_aspect('equal')
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

im = ax.imshow(rho[..., 0], origin='lower', cmap='plasma',
               extent=[x0, xf, y0, yf])

quiv = ax.quiver(x, y, ux[..., 0], uy[..., 0], angles='xy',
                 scale_units='xy', scale=1e-7, color='white',
                 alpha=0.6)

fig.colorbar(im, cax=cax, label='Density')
# plt.show()
# exit()
def update(i):
    time = r'$t=$' + f'${t[i]:.3e}$'

    r = rho[..., i]
    vmin = np.min(r)
    vmax = np.max(r)

    im.set_data(r)
    im.set_clim(vmin, vmax)

    quiv.set_UVC(ux[..., i], uy[..., i])

    txt.set_text(hd.scheme_name + f'\n{time}')

ani = FuncAnimation(fig, update, frames, interval=50)
ani.save('test.gif', PillowWriter(fps=30))