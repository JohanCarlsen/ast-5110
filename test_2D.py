import numpy as np 
from hdsolve.hydrosolve import HDSolver
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import perf_counter_ns

plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})

ux0 = 1 
uy0 = 1 
rho0 = 1
Pg0 = 1

n_regions = 3

'''
Notes:
Roe works with 150x150, nt=6000, and Pg0=0.25
LW works with 150x150, nt=4000, and Pg0=15
'''
HD = HDSolver(150, 150, nt=4000, Pg0=15, KHI=True, n_regions=n_regions)
t1 = perf_counter_ns()
HD.evolve('mc', bc='noslip', verbose=True)
t2 = perf_counter_ns()

if n_regions == 2:
    fname_ext = '-KHI-OWN.gif'

elif n_regions == 3:
    fname_ext = '-KHI.gif'

# HD = HDSolver(150, 150, rho0=rho0, nt=100)
# t1 = perf_counter_ns()
# HD.evolve('roe', bc='periodic', verbose=True)
# t2 = perf_counter_ns()
# fname_ext = '-test2D.gif'

time = (t2-t1) * 1e-9
print(f'Finished in {time:.3f} seconds.')

t, rho, ux, uy, e, Pg, T = HD.get_arrays()
vars = [rho, ux, uy, e, Pg, T]
titles = ['Density', 'Horizontal velocity', 'Vertical velocity',
          'Internal energy', 'Gas pressure', 'Gas temperature']

labels = [('', r'$y$'), ('', ''), ('', ''),
          (r'$x$', r'$y$'), (r'$x$', ''), (r'$x$', '')]
ims = []
frames = np.arange(0, len(t), HD.N//200)
if len(frames) < 2:
    frames = np.arange(len(t))

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
time = r'$t=$' + f'${t[0]:.3e}$'
txt = fig.suptitle(HD.scheme_name + f'\n{time}')

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

    txt.set_text(HD.scheme_name + f'\n{time}')

ani = FuncAnimation(fig, update, frames, interval=50)
ani.save('figures/' + HD.scheme_name + fname_ext, PillowWriter(fps=50))