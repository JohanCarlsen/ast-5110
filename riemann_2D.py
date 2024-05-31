import numpy as np 
from hdsolve.hydrosolve import HDSolver2D
from hdsolve.initial_condition import InitConds
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'xtick.direction': 'out',
                     'ytick.direction': 'out',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})

basepath = 'figures/riemann-2D/'

def get_params(solver):
    if solver == 'roe':
        cfl = 1
        nts = [281, 237, 388, 401, 362, 335]

    elif solver == 'muscl':
        cfl = 0.6
        nts = [440, 450, 700, 401, 638, 639]
        
    return cfl, nts    

nx, ny = 400, 400
gamma = 1.4
x0 = 0
xf = 1 
y0 = x0 
yf = xf
extent = [x0, xf, y0, yf]
step = 43
cmap = 'rainbow'

X = np.linspace(x0, xf, nx)
Y = np.linspace(y0, yf, ny)
x, y = np.meshgrid(X, Y)

test1 = {'Pgs': [1, 0.4, 1, 0.4],
         'rhos': [1, 0.5197, 1, 0.5197],
         'uxs': [0, -0.7259, -0.7259, 0],
         'uys': [0, 0, -0.7259, -0.7259],
         }
T1 = 0.2
txt1 = '1'
levlims1 = (40, 0, -20)

test2 = {'Pgs': [1, 0.4, 0.4, 0.4],
         'rhos': [1, 0.5197, 0.8, 0.5313],
         'uxs': [0.1, -0.6259, 0.1, 0.1],
         'uys': [-0.3, -0.3, -0.3, 0.4276],
         }
T2 = 0.2
txt2 = '2'
levlims2 = (35, 2, -3)

test3 = {'Pgs': [1.1, 0.35, 1.1, 0.35],
         'rhos': [1.1, 0.5065, 1.1, 0.5065],
         'uxs': [0, 0.8939, 0.8939, 0],
         'uys': [0, 0, 0.8939, 0.8939],
         }
T3 = 0.25
txt3 = '3'
levlims3 = (40, 10, -1)

test4 = {'Pgs': [1, 1, 1, 1],
         'rhos': [1, 2, 1, 3],
         'uxs': [0.75, 0.75, -0.75, -0.75],
         'uys': [-0.5, 0.5, 0.5, -0.5],
         }
T4 = 0.3
txt4 = '4'
levlims4 = (40, 0, -15)

test5 = {'Pgs': [1, 0.4, 0.4, 0.4],
         'rhos': [1, 0.5313, 0.8, 0.5313],
         'uxs': [0.1, 0.8276, 0.1, 0.1],
         'uys': [0, 0, 0, 0.7274],
         }
T5 = 0.3
txt5 = '5'
levlims5 = (35, 5, -5)

test6 = {'Pgs': [0.4, 1, 1, 1],
         'rhos': [0.5313, 1, 0.8, 1],
         'uxs': [0, 0.7276, 0, 0],
         'uys': [0, 0, 0, 0.7276],
         }
T6 = 0.25
txt6 = '6'
levlims6 = (30, 1, -1)

Ts = [T1, T2, T3, T4, T5, T6]
tests = [test1, test2, test3, test4, test5, test6]
txts = [txt1, txt2, txt3, txt4, txt5, txt6]
levlims = [levlims1, levlims2, levlims3, levlims4, levlims5, levlims6]
text_pos = [(0.75, 0.75), (0.25, 0.75), (0.25, 0.25), (0.75, 0.25)]


for Tstop, test, txt, levlim in zip(Ts, tests, txts, levlims):
    # if not txt in ['6']:
    #     continue

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 6), sharey=True, dpi=200)
    caxes = []

    for ax in axes:
        ax.set_xlabel(r'$x$')
        ax.set_xlim(x0, xf)
        ax.set_ylim(y0, yf)
        ax.set_aspect('equal')

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        caxes.append(cax)

    ax0, cax0 = axes[0], caxes[0]
    axes, caxes = axes[1:], caxes[1:]

    ax0.set_title('Initial conditions\n$t=0$')
    ax0.set_ylabel(r'$y$')
    
    info = 'RUNNING TEST ' + txt
    print('')
    print('-' * (len(info)))
    print(info)
    print('-' * (len(info)))

    path = basepath + 'test-' + txt + '.png'
    n_levs, min_lev, max_lev = levlim

    if txt == '4':
        color = 'red'

    else:
        color = 'white'

    ic = InitConds(nx, ny, x0, xf, y0, yf)
    ic.riemann2D(**test)
    dx, dy, rho0, ux0, uy0, E0, Pg0, T0 = ic.get_ICs()

    im0 = ax0.imshow(Pg0, origin='lower', cmap=cmap, extent=extent)

    ax0.quiver(x[::step, ::step], y[::step, ::step],
               ux0[::step, ::step], uy0[::step, ::step], width=0.0035)
    
    fig.colorbar(im0, cax0)

    for i, pos in enumerate(text_pos):
        text = f"$\\rho={test['rhos'][i]}$"

        ax0.text(pos[0], pos[1], text, color=color, ha='center', va='center',
                 transform=ax0.transAxes)

    for i, solver in enumerate(['roe', 'muscl']):
        ax, cax = axes[i], caxes[i]
        cfl, nts = get_params(solver)
        nt = nts[int(txt) - 1]
        
        print(f'Solver: {solver}')
        print(f'CFL: {cfl}')
        print(f'Timesteps: {nt}')

        hd = HDSolver2D(rho0, ux0, uy0, Pg0, E0, T0, dx, dy, x0, xf, y0,
                        yf, nt=nt, bc='transmissive', cfl_cut=cfl,
                        gamma=gamma, solver=solver)
        
        t, rho, ux, uy, e, Pg, T = hd.get_arrays()

        if t[-1] >= Tstop:
            idx = np.argwhere(t >= Tstop)[0][0]
            print(f'Reached T={Tstop} after {idx} timesteps\n')

        else:
            idx = -1
            print(f'Did not reach T={Tstop}\n')

        t = t[idx]
        rho = rho[..., idx]
        Pg = Pg[..., idx]
        ux = ux[::step, ::step, idx]
        uy = uy[::step, ::step, idx]

        title = hd.scheme_name + f' $CFL={cfl}$\n$t={t:.2f}$'
        ax.set_title(title)

        min_rho = np.min(np.array([rho0, rho]))
        max_rho = np.max(np.array([rho0, rho]))
        levs, drho = np.linspace(min_rho, max_rho, n_levs, retstep=True)

        im = ax.imshow(Pg, origin='lower', cmap=cmap, extent=extent)

        ax.contour(x, y, rho, colors='black', levels=levs,
                   linewidths=0.5)
        
        ax.quiver(x[::step, ::step], y[::step, ::step], ux, uy,
                  width=0.0035)
        
        cbar = fig.colorbar(im, cax)

    cbar.set_label('Gas pressure')

    fig.savefig(path, bbox_inches='tight')
    plt.close()

# fig.tight_layout()
# plt.show()