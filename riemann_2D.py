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

nx, ny = 400, 400
cfl = 1
gamma = 1.4
x0 = 0
xf = 1 
y0 = x0 
yf = xf
step = 30

test1 = {'Pgs': [1, 0.4, 1, 0.4],
         'rhos': [1, 0.5197, 1, 0.5197],
         'uxs': [0, -0.7259, -0.7259, 0],
         'uys': [0, 0, -0.7259, -0.7259],
         }
T1 = 0.2
txt1 = '1'
nt1 = 281
levlims1 = (35, 0, -5)

test2 = {'Pgs': [1, 0.4, 0.4, 0.4],
         'rhos': [1, 0.5197, 0.8, 0.5313],
         'uxs': [0.1, -0.6259, 0.1, 0.1],
         'uys': [-0.3, -0.3, -0.3, 0.4276],
         }
T2 = 0.2
txt2 = '2'
nt2 = 237
levlims2 = (35, 2, -3)

test3 = {'Pgs': [1.1, 0.35, 1.1, 0.35],
         'rhos': [1.1, 0.5065, 1.1, 0.5065],
         'uxs': [0, 0.8939, 0.8939, 0],
         'uys': [0, 0, 0.8939, 0.8939],
         }
T3 = 0.25
txt3 = '3'
nt3 = 388
levlims3 = (40, 10, -1)

test4 = {'Pgs': [1, 1, 1, 1],
         'rhos': [1, 2, 1, 3],
         'uxs': [0.75, 0.75, -0.75, -0.75],
         'uys': [-0.5, 0.5, 0.5, -0.5],
         }
T4 = 0.3
txt4 = '4'
nt4 = 401
levlims4 = (40, 0, -15)

test5 = {'Pgs': [1, 0.4, 0.4, 0.4],
         'rhos': [1, 0.5313, 0.8, 0.5313],
         'uxs': [0.1, 0.8276, 0.1, 0.1],
         'uys': [0, 0, 0, 0.7274],
         }
T5 = 0.3
txt5 = '5'
nt5 = 362
levlims5 = (35, 5, -5)

Ts = [T1, T2, T3, T4, T5]
tests = [test1, test2, test3, test4, test5]
txts = [txt1, txt2, txt3, txt4, txt5]
nts = [nt1, nt2, nt3, nt4, nt5]
levlims = [levlims1, levlims2, levlims3, levlims4, levlims5]

for Tstop, test, txt, nt, levlim in zip(Ts, tests, txts, nts, levlims):

    # if not txt == '2':
    #     continue

    X = np.linspace(x0, xf, nx)
    Y = np.linspace(y0, yf, ny)
    x, y = np.meshgrid(X, Y)

    ic = InitConds(nx, ny, x0, xf, y0, yf)
    ic.riemann2D(**test)
    dx, dy, rho0, ux0, uy0, E0, Pg0, T0 = ic.get_ICs()
    
    foo = '\nRunning test ' + txt
    print(foo)
    print('-' * (len(foo) - 1))
    print(f'Timesteps: {nt}')

    hd = HDSolver2D(rho0, ux0, uy0, Pg0, E0, T0, dx, dy, x0, xf, y0, yf,
                    nt=nt, bc='transmissive', cfl_cut=cfl, gamma=gamma)

    path = basepath + 'test-' + txt + '-' + hd.scheme_name + '.png'

    t, rho, ux, uy, e, Pg, T = hd.get_arrays()
    
    ux = ux[::step, ::step]
    uy = uy[::step, ::step]

    rho0 = rho[..., 0]
    Pg0 = Pg[..., 0]
    ux0 = ux[..., 0]
    uy0 = uy[..., 0]

    print(f'Wanted T={Tstop}, got T={t[-1]:.2f}')

    if t[-1] >= Tstop:
        idx = np.argwhere(t >= Tstop)[0][0]
        t = t[idx]
        rho = rho[..., idx]
        Pg = Pg[..., idx]
        ux = ux[..., idx]
        uy = uy[..., idx]

        print(f'{idx} timesteps sufficient for test {txt}')

    else:
        t = t[-1]
        rho = rho[..., -1]
        Pg = Pg[..., -1]
        ux = ux[..., -1]
        uy = uy[..., -1]

    rhos = [rho0, rho]
    Pgs = [Pg0, Pg]
    uxs = [ux0, ux]
    uys = [uy0, uy]
    ts = [0, t]

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 4.5), dpi=200)
    axes[0].set_ylabel(r'$y$')

    if txt == '4':
        color = 'red'

    else:
        color = 'white'

    axes[0].text(0.95, 0.95, '1', color=color, ha='center', va='center',
            transform=axes[0].transAxes)
    
    axes[0].text(0.45, 0.95, '2', color=color, ha='center', va='center',
            transform=axes[0].transAxes)
    
    axes[0].text(0.45, 0.45, '3', color=color, ha='center', va='center',
            transform=axes[0].transAxes)
    
    axes[0].text(0.95, 0.45, '4', color=color, ha='center', va='center',
            transform=axes[0].transAxes)

    for ax, rho, Pg, ux, uyt, t in zip(axes.flat, rhos, Pgs, uxs, uys,
                                       ts):        
        
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_xlim(x0, xf)
        ax.set_ylim(y0, yf)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        im = ax.imshow(Pg, origin='lower', cmap='rainbow',
                       extent=[x0, xf, y0, yf])

        n_levels, min_lev, max_lev = levlim
        level, drho = np.linspace(np.min(rho), np.max(rho), n_levels,
                                retstep=True)

        level = level[min_lev:max_lev]

        title1 = r'$\rho=$' + f'{test['rhos']}\n$t=0$'
        title2 = r'$\rho\in$' + f'[{level[0]:.3f}, {level[-1]:.3f}], ' \
               + f'step {drho:.3f}\n$t={t:.2f}$'
        
        if t == 0:
            title = title1
            clabel = ''

        else:
            title = title2
            clabel = 'Gas pressure'
        
        ax.set_title(title)

        print(f'rho = [{level[0]:.3f}, {level[-1]:.3f}], step = {drho:.3f}')

        ax.contour(x, y, rho, colors='black', levels=level,
                   linewidths=0.75)
        
        ax.quiver(x[::step, ::step], y[::step, ::step], ux,
                  uy, width=0.0035)        

        fig.colorbar(im, cax=cax, label=clabel)

    fig.savefig(path, bbox_inches='tight')
    plt.close()
        
# plt.show()