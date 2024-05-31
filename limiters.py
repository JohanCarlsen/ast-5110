import numpy as np 
from hdsolve.schemes.muscl import phi
import matplotlib.pyplot as plt 

plt.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'figure.figsize': (8, 4.5),
                     'legend.frameon': False,
                     'animation.embed_limit': 100,
                     'font.sans-serif': 'Helvetica'})

basepath = 'figures/limiters/'

nr = 1000
r0 = -10
rf = 10

r = np.linspace(r0, rf, nr)

limiters = ['minmod', 'superbee', 'vanleer', 'vanalbada']
lim_abbv = ['mm', 'sb', 'vl', 'va']

fig, ax = plt.subplots(figsize=(10.6, 6), dpi=200)

ax.set_title('Flux limiters')
ax.set_xlim(r0, rf)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\phi(r)$')

for lim, abb in zip(limiters, lim_abbv):
    res = np.zeros(nr)

    for i in range(nr):
        res[i] = phi(r[i], lim)

    ax.plot(r, res, lw=1.25, label=abb)

ax.legend(ncol=4)
fig.savefig(basepath + 'limiters.png', bbox_inches='tight')

plt.show()