# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 00:12:26 2014

@author: Max Klingsporn

    Raman Crystallinity Analysis for SiOx on AZO
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm

data = np.loadtxt("data/AMAT_AZO.txt")

test = np.linspace(0, 10, 100)
# values          a1,  a2,  a3,  a4,  a5,  a6,w1,w2,w3,w4,w5,w6,m0,a0, a7, w7
initial_guess = (0E4, 0E4, 0E4, 0E4, 1E4, 0E4,
                 77.8, 99, 50.9, 27.1, 9.5, 200,
                 15.38, 2.3E-5, 1E4, 30)

fit_range_lo = np.where(data[:, 0] >= 540)[0][0]-1
fit_range_hi = np.where(data[:, 0] >= 700)[0][0]-1

xc1 = 295
xc2 = 400
xc3 = 480
xc4 = 506
xc5 = 550
xc6 = 615
xc7 = 571


def gauss(a, w, xc, x):
    out = a / (w * np.sqrt(np.pi / 2)) * np.exp(- 2 * ((x - xc) / w)**2)
    return out


def background(m, a, x):
    out = a * x + m
    return out


def intense2min(params, x, data):
    a1 = params['a1'].value
    a2 = params['a2'].value
    a3 = params['a3'].value
    a4 = params['a4'].value
    a5 = params['a5'].value
    a6 = params['a6'].value
    w1 = params['w1'].value
    w2 = params['w2'].value
    w3 = params['w3'].value
    w4 = params['w4'].value
    w5 = params['w5'].value
    w6 = params['w6'].value
    m0 = params['m0'].value
    a0 = params['a0'].value
    a7 = params['a7'].value
    w7 = params['w7'].value

    model = \
        gauss(a1, w1, xc1, x) + gauss(a2, w2, xc2, x) + \
        gauss(a3, w3, xc3, x) + gauss(a4, w4, xc4, x) + \
        gauss(a5, w5, xc5, x) + gauss(a6, w6, xc6, x) + \
        gauss(a7, w7, xc7, x) + background(m0, a0, x)

    return model - data


params = lm.Parameters()
#         (Name ,      Value     ,  Vary, Min, Max, Expr)
params.add('a1',  initial_guess[0], False,   0, 1E5, None)
params.add('a2',  initial_guess[1], False,   0, 1E5, None)
params.add('a3',  initial_guess[2], False,   0, 1E5, None)
params.add('a4',  initial_guess[3], False,   0, 1E5, None)
params.add('a5',  initial_guess[4],  True,   0, 1E5, None)
params.add('a6',  initial_guess[5], False,   0, 1E5, None)
params.add('w1',  initial_guess[6], False,   0, 1E5, None)
params.add('w2',  initial_guess[7], False,   0, 1E5, None)
params.add('w3',  initial_guess[8], False,   0, 1E5, None)
params.add('w4',  initial_guess[9], False,   0, 1E5, None)
params.add('w5', initial_guess[10],  True,   0, 40, None)
params.add('w6', initial_guess[11], False,   0, 1E5, None)
params.add('m0', initial_guess[12], False,   0, 1E5, None)
params.add('a0', initial_guess[13], False,   0, 1E5, None)
params.add('a7', initial_guess[14],  True,   0, 1E5, None)
params.add('w7', initial_guess[15],  True,   1, 100, None)

result = lm.minimize(intense2min, params,
                     args=(data[fit_range_lo:fit_range_hi, 0],
                           data[fit_range_lo:fit_range_hi, 1]), maxfev=100000)

plt.plot(data[:, 0], data[:, 1])

plt.plot(data[:, 0], gauss(params['a1'].value,
         params['w1'].value, xc1, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))

plt.plot(data[:, 0], gauss(params['a2'].value,
         params['w2'].value, xc2, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))

plt.plot(data[:, 0], gauss(params['a3'].value,
         params['w3'].value, xc3, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))

plt.plot(data[:, 0], gauss(params['a4'].value,
         params['w4'].value, xc4, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))

plt.plot(data[:, 0], gauss(params['a5'].value,
         params['w5'].value, xc5, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))

plt.plot(data[:, 0], gauss(params['a6'].value,
         params['w6'].value, xc6, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))

plt.plot(data[:, 0], gauss(params['a7'].value,
         params['w7'].value, xc7, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))


plt.plot(data[:, 0],
         gauss(params['a1'].value, params['w1'].value, xc1, data[:, 0]) +
         gauss(params['a2'].value, params['w2'].value, xc2, data[:, 0]) +
         gauss(params['a3'].value, params['w3'].value, xc3, data[:, 0]) +
         gauss(params['a4'].value, params['w4'].value, xc4, data[:, 0]) +
         gauss(params['a5'].value, params['w5'].value, xc5, data[:, 0]) +
         gauss(params['a6'].value, params['w6'].value, xc6, data[:, 0]) +
         gauss(params['a7'].value, params['w7'].value, xc7, data[:, 0]) +
         background(params['m0'].value, params['a0'].value,
                    data[:, 0]))
