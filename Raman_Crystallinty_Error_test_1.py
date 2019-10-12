# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 00:12:26 2014

@author: Max Klingsporn

    Raman Crystallinity Analysis for SiOx on AZO
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit as lm
import os as os

fname = r'2014_11_06_12-32-18-4_50x_488_0001'

data = np.loadtxt("data/" + fname + ".txt")

#report = []

xc1 = 310
xc2 = 410
xc3 = 475
xc4 = 507
xc5 = 517
xc6 = 630
xc7 = 550
xc8 = 571
xc9 = 790

# values : a1,a2,a3,a4,a5,a6,w1,w2,w3,w4,w5,w6,m0,a0,a7,w7,a8,w8
initial_guess = (1E4, 1E4, 1E4, 1E4, 1E4, 1E4,
                 50, 89.27, 69.21, 26.53, 12, 64.5,
                 15.38, 2.3E-5, 1E4, 40, 1E4, 36,
                 xc1, xc2, xc3, xc4, xc5, xc6, xc9, 3E3, 34)

fit_range_lo = np.where(data[:, 0] >= 250)[0][0]-1
fit_range_hi = np.where(data[:, 0] >= 880)[0][0]-1


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
    a8 = params['a8'].value
    w8 = params['w8'].value
    xc1 = params['xc1'].value
    xc2 = params['xc2'].value
    xc3 = params['xc3'].value
    xc4 = params['xc4'].value
    xc5 = params['xc5'].value
    xc6 = params['xc6'].value
    xc9 = params['xc9'].value
    a9 = params['a9'].value
    w9 = params['w9'].value

    model = \
        gauss(a1, w1, xc1, x) + gauss(a2, w2, xc2, x) + \
        gauss(a3, w3, xc3, x) + gauss(a4, w4, xc4, x) + \
        gauss(a5, w5, xc5, x) + gauss(a6, w6, xc6, x) + \
        gauss(a7, w7, xc7, x) + gauss(a8, w8, xc8, x) + \
        gauss(a9, w9, xc9, x) + background(m0, a0, x)

    return model - data


params = lm.Parameters()
#         (Name ,      Value     ,  Vary, Min, Max, Expr)
params.add('a1',  initial_guess[0],  True,   0, 1E6, None)
params.add('a2',  initial_guess[1],  True,   0, 1E6, None)
params.add('a3',  initial_guess[2],  True,   0, 1E6, None)
params.add('a4',  initial_guess[3],  True,   0, 1E6, None)
params.add('a5',  initial_guess[4],  True,   0, 1E6, None)
params.add('a6',  initial_guess[5],  True,   0, 1E6, None)
params.add('w1',  initial_guess[6], False,   0, 100, None)
params.add('w2',  initial_guess[7], False,   0, 100, None)
params.add('w3',  initial_guess[8], False,   0, 1E5, None)
params.add('w4',  initial_guess[9], False,   0, 40, None)
params.add('w5', initial_guess[10], False,   12, 15, None)
params.add('w6', initial_guess[11], False,   0, 1E5, None)
params.add('m0', initial_guess[12],  True,   0, 1E5, None)
params.add('a0', initial_guess[13],  True,   0, 1E5, None)
params.add('a7', initial_guess[14],  True,   0, 1E5, None)
params.add('w7', initial_guess[15], False,   1, 100, None)
params.add('a8', initial_guess[16],  True,   0, 1E5, expr='1.477 * a7')
params.add('w8', initial_guess[17], False,   1, 100, None)
params.add('xc1', initial_guess[18],  True, xc1 * 0.99, xc1 * 1.01, None)
params.add('xc2', initial_guess[19],  True, xc2 * 0.99, xc2 * 1.01, None)
params.add('xc3', initial_guess[20],  True, xc3 * 0.99, xc3 * 1.01, None)
params.add('xc4', initial_guess[21],  True, xc4 * 0.99, xc4 * 1.01, None)
params.add('xc5', initial_guess[22],  True, xc5 * 0.99, xc5 * 1.01, None)
params.add('xc6', initial_guess[23],  True, xc6 * 0.99, xc6 * 1.01, None)
params.add('xc9', initial_guess[24], False, xc9 * 0.99, xc9 * 1.01, None)
params.add('a9',  initial_guess[25],  True,   0, 1E5, None)
params.add('w9',  initial_guess[26], False,   0, 100, None)

result = lm.minimize(intense2min, params,
                     args=(data[fit_range_lo:fit_range_hi, 0],
                           data[fit_range_lo:fit_range_hi, 1]), maxfev=100000)

report.append(lm.fit_report(params))



gauss_1 = gauss(params['a1'].value, params['w1'].value,
                params['xc1'].value, data[:, 0])
gauss_2 = gauss(params['a2'].value, params['w2'].value,
                params['xc2'].value, data[:, 0])
gauss_3 = gauss(params['a3'].value, params['w3'].value,
                params['xc3'].value, data[:, 0])
gauss_4 = gauss(params['a4'].value, params['w4'].value,
                params['xc4'].value, data[:, 0])
gauss_5 = gauss(params['a5'].value, params['w5'].value,
                params['xc5'].value, data[:, 0])
gauss_6 = gauss(params['a6'].value, params['w6'].value,
                params['xc6'].value, data[:, 0])
gauss_7 = gauss(params['a7'].value, params['w7'].value, xc7, data[:, 0])
gauss_8 = gauss(params['a8'].value, params['w8'].value, xc8, data[:, 0])
gauss_9 = gauss(params['a9'].value, params['w9'].value,
                params['xc9'].value, data[:, 0])
background_1 = background(params['m0'].value, params['a0'].value,
                          data[:, 0])

print "Crystallinity:", ((params['a4'].value + params['a5'].value) /
            (params['a3'].value + params['a4'].value + params['a5'].value))

plt.plot(data[:, 0], data[:, 1])
plt.plot(data[:, 0], gauss_1 + background_1)
plt.plot(data[:, 0], gauss_2 + background_1)
plt.plot(data[:, 0], gauss_3 + background_1)
plt.plot(data[:, 0], gauss_4 + background_1)
plt.plot(data[:, 0], gauss_5 + background_1)
plt.plot(data[:, 0], gauss_6 + background_1)
plt.plot(data[:, 0], gauss_7 + background_1)
plt.plot(data[:, 0], gauss_8 + background_1)
plt.plot(data[:, 0], gauss_9 + background_1)
plt.plot(data[:, 0], gauss_1 + gauss_2 + gauss_3 + gauss_4 +
         gauss_5 + gauss_6 + gauss_7 + gauss_8 + background_1)
"""
if (not os.path.exists('output/'+fname)):
    os.mkdir('output/'+fname)

np.savetxt("output/" + fname + "/gauss_1.txt", gauss_1)
np.savetxt("output/" + fname + "/gauss_2.txt", gauss_2)
np.savetxt("output/" + fname + "/gauss_3.txt", gauss_3)
np.savetxt("output/" + fname + "/gauss_4.txt", gauss_4)
np.savetxt("output/" + fname + "/gauss_5.txt", gauss_5)
np.savetxt("output/" + fname + "/gauss_6.txt", gauss_6)
np.savetxt("output/" + fname + "/gauss_7.txt", gauss_7)
np.savetxt("output/" + fname + "/gauss_8.txt", gauss_8)
np.savetxt("output/" + fname + "/gauss_9.txt", gauss_9)
np.savetxt("output/" + fname + "/background.txt", background_1)
np.savetxt("output/" + fname + "/fit_report.txt", np.array(report), fmt="%s")
"""