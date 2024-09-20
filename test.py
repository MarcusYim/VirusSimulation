import random

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy.fft as fft

def top_n_indexes(lst, n):
    # Sort the list with indexes and take the top N
    return [i[0] for i in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:n]]


dta = pd.read_stata("lutkepohl2.dta")
dta.index = dta.qtr
dta.index.freq = dta.index.inferred_freq

N = 600        # Number of data points
T = 1.0 / 800  # Sample spacing
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

y_noise = y + 2.5*np.random.randn(N)

window = np.hanning(N)
y_win = y_noise * window
# Apply FFT
yf = fft.fft(y_win)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

top_yf = top_n_indexes(yf, len(yf) // 8)

yf_clean = yf.copy()
yf_clean[np.abs(yf) < np.abs(yf[top_yf[-1]])] = 0

print(yf_clean)

# Plotting the result
plt.plot(xf, 2.0/N * np.abs(yf_clean[:N//2]))
plt.grid()
plt.show()