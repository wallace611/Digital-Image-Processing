from matplotlib import pyplot as plt
import numpy as np

t = np.arange(0, 3, 0.003)
x = np.sin(2*np.pi*100*t)
Y = np.fft.fft(x)

plt.plot(np.abs(Y))
plt.show()