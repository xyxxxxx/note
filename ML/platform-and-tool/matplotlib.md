# tensorflow example

## gray image

```python
# draw image
plt.figure()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# draw multiple images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```



## line chart

```python
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()
```





# gallery

> 参考[matplotlib gallery](https://matplotlib.org/gallery/index.html)

![Histogram of IQ: $\mu=100$, $\sigma=15$](https://matplotlib.org/_images/sphx_glr_histogram_features_001.png)

```python
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

# example data
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(437)

num_bins = 50

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=True) 
# plot histogram
# range divided into 50 spans; normalize the area
# return bins = array of edges of bins

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))  # normal function
ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

fig.tight_layout()
plt.show()
```





![cohere](https://matplotlib.org/_images/sphx_glr_cohere_001.png)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

dt = 0.01
t = np.arange(0, 30, dt)           # 0,0.01,0.02,...,29.99
nse1 = np.random.randn(len(t))     # standard normal distribution
nse2 = np.random.randn(len(t))

# Two signals with a coherent part at 10Hz and a random part
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2

fig, axs = plt.subplots(2, 1)  # 2*1 subplots, fig controls figure, axs controls data
axs[0].plot(t, s1, t, s2)      # plot s1-t, s2-t in fig 1
axs[0].set_xlim(0, 2)          # set x-axis limit
axs[0].set_xlabel('time')      # set x-axis label
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)              # show axis

cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt) # plot coherence between s1 and s2
                                             # NFFT = 256
                                             # sampling frequency = 1./dt
axs[1].set_ylabel('coherence')

fig.tight_layout()             # adjust figure
plt.show()
```





![date](https://matplotlib.org/_images/sphx_glr_date_001.png)

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()    # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load a numpy structured array from yahoo csv data with fields date, open, max, min
# close, volume, adj_close from the mpl-data/example directory.
data = cbook.get_sample_data('goog.npz', np_load=True)['price_data']

fig, ax = plt.subplots()
ax.plot('date', 'adj_close', data=data)  # draw 'adj-close' column-'date' column in data

# format the ticks
ax.xaxis.set_major_locator(years)   # set locator
ax.xaxis.set_major_formatter(years_fmt)
ax.xaxis.set_minor_locator(months)

# round to nearest years.
datemin = np.datetime64(data['date'][0], 'Y')
datemax = np.datetime64(data['date'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)

# format the coords message box
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()

plt.show()
```

