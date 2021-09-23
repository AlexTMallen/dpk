#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alex Mallen (atmallen@uw.edu)
"""

import numpy as np
import matplotlib.pyplot as plt

from dpk import koopman_probabilistic, model_objs

# generate toy data with periodic uncertainty
t = np.linspace(0, 10_000 * np.pi, 100_000)
mu_t = (1.2 + np.cos(t)) * (np.sin(2 * t) + 1) - 0.5
sigma_t = 0.1 * (2 + 0.7 * np.sin(t)) ** 2
data = np.random.normal(mu_t, sigma_t)

# normalize the data
scale = np.std(data)
loc = np.mean(data)
x = (data - loc) / scale
x = x.reshape(-1, 1)

# define a model
periods = [20, 10]  # 20 idxs is a period, 10 idxs is another driving period
model_obj = model_objs.SkewNormalNLL(x_dim=x.shape[1], num_freqs=len(periods))

# train the model
k = koopman_probabilistic.KoopmanProb(model_obj, device='cpu')
k.init_periods(periods)
k.fit(x, iterations=10, weight_decay=0, verbose=True)
params = k.predict(T=110_000)
# de-normalize to original scale and loc
params = model_obj.rescale(loc, scale, params)
loc_hat, scale_hat, alpha_hat = params
x_hat = model_obj.mean(params)
std_hat = model_obj.std(params)

# plot
plt.plot(x_hat, "tab:orange", label="$\hat x$")
plt.plot(data, "tab:blue", label="$x$")
plt.plot(x_hat + std_hat, "--k", label="$\hat x \pm \hat \sigma$")
plt.plot(x_hat - std_hat, "--k")
plt.xlim([9_900, 10_100])
plt.legend()
plt.show()

plt.plot(mu_t, label="$\mu$")
plt.plot(x_hat, ":k", label="$\hat \mu$")
plt.plot(sigma_t, label="$\sigma$")
plt.plot(std_hat, "--k", label="$\hat \sigma$")
plt.xlim([50_900, 51_100])
plt.legend()
plt.show()
