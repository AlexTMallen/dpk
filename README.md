# dpk
Deep Probabilistic Koopman: long-term time-series forecasting under quasi-periodic uncertainty

This is an ergonomic version of
[this](https://github.com/AlexTMallen/koopman-forecasting) repo (which
contains the code to reproduce results from our
[paper](https://arxiv.org/abs/2106.06033)).

# Deep Probabilistic Koopman (DPK): Long-term time-series forecasting under periodic uncertainties
**Stable, long-term, probabilistic forecasts with calibrated uncertainty
measures.** It's fast to pick up, and fast to make forecasts since no
time-stepping is required. It is especially useful for data that exhibit
multiple frequencies of seasonality and have little long term trend.

## Installing
DPK is available through pip

`pip install dpk-forecast`

It has 3 dependencies:

- torch
- numpy
- scipy

## Training a DPK model
1. Load your time-series data into memory as a time-by-n numpy array.  
   (If the samples are not uniform over time, you will also need to
   provide a 1D array of these times).
2. Input the seasonalities your data exhibits. This is usually as easy
   as 24 hours, 1 week, and 1 year, but sometimes the frequencies must
   be found by DPK by solving a global optimization problem or via the
   FFT.
3. Choose a model object from `model_obs.py` or write your own. We
   recommend starting out with `SkewNLLwithTime`, which assumes your
   data is drawn from a time-varying skew-normal distribution at every
   point in time. "withTime" indicates that this model object allows for
   non-periodic trends.
4. Call fit!

```
x = np.sin(np.linspace(0, 1000 * np.pi, 10000)).reshape(-1, 1)  # for example
periods = [20,]  # 20 idxs is a period
num_freqs = [len(periods),] * 3  # Skew-normal distrs are parametrized by 3 numbers, all must be forecast
model_obj = model_objs.SkewNLLwithTime(x_dim=x.shape[1], num_freqs=num_freqs)

k = koopman_probabilistic.KoopmanProb(model_obj, device='cpu')
k.init_periods(periods)
k.fit(x, iterations=10, weight_decay=0, verbose=True)
```

## Forecasting
1. Call predict! This returns the time-varying parameters of the
   distribution defined by your model object. To obtain a point forecast
   with uncertainty, simply call `model_obj.mean(params)` and
   `model_obj.std(params)`. Calculating the forecast should be
   near-instantaneous since DPK does not require time-stepping for
   predictions.

```
params = k.predict(T=15000)
loc_hat, scale_hat, alpha_hat = params  # time-varying parameters of skew-normal distribution
x_hat = model_obj.mean(params)
std_hat = model_obj.std(params)  # uncertainty over time
```

A similar but slightly more sophisticated example can be found in
`example.py`.

### Missing/Non-uniform Observations
In the case of non-uniform data, `k.fit` can be called with an
additional `tt` parameter that indicates the time of each observation.
`k.predict` can also be called with an array of time values, rather than
forecasting for each integer from 0 to `T`.

