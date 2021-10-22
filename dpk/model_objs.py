#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Alex Mallen (atmallen@uw.edu)
Built on code from Henning Lange (helange@uw.edu)
"""

import torch
from torch import nn
import numpy as np
from scipy.special import factorial


class ModelObject(nn.Module):

    def __init__(self, num_freqs, num_covariates):
        super(ModelObject, self).__init__()
        self.num_freqs = num_freqs
        self.num_covariates = num_covariates
        self.total_freqs = sum(num_freqs)

        self.param_idxs = []
        cumul = 0
        for num_freq in self.num_freqs:
            idxs = np.concatenate([cumul + np.arange(num_freq), self.total_freqs + cumul + np.arange(num_freq)])
            self.param_idxs.append(idxs)
            cumul += num_freq

    def forward(self, w, data, training_mask=None):
        """
        Forward computes the error.
        Input:
            y: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
            x: data set
                type: torch.tensor
                dimensions: [T, ...]
        """

        raise NotImplementedError()

    def decode(self, w):
        """
        Evaluates f at temporal snapshots w
        Input:
            w: temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
        Return torch tensor of predictions
        """
        raise NotImplementedError()

    def output(self, w):
        """
        Evaluates f at temporal snapshots w
        :param w:
        temporal snapshots of the linear system
                type: torch.tensor
                dimensions: [T, (batch,) num_frequencies ]
        :return: numpy array of predictions
        """
        return tuple(param.cpu().detach().numpy() for param in self.decode(w))

    @staticmethod
    def mean(params):
        """returns the mean of a distribution with the given params"""
        return params[0]

    @staticmethod
    def std(params):
        """returns the standard deviation of a distribution with the given params"""
        return np.ones(params[0].shape)


class SkewNormalNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n=256, n2=64, num_covariates=0):
        """
        neural network that takes a vector of sines and cosines and produces a skew-normal distribution with parameters
        mu, sigma, and alpha (the outputs of the NN). trains using NLL.
        :param x_dim: number of dimensions spanned by the probability distr
        :param num_freqs: int or list. number of frequencies used for each of the 3 parameters: [num_mu, num_sig, num_alpha]
        :param n: size of 1st hidden layer of NN
        :param n2: size of 2nd hidden layer of NN
        :param num_covariates: number of covariates that will be given as inputs to the NN
        """
        if type(num_freqs) is int:
            num_freqs = [num_freqs] * 3
        super(SkewNormalNLL, self).__init__(num_freqs, num_covariates)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0] + num_covariates, n)
        self.l2_mu = nn.Linear(n, n2)
        self.l3_mu = nn.Linear(n2, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1] + num_covariates, n)
        self.l2_sig = nn.Linear(n, n2)
        self.l3_sig = nn.Linear(n2, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[2] + num_covariates, n)
        self.l2_a = nn.Linear(n, n2)
        self.l3_a = nn.Linear(n2, x_dim)

        self.norm = torch.distributions.normal.Normal(0, 1)

    def decode(self, w):
        w_mu = w[..., (*self.param_idxs[0], *np.arange(-self.num_covariates, 0))]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., (*self.param_idxs[1], *np.arange(-self.num_covariates, 0))]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 10 * nn.Softplus()(self.l3_sig(z2))  # start with large uncertainty to avoid small probabilities

        w_a = w[..., (*self.param_idxs[2], *np.arange(-self.num_covariates, 0))]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(a1))
        a = self.l3_a(a2)

        return y, z, a

    def forward(self, w, data, training_mask=None):
        data = data.type(w.type())
        mu, sig, alpha = self.decode(w)
        if training_mask is None:
            y = mu
            z = sig
            a = alpha
        else:
            y = training_mask * mu + (1 - training_mask) * mu.detach()
            # z = (1 - training_mask) * sig + training_mask * sig.detach()
            # a = (1 - training_mask) * alpha + training_mask * alpha.detach()
            z = sig
            a = alpha

        losses = (data - y)**2 / (2 * z**2) + z.log() - self._norm_logcdf(a * (data - y) / z)
        avg = torch.mean(losses, dim=-1)
        return avg

    def _norm_logcdf(self, z):

        if (z < -7).any():  # these result in NaNs otherwise
            # print("THIS BATCH USING LOG CDF APPROXIMATION (large z-score can otherwise cause numerical instability)")
            # https://stats.stackexchange.com/questions/106003/approximation-of-logarithm-of-standard-normal-cdf-for-x0/107548#107548?newreg=5e5f6365aa7046aba1c447e8ae263fec
            # I found this approx to be good: less than 0.04 error for all -20 < x < -5
            # approx = lambda x: -0.5 * x ** 2 - 4.8 + 2509 * (x - 13) / ((x - 40) ** 2 * (x - 5))
            ans = torch.where(z < -0.1, -0.5 * z ** 2 - 4.8 + 2509 * (z - 13) / ((z - 40) ** 2 * (z - 5)),
                                        -torch.exp(-z * 2) / 2 - torch.exp(-(z - 0.2) ** 2) * 0.2)
        else:
            ans = self.norm.cdf(z).log()

        return ans

    @staticmethod
    def mean(params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return mu + sigma * delta * (2 / np.pi) ** 0.5

    @staticmethod
    def std(params):
        mu, sigma, alpha = params
        delta = alpha / (1 + alpha ** 2) ** 0.5
        return sigma * (1 - 2 * delta ** 2 / np.pi) ** 0.5

    @staticmethod
    def rescale(loc, scale, params):
        """rescales a skew-normal distribution with the given parameters so that its scale is
        multiplied by scale and its center is shifted by loc"""
        mu_hat, sigma_hat, a_hat = params
        sigh, ah = sigma_hat, a_hat
        delta = ah / np.sqrt(1 + ah ** 2)
        muh = mu_hat * scale + (scale - 1) * delta * sigh * np.sqrt(2 / np.pi)
        muh = muh + loc - (scale - 1) * delta * sigh * np.sqrt(2 / np.pi)
        sigh = sigh * scale
        return muh, sigh, ah


class NormalNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n=128, n2=64, num_covariates=0):
        """
        Negative Log Likelihood neural network assuming Gaussian distribution of x at every point in time.
        Trains using NLL and trains mu and sigma separately to prevent
        overfitting
        :param x_dim: dimension of what will be modeled
        :param num_freqs: int or list of the number of frequencies used to model each parameter: [num_mu, num_sigma]
        :param n: size of 1st hidden layer of NN
        :param n2: size of 2nd hidden layer of NN
        :param num_covariates: number of covariates that will be given as inputs to the NN
        """
        if type(num_freqs) is int:
            num_freqs = [num_freqs] * 2
        super(NormalNLL, self).__init__(num_freqs, num_covariates)

        self.l1_mu = nn.Linear(2 * self.num_freqs[0] + num_covariates, n)
        self.l2_mu = nn.Linear(n, n2)
        self.l3_mu = nn.Linear(n2, x_dim)

        self.l1_sig = nn.Linear(2 * self.num_freqs[1] + num_covariates, n)
        self.l2_sig = nn.Linear(n, n2)
        self.l3_sig = nn.Linear(n2, x_dim)

    def decode(self, w):
        w_mu = w[..., (*self.param_idxs[0], *np.arange(-self.num_covariates, 0))]
        y1 = nn.Tanh()(self.l1_mu(w_mu))
        y2 = nn.Tanh()(self.l2_mu(y1))
        y = self.l3_mu(y2)

        w_sigma = w[..., (*self.param_idxs[1], *np.arange(-self.num_covariates, 0))]
        z1 = nn.Tanh()(self.l1_sig(w_sigma))
        z2 = nn.Tanh()(self.l2_sig(z1))
        z = 10 * nn.Softplus()(self.l3_sig(z2))  # start big to avoid infinite gradients

        return y, z

    def forward(self, w, data, training_mask=None):
        mu, sig = self.decode(w)
        if training_mask is None:
            y = mu
            z = sig
        else:
            y = training_mask * mu + (1 - training_mask) * mu.detach()
            # z = (1 - training_mask) * sig + training_mask * sig.detach()
            z = sig

        losses = (data.type(w.type()) - y) ** 2 / (2 * z ** 2) + torch.log(z)
        avg = torch.mean(losses, dim=-1)
        return avg

    @staticmethod
    def rescale(loc, scale, params):
        """rescales a normal distribution with the given parameters so that its scale is
                multiplied by scale and its center is shifted by loc"""
        return params[0] * scale + loc, params[1] * scale

    @staticmethod
    def mean(params):
        return params[0]

    @staticmethod
    def std(params):
        return params[1]


class GammaNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n=256, n2=64, num_covariates=0):
        """
        Negative Log Likelihood neural network assuming Gamma distribution of x at every point in time.
        Trains using NLL
        :param x_dim: dimension of what will be modeled
        :param num_freqs: int or list. list of the number of frequencies used to model each parameter: [num_rate, num_a]
        :param n: size of 1st hidden layer of NN
        :param n2: size of 2nd hidden layer of NN
        :param num_covariates: number of covariates that will be given as inputs to the NN
        """
        if type(num_freqs) is int:
            num_freqs = [num_freqs] * 2
        super(GammaNLL, self).__init__(num_freqs, num_covariates)

        self.l1_rate = nn.Linear(2 * self.num_freqs[0] + num_covariates, n)
        self.l2_rate = nn.Linear(n, n2)
        self.l3_rate = nn.Linear(n2, x_dim)

        self.l1_a = nn.Linear(2 * self.num_freqs[1] + num_covariates, n)
        self.l2_a = nn.Linear(n, n2)
        self.l3_a = nn.Linear(n2, x_dim)

    def decode(self, w):
        w_rate = w[..., (*self.param_idxs[0], *np.arange(-self.num_covariates, 0))]
        rate1 = nn.Tanh()(self.l1_rate(w_rate))
        rate2 = nn.Tanh()(self.l2_rate(rate1))
        rate = 1 / nn.Softplus()(self.l3_rate(rate2))  # helps convergence

        w_a = w[..., (*self.param_idxs[1], *np.arange(-self.num_covariates, 0))]
        a1 = nn.Tanh()(self.l1_a(w_a))
        a2 = nn.Tanh()(self.l2_a(a1))
        a = nn.Softplus()(self.l3_a(a2))

        return rate, a

    def forward(self, w, data, training_mask=None):
        rate, a = self.decode(w)
        if training_mask is not None:
            mean = a / rate
            var = a / rate ** 2
            # don't change prediction of mean based on 0-training_mask indices
            mean = training_mask * mean + (1 - training_mask) * mean.detach()
            rate = mean / var
            a = mean / var ** 2

        losses = -torch.distributions.gamma.Gamma(a, rate).log_prob(data.type(w.type()))
        avg = torch.mean(losses, dim=-1)
        return avg

    @staticmethod
    def mean(params):
        return params[1] / params[0]

    @staticmethod
    def std(params):
        return np.sqrt(params[1] / params[0] ** 2)

    @staticmethod
    def rescale(scale, params):
        return params[0] / scale, params[1]


class PoissonNLL(ModelObject):

    def __init__(self, x_dim, num_freqs, n=128, n2=64, num_covariates=0):
        """
        Negative Log Likelihood neural network assuming Poisson distribution of x at every point in time.
        Trains using NLL
        :param x_dim: dimension of what will be modeled
        :param num_freqs: int or list. list of the number of frequencies used to model each parameter: [num_rate,]
        :param n: size of 1st hidden layer of NN
        :param n2: size of 2nd hidden layer of NN
        :param num_covariates: number of covariates that will be given as inputs to the NN
        """
        if type(num_freqs) is int:
            num_freqs = [num_freqs] * 1
        super(PoissonNLL, self).__init__(num_freqs, num_covariates)

        self.l1_rate = nn.Linear(2 * self.num_freqs[0] + num_covariates, n)
        self.l2_rate = nn.Linear(n, n2)
        self.l3_rate = nn.Linear(n2, x_dim)

    def decode(self, w):
        w_rate = w[..., (*self.param_idxs[0], *np.arange(-self.num_covariates, 0))]
        rate1 = nn.Tanh()(self.l1_rate(w_rate))
        rate2 = nn.Tanh()(self.l2_rate(rate1))
        rate = 1 / nn.Softplus()(self.l3_rate(rate2))  # helps convergence

        return rate,

    def forward(self, w, data, training_mask=None):
        assert (training_mask is None), "Poisson distributions don't support training masks"
        rate, = self.decode(w)

        losses = -torch.distributions.poisson.Poisson(rate).log_prob(data.type(w.type()))
        avg = torch.mean(losses, dim=-1)
        return avg

    @staticmethod
    def mean(params):
        return params[0]

    @staticmethod
    def std(params):
        return np.sqrt(params[0])


class DeterministicMSE(ModelObject):

    def __init__(self, x_dim, num_freqs, n=128, n2=64, num_covariates=0):
        """
        Negative Log Likelihood neural network assuming Poisson distribution of x at every point in time.
        Trains using NLL
        :param x_dim: dimension of what will be modeled
        :param num_freqs: int or list. list of the number of frequencies used to model each parameter: [num_rate,]
        :param n: size of 1st hidden layer of NN
        :param n2: size of 2nd hidden layer of NN
        :param num_covariates: number of covariates that will be given as inputs to the NN
        """
        if type(num_freqs) is int:
            num_freqs = [num_freqs] * 1
        super(DeterministicMSE, self).__init__(num_freqs, num_covariates)

        self.l1 = nn.Linear(2 * self.num_freqs[0] + num_covariates, n)
        self.l2 = nn.Linear(n, n2)
        self.l3 = nn.Linear(n2, x_dim)

    def decode(self, w):
        w = w[..., (*self.param_idxs[0], *np.arange(-self.num_covariates, 0))]
        w1 = nn.Tanh()(self.l1(w))
        w2 = nn.Tanh()(self.l2(w1))
        xhat = self.l3(w2)

        return xhat,

    def forward(self, w, data, training_mask=None):
        assert (training_mask is None), "Deterministic forecasts don't support training masks"
        xhat, = self.decode(w)

        losses = (xhat - data.type(w.type())) ** 2
        avg = torch.mean(losses, dim=-1)
        return avg

    @staticmethod
    def mean(params):
        return params[0]

    @staticmethod
    def std(params):
        if isinstance(params[0], torch.Tensor):
            return torch.zeros(params[0].shape)
        return np.zeros(params[0].shape)

    @staticmethod
    def rescale(loc, scale, params):
        """rescales a point forecast distribution with the given parameters so that its scale is
                multiplied by scale and its center is shifted by loc"""
        return params[0] * scale + loc,
