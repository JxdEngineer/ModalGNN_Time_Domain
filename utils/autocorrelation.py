import torch

def autocorrelation(signal):
    N = signal.size(0)
    signal_mean = signal.mean(dim=0)
    signal_centered = signal - signal_mean
    autocorr = torch.zeros_like(signal)
    for lag in range(N):
        autocorr[lag] = (signal_centered[:N-lag] * signal_centered[lag:]).sum(dim=0) / (N - lag)
    return autocorr