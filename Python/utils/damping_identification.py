import numpy as np
from scipy.signal import find_peaks, savgol_filter

def adaptive_threshold(signal, percentile):
    """Automatically determine a signal threshold based on a given percentile."""
    return np.percentile(abs(signal), percentile)

def random_decrement(signal, threshold, segment_length, fs):
    """Extract free-decay response using RDT."""
    triggers = np.where(signal > threshold)[0]
    segments = []

    for trigger in triggers:
        if trigger + segment_length < len(signal):
            segments.append(signal[trigger:trigger + segment_length])

    if len(segments) == 0:
        raise ValueError("No valid segments found.")

    return np.mean(segments, axis=0)

def RDT(signal, fs, threshold=None, segment_length_sec=1.0):
    """Compute damping ratio using Random Decrement Technique (RDT)."""
    segment_length = int(segment_length_sec * fs)

    if threshold is None:
        threshold = adaptive_threshold(signal, percentile=60)  # Auto threshold

    rd_response = random_decrement(signal, threshold, segment_length, fs)

    # Apply Savitzky-Golay filter to smooth the response
    rd_response_smooth = savgol_filter(rd_response, window_length=5, polyorder=2)

    # Detect peaks with adaptive prominence
    prominence_threshold = max(0.05 * np.max(abs(rd_response_smooth)), np.mean(abs(rd_response_smooth)) * 1)
    peaks, _ = find_peaks(rd_response_smooth, prominence=prominence_threshold, distance=int(fs * 0.02))

    if len(peaks) <= 5:
        return 0  # Return 0 when not enough peaks

    # Compute damping ratio using multiple peak pairs
    zetas = []
    for i in range(len(peaks) - 1):  # Use more peak pairs for robustness
        x_i = abs(rd_response_smooth[peaks[i]])
        x_next = abs(rd_response_smooth[peaks[i+1]]) 
        
        # Ensure x_i and x_next are positive and reasonable
        if x_i <= 0 or x_next <= 0 or x_i <= x_next:
            continue

        delta = np.log(x_i / x_next)
        
        # Compute damping ratio
        zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)

        # Apply realistic constraints on damping ratio**
        if 0 < zeta < 0.05:  # Ensuring physically meaningful damping ratios
            zetas.append(zeta)

    if len(zetas) == 0:
        return 0  # No valid damping ratios found

    return np.mean(zetas)
