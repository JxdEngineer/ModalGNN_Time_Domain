import numpy as np

def MAC(x, y):
    assert x.shape == y.shape, "Mode shapes must have the same shape"
    numerator = np.abs(np.dot(x, y.T))**2
    denominator_x = np.dot(x, x.T)
    denominator_y = np.dot(y, y.T)
    mac = numerator / (denominator_x * denominator_y)
    return mac

def match_mode(freq_true, freq_pred, phi_true, phi_pred):
    freq_pred_match = []
    MAC_pred_match = []
    for i in range(5):
        if np.min(np.abs(freq_pred - freq_true[i])) > 1: # this mode is not identified, so record empty data
            freq_pred_match.append(0)
            MAC_pred_match.append(0)
        else:
            if np.sum(np.abs(freq_pred - freq_true[i]) < 1) > 1: # more than one possible mode is identified, within the frequency boundary of +-1
                MAC_comparison = []
                match_indexes = np.where(np.abs(freq_pred - freq_true[i]) < 1)[0]
                for j in range(np.sum(np.abs(freq_pred - freq_true[i]) < 1)): # correlate the pair with highest MAC
                    MAC_comparison.append(MAC(phi_true[:, i], phi_pred[:, match_indexes[j]]))
                match_index = match_indexes[np.argmax(np.array(MAC_comparison))]
                MAC_pred_match.append(MAC(phi_true[:, i], phi_pred[:, match_index]))
                freq_pred_match.append(freq_pred[match_index])
            else:
                match_index = np.abs(freq_pred - freq_true[i]).argmin()
                MAC_pred_match.append(MAC(phi_true[:, i], phi_pred[:, match_index]))
                freq_pred_match.append(freq_pred[match_index])
    return freq_pred_match, MAC_pred_match