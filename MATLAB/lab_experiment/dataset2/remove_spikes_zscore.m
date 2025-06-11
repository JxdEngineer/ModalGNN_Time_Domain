function clean_signal = remove_spikes_zscore(signal, z_threshold)
    % Remove spikes based on z-score threshold
    z_scores = (signal - mean(signal)) / std(signal); % Compute z-scores
    spikes = abs(z_scores) > z_threshold; % Identify spikes
    clean_signal = signal;
    clean_signal(spikes) = interp1(find(~spikes), signal(~spikes), find(spikes), 'linear', 'extrap'); % Interpolate spikes
end