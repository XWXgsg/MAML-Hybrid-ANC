%% Title : Multi-channel FxLMS algorithm (1x2x2)
%  K=1 reference, M=2 speakers, L=2 error microphones
%  Based on SimulationPlatformOSPM_MC
%  Author: Adapted from Dongyan Shi's single-channel FxLMS

function [Er, platform] = McFxLMS(Len_Filter, Wc_initial, x_ref, s_path, p_path, s_hat, muw, N_samples)
% Len_Filter  : length of each control sub-filter (I)
% Wc_initial  : initial control filter [I*M*K, 1] = [I*2*1, 1]
%               Stacked as: [w_11; w_21] where w_mk is filter for ref k -> speaker m
% x_ref       : reference signal [N_samples, K]
% s_path      : physical secondary path [J, M, L]
% p_path      : physical primary path   [O, K, L]
% s_hat        : estimated secondary path [J, M, L] (for filtered-x)
% muw         : step size
% N_samples   : number of samples to simulate

    K = 1;   % Number of references
    M = 2;   % Number of speakers
    L = 2;   % Number of error mics
    I = Len_Filter;
    J = size(s_hat, 1);

    % Initialize the simulation platform
    platform = SimulationPlatformOSPM_MC(x_ref, s_path, p_path);

    % Control filter: w_mk each of length I, stacked as [w_11; w_21]
    Wc = Wc_initial;

    % Filtered-reference buffer for each (m, l, k) combination
    % x'_{mlk}(n) = s_hat_{ml} * x_k(n)
    % We need buffer of x_k for filtering through s_hat_{ml}
    % Reference buffer for generating filtered-x
    xBuf = zeros(K, J);  % buffer for x to filter through s_hat

    % Filtered-reference vectors: rf_{mlk}(n) is a vector of length I
    % rf_{mlk}(n) = [x'_{mlk}(n), x'_{mlk}(n-1), ..., x'_{mlk}(n-I+1)]
    rfBuf = zeros(I, M, L, K);  % filtered-reference signal buffer

    % Store filtered-x scalar outputs for building the buffer
    fx_scalar = zeros(M, L, K);  % current filtered-x sample

    % Error recording
    Er = zeros(N_samples, L);

    for tt = 1:N_samples
        % --- Step 1: Get current reference sample ---
        x_n = x_ref(tt, :)';  % [K, 1]

        % --- Step 2: Update reference buffer ---
        xBuf = [x_n', xBuf(:, 1:end-1)];  % [K, J]

        % --- Step 3: Compute filtered-reference signals x'_{mlk}(n) ---
        for k = 1:K
            for m = 1:M
                for l = 1:L
                    fx_scalar(m, l, k) = s_hat(:, m, l)' * xBuf(k, :)';
                end
            end
        end

        % --- Step 4: Update filtered-reference buffers ---
        for k = 1:K
            for m = 1:M
                for l = 1:L
                    rfBuf(:, m, l, k) = [fx_scalar(m, l, k); rfBuf(1:end-1, m, l, k)];
                end
            end
        end

        % --- Step 5: Compute control signals u(m) = sum_k w_mk' * x_buf_k ---
        % We need a separate x buffer of length I for the control filter
        % Actually we can reuse: the control filter convolves with x_k directly
        % We need xCtrlBuf [K, I]
        if tt == 1
            xCtrlBuf = zeros(K, I);
        end
        xCtrlBuf = [x_n', xCtrlBuf(:, 1:end-1)];  % [K, I]

        u = zeros(M, 1);
        for m = 1:M
            for k = 1:K
                idx_start = ((m-1)*K + (k-1)) * I + 1;
                idx_end   = idx_start + I - 1;
                w_mk = Wc(idx_start:idx_end);
                u(m) = u(m) + w_mk' * xCtrlBuf(k, :)';
            end
        end

        % --- Step 6: Run simulation platform (physical plant) ---
        platform.sim(u);

        % --- Step 7: Get error signal ---
        e = platform.e;  % [L, 1]
        Er(tt, :) = e';

        % --- Step 8: Update control filter using McFxLMS ---
        for m = 1:M
            for k = 1:K
                idx_start = ((m-1)*K + (k-1)) * I + 1;
                idx_end   = idx_start + I - 1;
                grad = zeros(I, 1);
                for l = 1:L
                    grad = grad + e(l) * rfBuf(:, m, l, k);
                end
                Wc(idx_start:idx_end) = Wc(idx_start:idx_end) - muw * grad;
            end
        end
    end
end
