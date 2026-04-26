%% MAML-Notch: Multi-channel training with reference / error signal delays
%
%  Two delays are added to simulate real hardware latency:
%
%  d_ref  (samples) — reference signal delay
%    Models: RPM sensor → ADC → controller pipeline latency.
%    In the controller, theta accumulates using the delayed frequency, so
%    sin(theta) / cos(theta) are phase-shifted relative to the true noise.
%    Implementation: sinBuffer and cosBuffer are filled with sin/cos values
%    computed d_ref steps in the past.  Concretely the FIFO buffers hold the
%    reference history, so we simply shift the phase origin by d_ref steps
%    when computing filtered-x:
%      r_sin(n, m, l) = s(:,m,l)' * sinBuf_delayed(n-d_ref)'
%
%  d_err  (samples) — error microphone signal delay
%    Models: error mic → ADC → controller pipeline latency.
%    The LMS update uses e(n - d_err) instead of e(n).
%    Implementation: maintain a FIFO e_buffer [L x (d_err+1)] and pass
%    e_buffer(:, end) (oldest = d_err steps old) to MAML_initial as Di.
%    Stability condition: d_err < fs / (4 * f0).
%
%  Training signals are generated with the same delays so that the
%  trained Phi matches the deployed system exactly.
%
%  Phi_sin [M x L] / Phi_cos [M x L] — one pair per (speaker, mic).
%  weight_table layout [N_freq x (1 + 2*M*L)]:
%    col 1 : frequency
%    rest  : [sin(m=1,l=1), cos(m=1,l=1), sin(m=2,l=1), cos(m=2,l=1), ...
%             sin(m=1,l=2), cos(m=1,l=2), ...]   (l outer, m inner)
%
%  No primary path. K=1 tone per task.
%  Files needed: MAML_notch.m, s.mat
%--------------------------------------------------------------------------
close all; clear; clc;
addpath("class\");
addpath("data\");

%% ========================================================================
%  USER CONFIG
%% ========================================================================
step_size_table = [
%   freq      mu       alpha
    10,     1.2,     0.2;
    30,     1,     0.2;
    50,     1,     0.1;
    80,     1,     0.1;
    100,    0.8,     0.1;
    130,    0.5,     0.05;
    160,    0.3,     0.05;
    200,    0.2,     0.05;
    250,    0.1,     0.05;
    300,    0.1,     0.05;
];

%% =====================  System parameters  ==============================
fs = 16000;
J  = 128;
M  = 2;    % loudspeakers
L  = 2;    % error mics

% ── Delay config ─────────────────────────────────────────────────────────
d_ref = 0;   % reference signal delay (samples) — sensor/ADC latency
d_err = 3;   % error signal delay    (samples) — mic ADC latency
% Stability check: d_err < fs/(4*f0_max); warn if violated.
fprintf('Delays: d_ref=%d samp (%.2f ms)  d_err=%d samp (%.2f ms)\n', ...
        d_ref, d_ref/fs*1000, d_err, d_err/fs*1000);

f_lo   =50;
f_hi   = 200;
f_step = 1;
freq_list = (f_lo : f_step : f_hi)';
N_freq    = length(freq_list);

% Stability warning for error delay
f_crit = fs / (4 * d_err);
if d_err > 0 && any(freq_list > f_crit)
    warning('d_err=%d exceeds stability limit fs/(4*f0) at f0 > %.1f Hz', ...
            d_err, f_crit);
end

%% =====================  Interpolate step sizes  =========================
mu_table    = interp1(step_size_table(:,1), step_size_table(:,2), ...
                      freq_list, 'linear', 'extrap');
alpha_table = interp1(step_size_table(:,1), step_size_table(:,3), ...
                      freq_list, 'linear', 'extrap');
mu_table    = 40 * max(mu_table,    1e-6);
alpha_table = max(alpha_table, 1e-6);

fprintf('Frequency band : %d - %d Hz,  %d points\n', f_lo, f_hi, N_freq);
fprintf('System         : M=%d loudspeakers,  L=%d mics\n', M, L);

%% =====================  Secondary path  =================================
load("s.mat");
if ~isequal(size(s), [J, M, L])
    error('s must be [%d, %d, %d], got [%s]', J, M, L, num2str(size(s)));
end

%% =====================  MAML parameters  ================================
T_train = 2;
N_train = round(T_train * fs);
Len_ep  = 512;
N_epoch = 20000;
lamda_m = 0.99;
eps_m   = 0.5;

%% =====================  Storage  ========================================
weight_table = zeros(N_freq, 1 + 2*M*L);
weight_table(:, 1) = freq_list;
sinBuf_table = zeros(N_freq, J);
cosBuf_table = zeros(N_freq, J);
theta_table  = zeros(N_freq, 1);
final_Er     = zeros(N_freq, 1);

maml_obj = MAML_notch(M, L);

%% =====================  Training loop  ==================================
fprintf('\n========== MAML Training (d_ref=%d, d_err=%d) ==========\n', ...
        d_ref, d_err);
tic;

for fi = 1:N_freq
    f0   = freq_list(fi);
    mu_f = mu_table(fi);
    maml_obj.reset();

    % ── Generate training signals with delays ─────────────────────────────
    %
    % Reference delay d_ref:
    %   The controller receives the reference signal d_ref steps late.
    %   => sinBuffer and cosBuffer at time n hold values from n-d_ref.
    %   => filtered-x r(n,m,l) = s(:,m,l)' * sinBuffer(n-d_ref)'
    %   Implementation: generate the full sin/cos sequence first, then
    %   when computing r, read the buffer d_ref steps behind.
    %
    % Error delay d_err:
    %   LMS update at time n uses e(n - d_err).
    %   Implementation: maintain e_buf [L x (d_err+1)]; pass oldest entry
    %   to MAML as Di.

    theta_acc = 0;

    % Full reference signal history (needed for delayed buffer reads)
    % We pre-generate N_train + d_ref samples so index n-d_ref is valid.
    N_gen   = N_train + max(d_ref, d_err) + J + 10;
    xs_all  = zeros(N_gen, 1);
    xc_all  = zeros(N_gen, 1);
    th_all  = zeros(N_gen, 1);
    for n = 1:N_gen
        theta_acc = mod(theta_acc + 2*pi*f0/fs, 2*pi);
        xs_all(n) = sin(theta_acc);
        xc_all(n) = cos(theta_acc);
        th_all(n) = theta_acc;
    end

    % Disturbance (no primary path): d(l,n) = sin(theta(n))
    % (same on all mics; add spatial variation here if needed)
    d_tr     = zeros(N_train, L);
    r_sin_tr = zeros(N_train, M, L);
    r_cos_tr = zeros(N_train, M, L);
    e_buf_tr = zeros(L, d_err + 1);   % FIFO for error delay: col 1 = newest

    % sinBuf / cosBuf: filled with d_ref-delayed reference
    sinBuf = zeros(1, J);
    cosBuf = zeros(1, J);

    for n = 1:N_train
        % True disturbance at mic (no primary path, no error delay in d_tr)
        d_n = xs_all(n) * ones(1, L);   % [1 x L]
        d_tr(n, :) = d_n;

        % Reference at controller (delayed by d_ref)
        n_ref = max(n - d_ref, 1);
        xs_ref = xs_all(n_ref);
        xc_ref = xc_all(n_ref);

        % Update FIFO buffers with delayed reference
        sinBuf = [xs_ref, sinBuf(1:end-1)];
        cosBuf = [xc_ref, cosBuf(1:end-1)];

        % Filtered-x using delayed reference buffers
        for m = 1:M
            for l = 1:L
                r_sin_tr(n, m, l) = s(:, m, l)' * sinBuf';
                r_cos_tr(n, m, l) = s(:, m, l)' * cosBuf';
            end
        end

        % Error signal seen by controller (delayed by d_err)
        % During training we compute the "true" error from the
        % undisturbed disturbance (no control output yet in training).
        % True e(n) = d(n)  (no control; primary disturbance only).
        % Delayed error: e(n - d_err) -> use e_buf
        e_true = d_n';    % [L x 1], true error at this step

        % Shift error FIFO: push new true error, oldest = delayed error
        e_buf_tr = [e_true, e_buf_tr(:, 1:end-1)];   % col 1 = newest

        % The MAML training data uses the delayed error as Di
        % (e_buf_tr(:, end) = error d_err steps ago)
        % We store the disturbance for Di below using d_tr directly
        % shifted by d_err in the episode sampling stage.
    end

    % Record steady-state buffers and phase (at end of training, no delay)
    theta_table(fi)    = th_all(N_train);
    sinBuf_table(fi,:) = sinBuf;
    cosBuf_table(fi,:) = cosBuf;

    % ── Random episode sampling with aligned delays ───────────────────────
    % For each episode: sample a window of length Len_ep.
    % Fx_sin/Fx_cos come from r_sin_tr/r_cos_tr (already delayed by d_ref).
    % Di comes from d_tr shifted forward by d_err (to match e(n-d_err)):
    %   Di(t) = d_tr(t + d_err)  — the error the controller sees at time t
    %           is d_tr at d_err steps later (controller is d_err behind).
    % Clamp indices to valid range.

    Fx_sin_ep = zeros(Len_ep, N_epoch, M, L);
    Fx_cos_ep = zeros(Len_ep, N_epoch, M, L);
    Di_ep     = zeros(Len_ep, N_epoch, L);

    for jj = 1:N_epoch
        idx_end = randi([Len_ep + J, N_train - d_err]);
        idx_fx  = (idx_end - Len_ep + 1) : idx_end;
        idx_di  = idx_fx + d_err;   % Di is d_err steps ahead of Fx

        Di_ep(:, jj, :) = d_tr(idx_di, :);
        for m = 1:M
            for l = 1:L
                Fx_sin_ep(:, jj, m, l) = r_sin_tr(idx_fx, m, l);
                Fx_cos_ep(:, jj, m, l) = r_cos_tr(idx_fx, m, l);
            end
        end
    end

    % ── MAML epochs ───────────────────────────────────────────────────────
    Er_last = 0;
    for jj = 1:N_epoch
        Er_last = maml_obj.MAML_initial( ...
            squeeze(Fx_sin_ep(:, jj, :, :)), ...
            squeeze(Fx_cos_ep(:, jj, :, :)), ...
            squeeze(Di_ep(:, jj, :)),         ...
            mu_f, lamda_m, eps_m);
    end
    final_Er(fi) = Er_last;

    % ── Store weights ─────────────────────────────────────────────────────
    for l = 1:L
        for m = 1:M
            col = 2 + (l-1)*2*M + (m-1)*2;
            weight_table(fi, col)   = maml_obj.Phi_sin(m, l);
            weight_table(fi, col+1) = maml_obj.Phi_cos(m, l);
        end
    end

    if mod(fi, 50) == 0 || fi == 1 || fi == N_freq
        phi_str = '';
        for m = 1:M
            for l = 1:L
                phi_str = [phi_str, sprintf('  (m%d,l%d)[%+.3f,%+.3f]', m, l, ...
                    maml_obj.Phi_sin(m,l), maml_obj.Phi_cos(m,l))]; %#ok
            end
        end
        fprintf('  [%3d/%3d] %4dHz  mu=%.4f%s  Er=%.3e\n', ...
            fi, N_freq, f0, mu_f, phi_str, Er_last);
    end
end

train_time = toc;
fprintf('Training complete. %.1f s\n', train_time);

%% =====================  Save  ===========================================
save('MAML_weight_table_manual.mat', ...
    'weight_table', 'freq_list', 'fs', 'J', 's', ...
    'mu_table', 'alpha_table', 'step_size_table', ...
    'theta_table', 'sinBuf_table', 'cosBuf_table', ...
    'lamda_m', 'eps_m', 'N_epoch', 'M', 'L', 'd_ref', 'd_err');
fprintf('Saved: MAML_weight_table_manual.mat\n');
fprintf('  weight_table : [%d x %d]  d_ref=%d  d_err=%d\n', ...
        N_freq, 1+2*M*L, d_ref, d_err);

%% =====================  Print sample  ===================================
fprintf('\n===== Weight Table Sample =====\n');
hdr = sprintf('%5s  %6s', 'Freq', 'alpha');
for l = 1:L
    for m = 1:M
        hdr = [hdr, sprintf('  sin(m%d,l%d)  cos(m%d,l%d)', m,l,m,l)]; %#ok
    end
end
fprintf('%s\n%s\n', hdr, repmat('-',1,length(hdr)));
for si = round(linspace(1, N_freq, min(20, N_freq)))
    fprintf('%5.0f  %6.4f', freq_list(si), alpha_table(si));
    for l = 1:L
        for m = 1:M
            col = 2 + (l-1)*2*M + (m-1)*2;
            fprintf('  %+10.6f  %+10.6f', weight_table(si,col), weight_table(si,col+1));
        end
    end
    fprintf('\n');
end

%% =====================  Plots  ==========================================
figure('Name','Phi vs Freq');
panel = 0;
for l = 1:L
    for m = 1:M
        panel = panel + 1;
        col   = 2 + (l-1)*2*M + (m-1)*2;
        subplot(M*L, 2, 2*panel-1);
        plot(freq_list, weight_table(:,col),   'b.-', 'MarkerSize',3);
        ylabel('\Phi_{sin}');
        title(sprintf('m=%d,l=%d  \\Phi_{sin}  (d_{ref}=%d,d_{err}=%d)', m,l,d_ref,d_err));
        grid on;
        subplot(M*L, 2, 2*panel);
        plot(freq_list, weight_table(:,col+1), 'r.-', 'MarkerSize',3);
        ylabel('\Phi_{cos}');
        title(sprintf('m=%d,l=%d  \\Phi_{cos}', m,l)); grid on;
    end
end
xlabel('Frequency (Hz)');

figure('Name','Training error');
plot(freq_list, abs(final_Er), 'k.-', 'MarkerSize',3);
xlabel('Freq (Hz)'); ylabel('|Er|');
title(sprintf('Final training error  (d_{ref}=%d, d_{err}=%d)', d_ref, d_err));
grid on;

fprintf('\n=== Training Done ===\n');