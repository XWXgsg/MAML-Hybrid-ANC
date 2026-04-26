%% MAML-Notch: Single-channel training with reference / error signal delays
%
%  Single channel: M=1 loudspeaker, L=1 error mic.
%  Trains scalar Phi_sin / Phi_cos for each frequency in freq_list.
%
%  Two delays simulate real hardware ADC/pipeline latency:
%
%  d_ref (samples) — reference signal delay
%    The controller receives the reference (RPM/frequency) d_ref steps late.
%    => sinBuffer / cosBuffer are filled with d_ref-delayed sin/cos values.
%    => filtered-x r(n) = s(:)' * sinBuffer(n-d_ref)'
%
%  d_err (samples) — error microphone signal delay
%    The LMS update uses e(n - d_err) instead of e(n).
%    => episode Di is shifted +d_err relative to Fx to align properly.
%    Stability condition: d_err < fs / (4 * f0).
%
%  weight_table layout [N_freq x 3]:
%    col 1 : frequency
%    col 2 : Phi_sin
%    col 3 : Phi_cos
%
%  No primary path. K=1 tone per task.
%  Files needed: MAML_notch.m, s.mat  (s must be [J,1,1] or [J,1])
%--------------------------------------------------------------------------
close all; clear; clc;
addpath("class\");
addpath("data\");

%% ========================================================================
%  USER CONFIG
%% ========================================================================
%% ========================================================================
step_size_table = [
%   freq      mu       alpha
    10,     0.05,     0.002;
    30,     0.05,     0.002;
    50,     0.05,     0.002;
    80,     0.05,     0.002;
    100,    0.03,     0.001;
    130,    0.03,     0.001;
    160,    0.03,     0.01;
    200,    0.02,     0.01;
    250,    0.01,     0.01;
    300,    0.01,     0.01;
];

%% =====================  System parameters  ==============================
fs = 16000;
J  = 256;
M  = 1;    % single loudspeaker
L  = 1;    % single error mic

% ── Delay config ─────────────────────────────────────────────────────────
d_ref = 0;   % reference signal delay (samples)
d_err = 0;   % error signal delay     (samples)
fprintf('Delays: d_ref=%d samp (%.2f ms)  d_err=%d samp (%.2f ms)\n', ...
        d_ref, d_ref/fs*1000, d_err, d_err/fs*1000);

f_lo   = 20;
f_hi   = 200;
f_step = 1;
freq_list = (f_lo : f_step : f_hi)';
N_freq    = length(freq_list);

% Stability warning
if d_err > 0
    f_crit = fs / (4 * d_err);
    if any(freq_list > f_crit)
        warning('d_err=%d: stability limit fs/(4*d_err) = %.1f Hz < f_hi = %.1f Hz', ...
                d_err, f_crit, f_hi);
    end
end

%% =====================  Interpolate step sizes  =========================
mu_table    = interp1(step_size_table(:,1), step_size_table(:,2), ...
                      freq_list, 'linear', 'extrap');
alpha_table = interp1(step_size_table(:,1), step_size_table(:,3), ...
                      freq_list, 'linear', 'extrap');
mu_table    = max(mu_table,    1e-6);
alpha_table = max(alpha_table, 1e-6);

fprintf('Frequency band : %d - %d Hz,  %d points\n', f_lo, f_hi, N_freq);

%% =====================  Secondary path  =================================
load("SecondaryPath_2x2_256order.mat");
% Accept [J,1,1], [J,1], or [J] — normalise to column vector
s_coef = S11;
s_coef = s_coef(:);   % [J x 1]
if length(s_coef) ~= J
    error('s_coef must have %d taps, got %d', J, length(s_coef));
end
s_3d = reshape(s_coef, [J, 1, 1]);   % for SimulationPlatform

%% =====================  MAML parameters  ================================
T_train = 2;
N_train = round(T_train * fs);
Len_ep  = 512;
N_epoch = 40000;
lamda_m = 0.99;
eps_m   = 0.5;

%% =====================  Storage  ========================================
weight_table = zeros(N_freq, 3);   % [freq, Phi_sin, Phi_cos]
weight_table(:,1) = freq_list;
sinBuf_table = zeros(N_freq, J);
cosBuf_table = zeros(N_freq, J);
theta_table  = zeros(N_freq, 1);
final_Er     = zeros(N_freq, 1);

maml_obj = MAML_notch();

%% =====================  Training loop  ==================================
fprintf('\n========== MAML Training (M=1, L=1, d_ref=%d, d_err=%d) ==========\n', ...
        d_ref, d_err);
tic;

for fi = 1:N_freq
    f0   = freq_list(fi);
    mu_f = mu_table(fi);
    maml_obj.reset();

    % ── Generate training signals ─────────────────────────────────────────
    N_gen    = N_train + max(d_ref, d_err) + J + 10;
    theta_acc = 0;
    xs_all   = zeros(N_gen, 1);
    xc_all   = zeros(N_gen, 1);
    th_all   = zeros(N_gen, 1);
    for n = 1:N_gen
        theta_acc = mod(theta_acc + 2*pi*f0/fs, 2*pi);
        xs_all(n) = sin(theta_acc);
        xc_all(n) = cos(theta_acc);
        th_all(n) = theta_acc;
    end

    d_tr     = zeros(N_train, 1);   % disturbance (= sin, no primary path)
    r_sin_tr = zeros(N_train, 1);   % filtered-x sin
    r_cos_tr = zeros(N_train, 1);   % filtered-x cos

    sinBuf = zeros(1, J);
    cosBuf = zeros(1, J);

    for n = 1:N_train
        d_tr(n) = xs_all(n);

        % Delayed reference
        n_ref  = max(n - d_ref, 1);
        xs_ref = xs_all(n_ref);
        xc_ref = xc_all(n_ref);

        sinBuf = [xs_ref, sinBuf(1:end-1)];
        cosBuf = [xc_ref, cosBuf(1:end-1)];

        r_sin_tr(n) = s_coef' * sinBuf';
        r_cos_tr(n) = s_coef' * cosBuf';
    end

    % Record steady-state buffer and phase
    theta_table(fi)    = th_all(N_train);
    sinBuf_table(fi,:) = sinBuf;
    cosBuf_table(fi,:) = cosBuf;

    % ── Random episode sampling ───────────────────────────────────────────
    % Fx from r_sin_tr/r_cos_tr (delayed by d_ref).
    % Di  from d_tr shifted +d_err (controller sees error d_err steps late).
    Fx_sin_ep = zeros(Len_ep, N_epoch);
    Fx_cos_ep = zeros(Len_ep, N_epoch);
    Di_ep     = zeros(Len_ep, N_epoch);

    for jj = 1:N_epoch
        idx_end = randi([Len_ep + J, N_train - d_err]);
        idx_fx  = (idx_end - Len_ep + 1) : idx_end;
        idx_di  = idx_fx + d_err;
        Fx_sin_ep(:, jj) = r_sin_tr(idx_fx);
        Fx_cos_ep(:, jj) = r_cos_tr(idx_fx);
        Di_ep(:, jj)     = d_tr(idx_di);
    end

    % ── MAML epochs ───────────────────────────────────────────────────────
    Er_last = 0;
    for jj = 1:N_epoch
        Er_last = maml_obj.MAML_initial( ...
            Fx_sin_ep(:, jj), Fx_cos_ep(:, jj), Di_ep(:, jj), ...
            mu_f, lamda_m, eps_m);
    end
    final_Er(fi) = Er_last;

    weight_table(fi, 2) = maml_obj.Phi_sin;
    weight_table(fi, 3) = maml_obj.Phi_cos;

    if mod(fi, 50) == 0 || fi == 1 || fi == N_freq
        fprintf('  [%3d/%3d] %4dHz  mu=%.4f  Phi=[%+.4f,%+.4f]  Er=%.3e\n', ...
            fi, N_freq, f0, mu_f, maml_obj.Phi_sin, maml_obj.Phi_cos, Er_last);
    end
end

train_time = toc;
fprintf('Training complete. %.1f s\n', train_time);

%% =====================  Save  ===========================================
save('MAML_weight_table_manual.mat', ...
    'weight_table', 'freq_list', 'fs', 'J', 's_coef', 's_3d', ...
    'mu_table', 'alpha_table', 'step_size_table', ...
    'theta_table', 'sinBuf_table', 'cosBuf_table', ...
    'lamda_m', 'eps_m', 'N_epoch', 'M', 'L', 'd_ref', 'd_err');
fprintf('Saved: MAML_weight_table_manual.mat\n');
fprintf('  weight_table : [%d x 3]  d_ref=%d  d_err=%d\n', N_freq, d_ref, d_err);

%% =====================  Print sample  ===================================
fprintf('\n===== Weight Table Sample =====\n');
fprintf('%5s  %6s  %+12s  %+12s\n', 'Freq','alpha','Phi_sin','Phi_cos');
fprintf('%s\n', repmat('-',1,42));
for si = round(linspace(1, N_freq, min(20, N_freq)))
    fprintf('%5.0f  %6.4f  %+12.6f  %+12.6f\n', ...
        freq_list(si), alpha_table(si), weight_table(si,2), weight_table(si,3));
end

%% =====================  Plots  ==========================================
figure('Name','Phi vs Freq');
subplot(2,1,1);
plot(freq_list, weight_table(:,2), 'b.-', 'MarkerSize',3);
ylabel('\Phi_{sin}');
title(sprintf('\\Phi_{sin}  (d_{ref}=%d, d_{err}=%d)', d_ref, d_err)); grid on;
subplot(2,1,2);
plot(freq_list, weight_table(:,3), 'r.-', 'MarkerSize',3);
xlabel('Frequency (Hz)'); ylabel('\Phi_{cos}'); grid on;

figure('Name','Training error');
plot(freq_list, abs(final_Er), 'k.-', 'MarkerSize',3);
xlabel('Freq (Hz)'); ylabel('|Er|');
title(sprintf('Final training error  (d_{ref}=%d, d_{err}=%d)', d_ref, d_err));
grid on;

fprintf('\n=== Training Done ===\n');
