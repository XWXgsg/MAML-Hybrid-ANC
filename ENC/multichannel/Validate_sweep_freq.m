%% MAML-Notch: Sweep validation — multi-channel residual LMS with delays
%
%  Extends the single-channel residual-LMS sweep validation to:
%    M loudspeakers, L error mics (multi-channel).
%    Optional reference signal delay d_ref and error signal delay d_err
%    to simulate real hardware ADC/pipeline latency.
%
%  ── Residual LMS ──────────────────────────────────────────────────────
%  w_base [2,1,M]: interpolated optimum at the current (delayed) frequency,
%                  updated every sample.
%  delta_w [2,1,M]: residual accumulated by LMS.
%  Each step:
%    delta_w  = w_prev - w_base_prev
%    w        = w_base_new + delta_w   <- rebased onto new optimum
%    iterate() runs normally            <- LMS updates w (i.e. delta_w)
%
%  ── Delays ────────────────────────────────────────────────────────────
%  d_ref (samples): controller receives freq_inst d_ref steps late.
%    -> ctrl.freqs = freq_buf(end)  (FIFO tail)
%    -> w_base also keyed on delayed frequency (training/validation match)
%  d_err (samples): controller receives error signal d_err steps late.
%    -> iterate() called with e_buf(:,end)  (FIFO tail)
%    Stability: d_err < fs/(4*f0).
%
%  ── weight_table layout [N_freq x (1 + 2*M*L)] ────────────────────────
%  col 1 : freq
%  rest  : [sin(m=1,l=1), cos(m=1,l=1), sin(m=2,l=1), cos(m=2,l=1), ...
%           sin(m=1,l=2), cos(m=1,l=2), ...]  (l outer, m inner)
%  w(1,1,m) = sum_l Phi_sin(m,l),  w(2,1,m) = sum_l Phi_cos(m,l)
%
%  No primary path.  K=1 tone.
%  Files needed: NotchController.m, SimulationPlatform.m
%               + MAML_weight_table_manual.mat
%--------------------------------------------------------------------------
close all; clear; clc;

%% =====================  Load  ===========================================
load('MAML_weight_table_manual.mat');
fprintf('Loaded: %d freqs (%.1f - %.1f Hz)  M=%d  L=%d\n', ...
        length(freq_list), freq_list(1), freq_list(end), M, L);

N_freq = length(freq_list);

% ── Delay config ─────────────────────────────────────────────────────────
% Read from mat if saved during training; override here if needed.
if ~exist('d_ref','var'); d_ref = 0; end
if ~exist('d_err','var'); d_err = 0; end
% d_ref = 3;   % uncomment to override
% d_err = 2;
fprintf('Delays: d_ref=%d samp (%.2f ms)   d_err=%d samp (%.2f ms)\n', ...
        d_ref, d_ref/fs*1000, d_err, d_err/fs*1000);

% Stability warning for error delay
if d_err > 0
    f_crit = fs / (4 * d_err);
    if freq_list(end) > f_crit
        warning('d_err=%d: stability limit %.1f Hz < f_end %.1f Hz', ...
                d_err, f_crit, freq_list(end));
    end
end

% ── Secondary path ───────────────────────────────────────────────────────
if exist('s','var') && isequal(size(s), [J, M, L])
    s_3d = s;
elseif exist('s_coef','var')
    s_3d = reshape(s_coef(:), [J, M, L]);
    warning('s_coef reshaped to [%d,%d,%d].', J, M, L);
else
    error('No secondary path variable found.');
end

%% =====================  Build w_init table (sum_l Phi per speaker)  =====
% w_sin_init [N_freq x M]: w_sin_init(fi,m) = sum_l Phi_sin(m,l)
% col of (m,l): 2 + (l-1)*2*M + (m-1)*2  -> Phi_sin
%               2 + (l-1)*2*M + (m-1)*2+1 -> Phi_cos
w_sin_init = zeros(N_freq, M);
w_cos_init = zeros(N_freq, M);
for l = 1:L
    for m = 1:M
        col = 2 + (l-1)*2*M + (m-1)*2;
        w_sin_init(:,m) = w_sin_init(:,m) + weight_table(:, col);
        w_cos_init(:,m) = w_cos_init(:,m) + weight_table(:, col+1);
    end
end

%% =====================  Build interpolation functions  ==================
% One handle per speaker; returns scalar for scalar input f.
interp_w_sin = cell(M, 1);
interp_w_cos = cell(M, 1);
for m = 1:M
    interp_w_sin{m} = @(f) interp1(freq_list, w_sin_init(:,m), f, 'linear', 'extrap');
    interp_w_cos{m} = @(f) interp1(freq_list, w_cos_init(:,m), f, 'linear', 'extrap');
end
interp_alpha = @(f) interp1(freq_list, alpha_table, f, 'linear', 'extrap');

fprintf('Interpolation ready: %.1f - %.1f Hz, step %.1f Hz\n', ...
        freq_list(1), freq_list(end), mean(diff(freq_list)));

%% =====================  Sweep config  ===================================
f_start    = freq_list(1);
f_end      = freq_list(end);
sweep_rate = 10;                     % Hz/s
bin_width  = 4;                      % Hz per NR bin

T_sweep   = (f_end - f_start) / sweep_rate;
N_sweep   = round(T_sweep * fs);
t_sweep   = (0:N_sweep-1)' / fs;
freq_inst = min(f_start + sweep_rate * t_sweep, f_end);  % true freq [N_sweep x 1]

fprintf('Sweep: %.1f -> %.1f Hz, %.1f Hz/s, %.2f s\n', ...
        f_start, f_end, sweep_rate, T_sweep);

%% =====================  Generate disturbance  ===========================
% Analytic chirp — no mod() wrapping.  Same signal on all L mics.
k_sweep     = sweep_rate;
phase_chirp = 2*pi * (f_start * t_sweep + 0.5 * k_sweep * t_sweep.^2);
d_mono      = sin(phase_chirp);              % [N_sweep x 1]
d_sweep     = repmat(d_mono, 1, L);          % [N_sweep x L]

%% =====================  Create controllers  =============================
% Use delayed starting frequency for init (= f_start when d_ref=0)
f_init  = f_start;
alpha_0 = 20*interp_alpha(f_init);

% --- Baseline: pure FxLMS cold start ------------------------------------
ctrl_fx = NotchController(fs, f_init, alpha_0, M, L, J);
ctrl_fx.s = s_3d;

% --- MAML residual LMS --------------------------------------------------
ctrl_ml = NotchController(fs, f_init, alpha_0, M, L, J);
ctrl_ml.s = s_3d;

% Initial w [2,1,M] = sum_l Phi at f_init  (delta_w = 0)
w_init      = get_w_base(f_init, interp_w_sin, interp_w_cos, M);
ctrl_ml.w   = w_init;

% Pre-warm sinBuffer [M,J] and cosBuffer [M,J] at f_init
j_vec      = (1:J)';
theta_pre  = -j_vec * 2*pi * f_init / fs;   % [J x 1] unwrapped
for m = 1:M
    ctrl_ml.sinBuffer(m,:) = sin(theta_pre)';
    ctrl_ml.cosBuffer(m,:) = cos(theta_pre)';
end
ctrl_ml.theta = zeros(M, 1);   % aligned with chirp phase at t=0

% w_base_prev [2,1,M] — baseline of the previous step
w_base_prev = w_init;

fprintf('\nMAML init at f_start = %.1f Hz:\n', f_start);
for m = 1:M
    fprintf('  m=%d: w_sin=%+.6f  w_cos=%+.6f\n', m, ...
            ctrl_ml.w(1,1,m), ctrl_ml.w(2,1,m));
end
fprintf('  delta_w = 0  (starts at interpolated optimum)\n');

plat_fx = SimulationPlatform(d_sweep, s_3d, M, L);
plat_ml = SimulationPlatform(d_sweep, s_3d, M, L);

% ── Pre-warm plat_ml.uBuffer ─────────────────────────────────────────────
% Problem: ctrl_ml starts with a non-zero w_init and a pre-warmed
% sinBuffer/cosBuffer, so it produces a non-zero u at step n=1.
% But plat_ml.uBuffer is all-zeros (cold start), so the secondary-path
% output y = S * uBuffer is wrong for the first J steps, causing a
% transient error spike (noise INCREASE) at the very start.
%
% Fix: pre-fill plat_ml.uBuffer with the control outputs that w_init
% would have produced at t < 0, assuming constant f_init.
% u(t = -k/fs) = sum_m [ w_sin(m)*sin(theta_k) + w_cos(m)*cos(theta_k) ]
% theta_k = -(k-1) * 2*pi*f_init/fs   (most recent: k=1, oldest: k=J)
%
% uBuffer layout [M x J]: col 1 = most recent u, col J = oldest u.
dth_init = 2*pi * f_init / fs;
for k = 1:J
    theta_k = -(k-1) * dth_init;        % phase at t = -(k-1)/fs
    u_pre_k = zeros(M, 1);
    for m = 1:M
        u_pre_k(m) = ctrl_ml.w(1,1,m) * sin(theta_k) ...
                   + ctrl_ml.w(2,1,m) * cos(theta_k);
    end
    plat_ml.uBuffer(:, k) = u_pre_k;    % col k: k steps before t=0
end
% Recompute plat_ml.y with the pre-warmed buffer so that plat_ml.e at
% the first call to sim() already reflects the correct y history.
plat_ml.ufilter2y();
% Note: plat_ml.e is still zeros here (no SoundSuperposition yet).
% The first sim() call at n=1 will update e correctly: e = d(1) - y_new.

fprintf('  plat_ml.uBuffer pre-warmed (%d taps at f_init=%.1f Hz)\n', J, f_init);

%% =====================  Delay FIFOs  ====================================
% freq_buf(1)=newest, freq_buf(end)=oldest (d_ref steps old)
freq_buf    = f_init * ones(d_ref + 1, 1);

% e_buf_fx/ml(:,1)=newest, (:,end)=oldest (d_err steps old)
e_buf_fx    = zeros(L, d_err + 1);
e_buf_ml    = zeros(L, d_err + 1);

%% =====================  Run simulation  =================================
fprintf('\nRunning ...\n');

e_fxlms    = zeros(N_sweep, L);
e_maml     = zeros(N_sweep, L);
w_traj_fx  = zeros(N_sweep, 2);   % speaker m=1 representative
w_traj_ml  = zeros(N_sweep, 2);
dw_traj_ml = zeros(N_sweep, 2);

tic;
for n = 1:N_sweep

    % ── Update reference frequency FIFO ──────────────────────────────────
    freq_buf = [freq_inst(n); freq_buf(1:end-1)];
    f_ctrl   = freq_buf(end);           % delayed freq seen by controller

    alpha_now = 20 * interp_alpha(f_ctrl);

    % Per-step freq/alpha update using DELAYED frequency
    ctrl_fx.freqs = f_ctrl;
    ctrl_fx.alpha = alpha_now * ones(1, L);
    ctrl_ml.freqs = f_ctrl;
    ctrl_ml.alpha = alpha_now * ones(1, L);

    % ---- Baseline: pure FxLMS (cold start) ------------------------------
    e_delayed_fx = e_buf_fx(:, end);          % [L x 1], d_err steps old
    u_fx = ctrl_fx.iterate(e_delayed_fx);
    plat_fx.sim(u_fx);
    e_true_fx = plat_fx.e;                    % [L x 1] true error
    e_buf_fx  = [e_true_fx, e_buf_fx(:,1:end-1)];   % push into FIFO

    e_fxlms(n,:)   = e_true_fx';
    w_traj_fx(n,:) = ctrl_fx.w(:,1,1)';

    % ---- MAML: residual LMS  --------------------------------------------
    % Step 1: new interpolated baseline keyed on delayed frequency [2,1,M]
    w_base_now = get_w_base(f_ctrl, interp_w_sin, interp_w_cos, M);

    % Step 2: carry residual delta_w onto new baseline
    delta_w   = ctrl_ml.w - w_base_prev;
    ctrl_ml.w = w_base_now + delta_w;

    % Step 3: LMS iterate with delayed error
    e_delayed_ml = e_buf_ml(:, end);
    u_ml = ctrl_ml.iterate(e_delayed_ml);
    plat_ml.sim(u_ml);
    e_true_ml = plat_ml.e;
    e_buf_ml  = [e_true_ml, e_buf_ml(:,1:end-1)];

    e_maml(n,:)     = e_true_ml';
    w_traj_ml(n,:)  = ctrl_ml.w(:,1,1)';
    dw_traj_ml(n,:) = (ctrl_ml.w(:,1,1) - w_base_now(:,1,1))';

    % Step 4: store baseline for next step
    w_base_prev = w_base_now;
end
fprintf('Done. %.1f s\n', toc);

%% =====================  Per-bin NR (averaged over L mics)  ==============
bin_edges   = (f_start : bin_width : f_end + bin_width)';
N_bins      = length(bin_edges) - 1;
bin_centers = bin_edges(1:end-1) + bin_width/2;
bin_idx     = min(floor((freq_inst - f_start) / bin_width) + 1, N_bins);

NR_fx_bin = nan(N_bins, 1);
NR_ml_bin = nan(N_bins, 1);
for bi = 1:N_bins
    mask = (bin_idx == bi);
    if sum(mask) < 10; continue; end
    dp_avg = mean(mean(d_sweep(mask,:).^2, 2));
    ep_fx  = mean(mean(e_fxlms(mask,:).^2, 2));
    ep_ml  = mean(mean(e_maml(mask,:).^2,  2));
    NR_fx_bin(bi) = 10*log10(ep_fx / dp_avg);
    NR_ml_bin(bi) = 10*log10(ep_ml / dp_avg);
end

%% =====================  Print  ==========================================
fprintf('\n===== Per-bin NR (d_ref=%d, d_err=%d, avg %d mics) =====\n', ...
        d_ref, d_err, L);
fprintf('%8s | %8s  %8s | %8s\n','Bin(Hz)','NR_FxLMS','NR_MAML','Improve');
fprintf('%s\n', repmat('-',1,42));
for bi = 1:N_bins
    if isnan(NR_fx_bin(bi)); continue; end
    fprintf('%4.0f-%3.0f | %+7.1fdB %+7.1fdB | %+6.1fdB\n', ...
        bin_edges(bi), bin_edges(bi+1), ...
        NR_fx_bin(bi), NR_ml_bin(bi), ...
        NR_fx_bin(bi) - NR_ml_bin(bi));
end
valid = ~isnan(NR_fx_bin);
fprintf('\nFull sweep — FxLMS: %.1f dB   MAML: %.1f dB   Advantage: %.1f dB\n', ...
        mean(NR_fx_bin(valid)), mean(NR_ml_bin(valid)), ...
        mean(NR_fx_bin(valid)) - mean(NR_ml_bin(valid)));

fprintf('\nPer-mic NR:\n');
for l = 1:L
    dp_l  = mean(d_sweep(:,l).^2);
    nr_fx = 10*log10(mean(e_fxlms(:,l).^2) / dp_l);
    nr_ml = 10*log10(mean(e_maml(:,l).^2)  / dp_l);
    fprintf('  Mic l=%d — FxLMS: %+.1f dB   MAML: %+.1f dB   Adv: %+.1f dB\n', ...
            l, nr_fx, nr_ml, nr_fx - nr_ml);
end

%% =====================  Plots  ==========================================
win  = round(0.02 * fs);
win2 = round(0.05 * fs);
ttl  = sprintf('(d_{ref}=%d, d_{err}=%d, M=%d, L=%d)', d_ref, d_err, M, L);

e_fx_avg = mean(e_fxlms.^2, 2);
e_ml_avg = mean(e_maml.^2,  2);
d_avg    = mean(d_sweep.^2,  2);

figure('Name','Error Power (time)');
plot(t_sweep, 10*log10(movmean(d_avg,    win)+eps), 'Color',[.7 .7 .7]); hold on;
plot(t_sweep, 10*log10(movmean(e_fx_avg, win)+eps), 'b', 'LineWidth',1);
plot(t_sweep, 10*log10(movmean(e_ml_avg, win)+eps), 'r', 'LineWidth',1);
xlabel('Time (s)'); ylabel('dB');
title(['Error Power vs Time ' ttl]);
legend('Off','FxLMS (cold start)','MAML (residual LMS)'); grid on;

figure('Name','Error Power (freq)');
plot(freq_inst, 10*log10(movmean(d_avg,    win2)+eps), 'Color',[.7 .7 .7]); hold on;
plot(freq_inst, 10*log10(movmean(e_fx_avg, win2)+eps), 'b', 'LineWidth',.8);
plot(freq_inst, 10*log10(movmean(e_ml_avg, win2)+eps), 'r', 'LineWidth',.8);
xlabel('Frequency (Hz)'); ylabel('dB');
title(['Error Power vs Frequency ' ttl]);
legend('Off','FxLMS (cold start)','MAML (residual LMS)'); grid on;

figure('Name','Error Power per mic');
tl = tiledlayout(L,1,'TileSpacing','compact','Padding','compact');
tl.Title.String = ['Error Power per mic ' ttl];
for l = 1:L
    nexttile;
    plot(t_sweep, 10*log10(movmean(d_sweep(:,l).^2, win)+eps), 'Color',[.7 .7 .7]); hold on;
    plot(t_sweep, 10*log10(movmean(e_fxlms(:,l).^2, win)+eps), 'b', 'LineWidth',.8);
    plot(t_sweep, 10*log10(movmean(e_maml(:,l).^2,  win)+eps), 'r', 'LineWidth',.8);
    ylabel('dB'); title(sprintf('Mic l=%d',l)); grid on;
    if l==1; legend('Off','FxLMS','MAML','Location','best'); end
end
xlabel('Time (s)');

figure('Name','Per-bin NR');
b = bar(bin_centers, [NR_fx_bin, NR_ml_bin]);
b(1).FaceColor = [0.3 0.5 0.9]; b(2).FaceColor = [0.9 0.3 0.3];
xlabel('Frequency (Hz)'); ylabel('NR (dB)');
title(['NR per ' num2str(bin_width) '-Hz bin ' ttl]);
legend('FxLMS (cold start)','MAML (residual LMS)'); grid on;

figure('Name','Improvement');
imp = NR_fx_bin - NR_ml_bin; imp(isnan(imp)) = 0;
bar(bin_centers, imp, 'FaceColor',[.2 .7 .3]);
xlabel('Frequency (Hz)'); ylabel('\DeltaNR (dB)');
title(['MAML Advantage ' ttl]);
yline(0,'k--'); grid on;

figure('Name','Weights — speaker m=1');
subplot(2,1,1);
plot(t_sweep, w_traj_fx(:,1),'b', t_sweep, w_traj_ml(:,1),'r','LineWidth',.8);
ylabel('w_{sin}'); title(['w_{sin} (m=1) ' ttl]);
legend('FxLMS','MAML residual'); grid on;
subplot(2,1,2);
plot(t_sweep, w_traj_fx(:,2),'b', t_sweep, w_traj_ml(:,2),'r','LineWidth',.8);
xlabel('Time (s)'); ylabel('w_{cos}'); title('w_{cos}  (m=1)');
legend('FxLMS','MAML residual'); grid on;

figure('Name','Residual delta-w (m=1)');
subplot(2,1,1);
plot(t_sweep, dw_traj_ml(:,1),'Color',[0.8 0.3 0.1],'LineWidth',.8);
yline(0,'k--'); ylabel('\deltaw_{sin}');
title(['\deltaw_{sin} = w - w_{interp}  (m=1) ' ttl]); grid on;
subplot(2,1,2);
plot(t_sweep, dw_traj_ml(:,2),'Color',[0.1 0.6 0.3],'LineWidth',.8);
yline(0,'k--'); ylabel('\deltaw_{cos}');
xlabel('Time (s)'); title('\deltaw_{cos}  (m=1)'); grid on;

% Interpolated weight table — one subplot per speaker
f_fine = linspace(freq_list(1), freq_list(end), 500)';
figure('Name','Interpolated weight table (sum_l Phi per speaker)');
for m = 1:M
    w_sin_fine = arrayfun(interp_w_sin{m}, f_fine);
    w_cos_fine = arrayfun(interp_w_cos{m}, f_fine);
    subplot(M, 2, 2*m-1);
    plot(f_fine, w_sin_fine, 'r-', 'LineWidth',1.2); hold on;
    plot(freq_list, w_sin_init(:,m), 'ro','MarkerSize',5,'MarkerFaceColor','r');
    ylabel('w_{sin}'); title(sprintf('m=%d  w_{sin} interp vs sum_l\\Phi',m)); grid on;
    subplot(M, 2, 2*m);
    plot(f_fine, w_cos_fine, 'b-', 'LineWidth',1.2); hold on;
    plot(freq_list, w_cos_init(:,m), 'bo','MarkerSize',5,'MarkerFaceColor','b');
    ylabel('w_{cos}'); title(sprintf('m=%d  w_{cos} interp vs sum_l\\Phi',m)); grid on;
end
xlabel('Frequency (Hz)');

fprintf('\n=== Done ===\n');

%% =====================  Local function  =================================
function wb = get_w_base(f, interp_w_sin, interp_w_cos, M)
    % Returns w_base [2, 1, M] at frequency f
    wb = zeros(2, 1, M);
    for m = 1:M
        wb(1,1,m) = interp_w_sin{m}(f);
        wb(2,1,m) = interp_w_cos{m}(f);
    end
end