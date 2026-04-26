%% MAML-Notch: Single-channel sweep validation — residual LMS with delays
%
%  M=1 loudspeaker, L=1 error mic.
%
%  Residual LMS:
%    w_base = interp(f_ctrl)   ← interpolated optimum, updated every step
%    delta_w                   ← residual accumulated by LMS
%    Each step:
%      delta_w = w_prev - w_base_prev
%      w       = w_base_now + delta_w   (rebase)
%      iterate() runs normally           (LMS updates w)
%
%  Delays (read from mat file, override below if needed):
%    d_ref — reference signal delay (samples): ctrl.freqs = freq(n - d_ref)
%    d_err — error signal delay     (samples): iterate(e(n - d_err))
%
%  uBuffer pre-warm:
%    plat_ml.uBuffer is pre-filled with the control outputs that w_init
%    would have produced at t < 0, eliminating the initial noise spike
%    caused by the mismatch between the pre-warmed sinBuffer/cosBuffer
%    and the cold uBuffer.
%
%  No primary path.
%  Files needed: NotchController.m, SimulationPlatform.m
%               + MAML_weight_table_manual.mat
%--------------------------------------------------------------------------
close all; clear; clc;
addpath("class\");

%% =====================  Load  ===========================================
load('MAML_weight_table_manual.mat');
fprintf('Loaded: %d freqs (%.1f - %.1f Hz)\n', ...
        length(freq_list), freq_list(1), freq_list(end));

M = 1;  L = 1;
if exist('s_3d','var') && isequal(size(s_3d),[J,1,1])
    % already correct
elseif exist('s_coef','var')
    s_3d = reshape(s_coef(:), [J, 1, 1]);
else
    error('No secondary path found in mat file.');
end
N_freq = length(freq_list);

% ── Delay config (from mat; override here if needed) ─────────────────────
if ~exist('d_ref','var'); d_ref = 0; end
if ~exist('d_err','var'); d_err = 0; end
% d_ref = 3;
% d_err = 2;
fprintf('Delays: d_ref=%d samp (%.2f ms)   d_err=%d samp (%.2f ms)\n', ...
        d_ref, d_ref/fs*1000, d_err, d_err/fs*1000);

if d_err > 0 && freq_list(end) > fs/(4*d_err)
    warning('d_err=%d: stability limit %.1f Hz < f_end %.1f Hz', ...
            d_err, fs/(4*d_err), freq_list(end));
end

%% =====================  Build interpolation functions  ==================
w_sin_table = weight_table(:, 2);
w_cos_table = weight_table(:, 3);

interp_w_sin = @(f) interp1(freq_list, w_sin_table, f, 'linear', 'extrap');
interp_w_cos = @(f) interp1(freq_list, w_cos_table, f, 'linear', 'extrap');
interp_alpha = @(f) interp1(freq_list, alpha_table,  f, 'linear', 'extrap');

fprintf('Interpolation ready: %.1f - %.1f Hz, step %.1f Hz\n', ...
        freq_list(1), freq_list(end), mean(diff(freq_list)));

%% =====================  Sweep config  ===================================
f_start    = 50;
f_end      = freq_list(end);
sweep_rate = 10;
bin_width  = 4;

T_sweep   = (f_end - f_start) / sweep_rate;
N_sweep   = round(T_sweep * fs);
t_sweep   = (0:N_sweep-1)' / fs;
freq_inst = min(f_start + sweep_rate * t_sweep, f_end);

fprintf('Sweep: %.1f -> %.1f Hz, %.1f Hz/s, %.2f s\n', ...
        f_start, f_end, sweep_rate, T_sweep);

%% =====================  Generate disturbance  ===========================
k_sweep     = sweep_rate;
phase_chirp = 2*pi * (f_start * t_sweep + 0.5 * k_sweep * t_sweep.^2);
d_sweep     = sin(phase_chirp);   % [N_sweep x 1]

%% =====================  Create controllers  =============================
f_init  = f_start;
alpha_0 = 0.1*interp_alpha(f_init);

% --- Baseline: pure FxLMS cold start ------------------------------------
ctrl_fx = NotchController(fs, f_init, alpha_0, M, L, J);
ctrl_fx.s = s_3d;

% --- MAML residual LMS --------------------------------------------------
ctrl_ml = NotchController(fs, f_init, alpha_0, M, L, J);
ctrl_ml.s = s_3d;

% Initial w (delta_w = 0 at t=0)
ctrl_ml.w(1,1,1) = interp_w_sin(f_init);
ctrl_ml.w(2,1,1) = interp_w_cos(f_init);

% --- Random-table residual LMS ------------------------------------------
ctrl_rand = NotchController(fs, f_init, alpha_0, M, L, J);
ctrl_rand.s = s_3d;
rng(0,'twister');
rand_weight_table = randn(N_freq, 2);
maml_weight_norm = vecnorm([w_sin_table, w_cos_table], 2, 2);
rand_weight_norm = vecnorm(rand_weight_table, 2, 2);
valid_rand = rand_weight_norm > 0;
rand_weight_table(valid_rand,:) = rand_weight_table(valid_rand,:) ./ rand_weight_norm(valid_rand) .* maml_weight_norm(valid_rand);
rand_w_sin_table = rand_weight_table(:,1);
rand_w_cos_table = rand_weight_table(:,2);
interp_rand_w_sin = @(f) interp1(freq_list, rand_w_sin_table, f, 'linear', 'extrap');
interp_rand_w_cos = @(f) interp1(freq_list, rand_w_cos_table, f, 'linear', 'extrap');
w_rand_init = [interp_rand_w_sin(f_init); interp_rand_w_cos(f_init)];
if norm(w_rand_init) == 0
    w_rand_init = randn(2,1);
end
ctrl_rand.w(:,1,1) = w_rand_init;

% Pre-warm sinBuffer [1 x J] and cosBuffer [1 x J]
j_vec                  = (1:J)';
theta_pre              = -j_vec * 2*pi * f_init / fs;
ctrl_ml.sinBuffer(1,:) = sin(theta_pre)';
ctrl_ml.cosBuffer(1,:) = cos(theta_pre)';
ctrl_ml.theta(1)       = 0;
ctrl_rand.sinBuffer(1,:) = sin(theta_pre)';
ctrl_rand.cosBuffer(1,:) = cos(theta_pre)';
ctrl_rand.theta(1)       = 0;

% w_base_prev: baseline used in the previous step
w_base_prev = [interp_w_sin(f_init); interp_w_cos(f_init)];
w_rand_base_prev = [interp_rand_w_sin(f_init); interp_rand_w_cos(f_init)];

fprintf('\nMAML init at f_start = %.1f Hz:\n', f_start);
fprintf('  w_sin = %+.6f\n', ctrl_ml.w(1,1,1));
fprintf('  w_cos = %+.6f\n', ctrl_ml.w(2,1,1));
fprintf('  delta_w = 0  (starts at interpolated optimum)\n');

plat_fx = SimulationPlatform(d_sweep, s_3d, M, L);
plat_rand = SimulationPlatform(d_sweep, s_3d, M, L);
plat_ml = SimulationPlatform(d_sweep, s_3d, M, L);

% ── Pre-warm plat_ml.uBuffer ─────────────────────────────────────────────
% ctrl_ml has non-zero w_init and pre-warmed sinBuffer, so it produces a
% non-zero u at step n=1.  plat_ml.uBuffer is cold (all zeros), causing
% y = S*uBuffer to be incorrect for the first J steps → initial error spike.
% Fix: fill uBuffer with the u values w_init would have produced at t < 0.
% uBuffer [M x J]: col 1 = most recent u (t=-1/fs), col J = oldest (t=-J/fs)
s_coef_vec = s_3d(:,1,1);   % [J x 1]
dth_init   = 2*pi * f_init / fs;
for k = 1:J
    theta_k = -(k-1) * dth_init;
    u_pre_k = ctrl_ml.w(1,1,1) * sin(theta_k) ...
            + ctrl_ml.w(2,1,1) * cos(theta_k);
    plat_ml.uBuffer(1, k) = u_pre_k;
    u_pre_rand_k = ctrl_rand.w(1,1,1) * sin(theta_k) ...
                 + ctrl_rand.w(2,1,1) * cos(theta_k);
    plat_rand.uBuffer(1, k) = u_pre_rand_k;
end
plat_ml.ufilter2y();   % recompute y with pre-warmed uBuffer
plat_rand.ufilter2y(); % recompute y with pre-warmed uBuffer
fprintf('  plat_ml.uBuffer pre-warmed (%d taps)\n', J);

%% =====================  Delay FIFOs  ====================================
freq_buf = f_init * ones(d_ref + 1, 1);   % freq FIFO
e_buf_fx = zeros(1, d_err + 1);           % error FIFO for baseline
e_buf_rand = zeros(1, d_err + 1);         % error FIFO for random initialization
e_buf_ml = zeros(1, d_err + 1);           % error FIFO for MAML

%% =====================  Run simulation  =================================
fprintf('\nRunning ...\n');

e_fxlms    = zeros(N_sweep, 1);
e_random   = zeros(N_sweep, 1);
e_maml     = zeros(N_sweep, 1);
w_traj_fx  = zeros(N_sweep, 2);
w_traj_rand = zeros(N_sweep, 2);
w_traj_ml  = zeros(N_sweep, 2);
dw_traj_ml = zeros(N_sweep, 2);
dw_traj_rand = zeros(N_sweep, 2);

tic;
for n = 1:N_sweep

    % ── Reference frequency FIFO ─────────────────────────────────────────
    freq_buf = [freq_inst(n); freq_buf(1:end-1)];
    f_ctrl   = freq_buf(end);   % delayed frequency seen by controller

    alpha_now = 0.1 * interp_alpha(f_ctrl);

    ctrl_fx.freqs = f_ctrl;    ctrl_fx.alpha = alpha_now;
    ctrl_rand.freqs = f_ctrl;  ctrl_rand.alpha = alpha_now;
    ctrl_ml.freqs = f_ctrl;    ctrl_ml.alpha = alpha_now;

    % ---- Baseline: pure FxLMS (cold start) ------------------------------
    e_delayed_fx = e_buf_fx(end);         % scalar, d_err steps old
    u_fx = ctrl_fx.iterate(e_delayed_fx);
    plat_fx.sim(u_fx);
    e_true_fx = plat_fx.e;               % scalar
    e_buf_fx  = [e_true_fx, e_buf_fx(1:end-1)];

    e_fxlms(n)     = e_true_fx;
    w_traj_fx(n,:) = ctrl_fx.w(:,1,1)';

    % ---- Random-table residual LMS --------------------------------------
    w_rand_base_now = [interp_rand_w_sin(f_ctrl); interp_rand_w_cos(f_ctrl)];
    dw_rand_prev = ctrl_rand.w(:,1,1) - w_rand_base_prev;
    ctrl_rand.w(:,1,1) = w_rand_base_now + dw_rand_prev;

    e_delayed_rand = e_buf_rand(end);
    u_rand = ctrl_rand.iterate(e_delayed_rand);
    plat_rand.sim(u_rand);
    e_true_rand = plat_rand.e;
    e_buf_rand  = [e_true_rand, e_buf_rand(1:end-1)];

    e_random(n)      = e_true_rand;
    w_traj_rand(n,:) = ctrl_rand.w(:,1,1)';
    dw_traj_rand(n,:) = (ctrl_rand.w(:,1,1) - w_rand_base_now)';
    w_rand_base_prev = w_rand_base_now;

    % ---- MAML: residual LMS ---------------------------------------------
    % Step 1: new interpolated baseline at delayed frequency
    w_base_now = [interp_w_sin(f_ctrl); interp_w_cos(f_ctrl)];

    % Step 2: carry residual delta_w from previous step
    dw_prev          = ctrl_ml.w(:,1,1) - w_base_prev;
    ctrl_ml.w(:,1,1) = w_base_now + dw_prev;

    % Step 3: LMS iterate with delayed error
    e_delayed_ml = e_buf_ml(end);
    u_ml = ctrl_ml.iterate(e_delayed_ml);
    plat_ml.sim(u_ml);
    e_true_ml = plat_ml.e;
    e_buf_ml  = [e_true_ml, e_buf_ml(1:end-1)];

    e_maml(n)       = e_true_ml;
    w_traj_ml(n,:)  = ctrl_ml.w(:,1,1)';
    dw_traj_ml(n,:) = (ctrl_ml.w(:,1,1) - w_base_now)';

    % Step 4: store baseline for next step
    w_base_prev = w_base_now;
end
fprintf('Done. %.1f s\n', toc);

%% =====================  Per-bin NR  =====================================
bin_edges   = (f_start : bin_width : f_end + bin_width)';
N_bins      = length(bin_edges) - 1;
bin_centers = bin_edges(1:end-1) + bin_width/2;
bin_idx     = min(floor((freq_inst - f_start) / bin_width) + 1, N_bins);

NR_fx_bin = nan(N_bins, 1);
NR_rand_bin = nan(N_bins, 1);
NR_ml_bin = nan(N_bins, 1);
for bi = 1:N_bins
    mask = (bin_idx == bi);
    if sum(mask) < 10; continue; end
    dp = mean(d_sweep(mask).^2);
    NR_fx_bin(bi) = 10*log10(mean(e_fxlms(mask).^2) / dp);
    NR_rand_bin(bi) = 10*log10(mean(e_random(mask).^2) / dp);
    NR_ml_bin(bi) = 10*log10(mean(e_maml(mask).^2)  / dp);
end

%% =====================  Print  ==========================================
ttl = sprintf('(d_{ref}=%d, d_{err}=%d)', d_ref, d_err);
fprintf('\n===== Per-bin NR %s =====\n', ttl);
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

%% =====================  Plots  ==========================================
win  = round(0.02 * fs);
win2 = round(0.05 * fs);

figure('Name','Error Power (time)');
plot(t_sweep, 10*log10(movmean(d_sweep.^2,  win)+eps), 'Color',[.7 .7 .7], 'LineStyle','-'); hold on;
plot(t_sweep, 10*log10(movmean(e_fxlms.^2,  win)+eps), 'b', 'LineWidth',1, 'LineStyle','-');
plot(t_sweep, 10*log10(movmean(e_random.^2, win)+eps), 'Color',[0.95 0.75 0.05], 'LineWidth',1, 'LineStyle','-');
plot(t_sweep, 10*log10(movmean(e_maml.^2,   win)+eps), 'r', 'LineWidth',1, 'LineStyle','-');
xlabel('Time (s)'); ylabel('dB');
title(['Error Power vs Time ' ttl]);
legend('Off','FxLMS (cold start)','Random residual LMS','MAML (residual LMS)'); grid on;
format_current_figure();

figure('Name','Noise Reduction vs Time');
p_off_time = movmean(d_sweep.^2, win) + eps;
nr_fx_time = 10*log10(p_off_time ./ (movmean(e_fxlms.^2, win) + eps));
nr_rand_time = 10*log10(p_off_time ./ (movmean(e_random.^2, win) + eps));
nr_ml_time = 10*log10(p_off_time ./ (movmean(e_maml.^2, win) + eps));
yyaxis left;
hold on;
plot(t_sweep, nr_rand_time, 'Color',[0.95 0.75 0.05], 'LineWidth',1, 'LineStyle','-');
plot(t_sweep, nr_fx_time, 'b', 'LineWidth',1, 'LineStyle','-');
plot(t_sweep, nr_ml_time, 'r', 'LineWidth',1, 'LineStyle','-');
ylabel('Noise reduction (dB)');
yyaxis right;
plot(t_sweep, freq_inst, 'k--', 'LineWidth',0.8);
ylabel('Sweep frequency (Hz)');
xlabel('Time (s)');
title('Simulated Engine Noise Cancellation');
legend('Random initialization','Zero initialization','MAML initialization','Sweep frequency', 'Location','northwest');
grid on;
format_current_figure();

figure('Name','Error Power (freq)');
plot(freq_inst, 10*log10(movmean(d_sweep.^2, win2)+eps), 'Color',[.7 .7 .7], 'LineStyle','-'); hold on;
plot(freq_inst, 10*log10(movmean(e_fxlms.^2, win2)+eps), 'b', 'LineWidth',.8, 'LineStyle','-');
plot(freq_inst, 10*log10(movmean(e_random.^2, win2)+eps), 'Color',[0.95 0.75 0.05], 'LineWidth',.8, 'LineStyle','-');
plot(freq_inst, 10*log10(movmean(e_maml.^2,  win2)+eps), 'r', 'LineWidth',.8, 'LineStyle','-');
xlabel('Frequency (Hz)'); ylabel('dB');
title(['Error Power vs Frequency ' ttl]);
legend('Off','FxLMS (cold start)','Random residual LMS','MAML (residual LMS)'); grid on;
format_current_figure();

figure('Name','Per-bin NR');
b = bar(bin_centers, [NR_fx_bin, NR_rand_bin, NR_ml_bin]);
b(1).FaceColor=[0.3 0.5 0.9]; b(2).FaceColor=[0.95 0.75 0.05]; b(3).FaceColor=[0.9 0.3 0.3];
xlabel('Frequency (Hz)'); ylabel('NR (dB)');
title(['NR per ' num2str(bin_width) '-Hz bin ' ttl]);
legend('FxLMS (cold start)','Random residual LMS','MAML (residual LMS)'); grid on;
format_current_figure();

figure('Name','Improvement');
imp = NR_fx_bin - NR_ml_bin; imp(isnan(imp)) = 0;
bar(bin_centers, imp, 'FaceColor',[.2 .7 .3]);
xlabel('Frequency (Hz)'); ylabel('\DeltaNR (dB)');
title(['MAML Advantage ' ttl]);
yline(0,'k-'); grid on;
format_current_figure();

figure('Name','Weights');
subplot(2,1,1);
plot(t_sweep, w_traj_fx(:,1),'b', t_sweep, w_traj_ml(:,1),'r','LineWidth',.8);
ylabel('w_{sin}'); title(['w_{sin} ' ttl]);
legend('FxLMS','MAML residual'); grid on;
subplot(2,1,2);
plot(t_sweep, w_traj_fx(:,2),'b', t_sweep, w_traj_ml(:,2),'r','LineWidth',.8);
xlabel('Time (s)'); ylabel('w_{cos}'); title('w_{cos}');
legend('FxLMS','MAML residual'); grid on;
format_current_figure();

figure('Name','Residual delta-w');
subplot(2,1,1);
plot(t_sweep, dw_traj_ml(:,1),'Color',[0.8 0.3 0.1],'LineWidth',.8);
yline(0,'k-'); ylabel('\deltaw_{sin}');
title(['\deltaw = w - w_{interp} ' ttl]); grid on;
subplot(2,1,2);
plot(t_sweep, dw_traj_ml(:,2),'Color',[0.1 0.6 0.3],'LineWidth',.8);
yline(0,'k-'); ylabel('\deltaw_{cos}');
xlabel('Time (s)'); title('\deltaw_{cos}'); grid on;
format_current_figure();

figure('Name','Interpolated weight table');
f_fine = linspace(freq_list(1), freq_list(end), 500)';
subplot(3,1,1);
plot(f_fine, interp_w_sin(f_fine),'r-','LineWidth',1.2); hold on;
plot(freq_list, w_sin_table,'ro','MarkerSize',5,'MarkerFaceColor','r');
ylabel('w_{sin}'); title('w_{sin}: interp (line) vs table (dots)'); grid on;
subplot(3,1,2);
plot(f_fine, interp_w_cos(f_fine),'b-','LineWidth',1.2); hold on;
plot(freq_list, w_cos_table,'bo','MarkerSize',5,'MarkerFaceColor','b');
ylabel('w_{cos}'); title('w_{cos}: interp (line) vs table (dots)'); grid on;
subplot(3,1,3);
plot(f_fine, arrayfun(interp_alpha, f_fine),'k-','LineWidth',1.2); hold on;
plot(freq_list, alpha_table,'ko','MarkerSize',5,'MarkerFaceColor','k');
xlabel('Frequency (Hz)'); ylabel('\alpha');
title('\alpha: interp (line) vs table (dots)'); grid on;
format_current_figure();

fprintf('\n=== Done ===\n');

function format_current_figure()
    set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 15);
    set(findall(gcf, 'Type', 'legend'), 'FontSize', 15);
    set(findall(gcf, 'Type', 'text'), 'FontSize', 15);
    set(findall(gcf, 'Type', 'text', '-regexp', 'Tag', '.*Title.*'), 'FontSize', 18);
    ax = findall(gcf, 'Type', 'axes');
    for ii = 1:numel(ax)
        ax(ii).Title.FontSize = 18;
        ax(ii).Title.FontName = 'Times New Roman';
        ax(ii).XLabel.FontSize = 15;
        ax(ii).XLabel.FontName = 'Times New Roman';
        ax(ii).YLabel.FontSize = 15;
        ax(ii).YLabel.FontName = 'Times New Roman';
        ax(ii).FontSize = 15;
        ax(ii).FontName = 'Times New Roman';
    end
end
