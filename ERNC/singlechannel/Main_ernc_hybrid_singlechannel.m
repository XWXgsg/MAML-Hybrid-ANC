%% ERNC hybrid single-channel validation
%
% This script integrates the already-validated single-channel RNC and ENC
% models into one hybrid single-channel ERNC simulation.
%
% Sign convention:
%   RNC branch code uses     e = d - S*u_rnc,  w <- w + mu*e*r.
%   ENC branch code uses     e = d + S*u_enc,  w <- w - alpha*e*r.
%
% To reuse both controllers without changing their internal update laws, the
% hybrid physical error is implemented as
%
%   e(n) = d_road(n) + d_engine(n) - S*u_rnc(n) + S*u_enc(n).
%
% Equivalently, the physical loudspeaker drive would be u_phys = u_rnc -
% u_enc if a single secondary-path sign convention e = d - S*u_phys is used.
%
% Compared cases:
%   1) ANC off
%   2) RNC only: broadband MAML-FxLMS initialized by Wc
%   3) ENC only: residual MAML-notch initialized by frequency table
%   4) Zero-init ERNC: zero-initialized RNC + zero-initialized ENC
%   5) MAML-init ERNC: RNC + ENC driven by the same residual error
%--------------------------------------------------------------------------
close all; clear; clc;

%% ===================== Paths ===========================================
this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(fileparts(this_dir));
rnc_dir  = fullfile(root_dir, 'Rnc', 'singlechannel');
enc_dir  = fullfile(root_dir, 'ENC', 'singlechannel');

addpath(fullfile(rnc_dir, 'class'));
addpath(fullfile(enc_dir, 'class'));

%% ===================== Load assets =====================================
enc = load(fullfile(enc_dir, 'MAML_weight_table_manual.mat'));
rnc = load(fullfile(rnc_dir, 'Weigth_initiate_Nstep_forget.mat'));
cclq = load(fullfile(rnc_dir, 'data', 'CCLQ_60_1.mat'), 'Signal_0');
pri = load(fullfile(rnc_dir, 'path', 'PrimaryPath_256order.mat'));
sec = load(fullfile(rnc_dir, 'path', 'SecondaryPath_2x2_256order.mat'));

fs = enc.fs;
J  = enc.J;
if fs ~= 16000
    error('Expected fs=16000 Hz from the existing single-channel models.');
end
if length(sec.S11(:)) ~= J
    error('Secondary path S11 length (%d) does not match ENC J (%d).', ...
        length(sec.S11(:)), J);
end

s_vec = sec.S11(:);                 % [J x 1]
s_3d  = reshape(s_vec, [J, 1, 1]);  % for Controller / NotchController
pri_path = conv(pri.P1(:), sec.S11(:));

Wc_rnc = rnc.Wc(:);
if length(Wc_rnc) ~= Controller.I
    error('RNC Wc length (%d) must match Controller.I (%d).', ...
        length(Wc_rnc), Controller.I);
end

%% ===================== User configuration ==============================
% Engine sweep. If the raw road-noise record is shorter than the full sweep,
% simulate only the available single segment; do not concatenate/repeat data.
f_start    = 50;
f_end      = enc.freq_list(end);
sweep_rate = 10;                  % Hz/s
bin_width  = 4;                   % Hz

if f_start < enc.freq_list(1) || f_start > enc.freq_list(end)
    error('f_start=%.1f Hz must lie inside the ENC weight table range %.1f-%.1f Hz.', ...
        f_start, enc.freq_list(1), enc.freq_list(end));
end

road_gain   = 1.0;
engine_gain = 0.5;

mu_rnc_base = 1e-6;               % step size used by RNC single-channel test
alpha_scale_enc = 0.05;            % same scale as ENC single-channel sweep

%% ===================== Generate hybrid disturbance =====================
T_sweep_full = (f_end - f_start) / sweep_rate;
N_sweep_full = round(T_sweep_full * fs);
road_mic = cclq.Signal_0.y_values.values;
fs_road_raw = 25600;
road_ref_all = resample(road_mic(:,2), fs, fs_road_raw);
N_road = numel(road_ref_all);
N_sim = min(N_sweep_full, N_road);
T_sim = N_sim / fs;
t     = (0:N_sim-1)' / fs;

road_ref_raw = road_ref_all(1:N_sim);
road_dist_raw = filter(pri_path, 1, road_ref_raw);

% Scale reference and disturbance together so the primary-path relation is
% preserved while the road component has a convenient RMS level.
road_scale = 1 / max(sqrt(mean(road_dist_raw.^2)), eps);
x_road = road_ref_raw * road_scale;
d_road = road_dist_raw * road_scale;
% Scaling both reference and disturbance by road_scale multiplies the LMS
% gradient e*r by road_scale^2. Compensate the step size so zero-init RNC
% has the same effective convergence rate as Main_tst_function.m.
mu_rnc = mu_rnc_base / (road_scale^2);

freq_inst = min(f_start + sweep_rate * t, f_end);
f_stop_actual = freq_inst(end);
phase_engine = 2*pi * (f_start * t + 0.5 * sweep_rate * t.^2);
d_engine = sin(phase_engine);
d_engine = d_engine / max(sqrt(mean(d_engine.^2)), eps);

d_road   = road_gain   * d_road;
d_engine = engine_gain * d_engine;
d_total  = d_road + d_engine;

if engine_gain == 0
    warning(['engine_gain is zero. ENC/ERNC cases still run the notch ', ...
        'controller, so they are not a valid hybrid-noise comparison; ', ...
        'use the RNC-only curve for road-noise-only validation.']);
end

fprintf('ERNC hybrid single-channel simulation\n');
fprintf('  fs=%d Hz, J=%d, duration=%.2f s\n', fs, J, T_sim);
fprintf('  engine sweep: %.1f -> %.1f Hz at %.1f Hz/s (requested end %.1f Hz)\n', ...
    f_start, f_stop_actual, sweep_rate, f_end);
fprintf('  road segment: first %d samples from %d CCLQ samples after %d->%d Hz resampling; no repetition\n', ...
    N_sim, N_road, fs_road_raw, fs);
fprintf('  RMS road=%.3f, engine=%.3f, total=%.3f\n', ...
    sqrt(mean(d_road.^2)), sqrt(mean(d_engine.^2)), sqrt(mean(d_total.^2)));
fprintf('  RNC mu: base %.3g, scaled %.3g (road_scale=%.3f)\n', ...
    mu_rnc_base, mu_rnc, road_scale);
df = diff(freq_inst);
fprintf('  frequency increment per sample: min %.6g Hz, max %.6g Hz\n', ...
    min(df), max(df));

%% ===================== Run compared cases ==============================
common = struct();
common.fs = fs;
common.J = J;
common.s_vec = s_vec;
common.s_3d = s_3d;
common.x_road = x_road;
common.d_total = d_total;
common.freq_inst = freq_inst;
common.f_stop_actual = f_stop_actual;
common.freq_list = enc.freq_list;
common.weight_table = enc.weight_table;
common.alpha_table = enc.alpha_table;
common.d_ref = getfield_default(enc, 'd_ref', 0);
common.d_err = getfield_default(enc, 'd_err', 0);
common.Wc_rnc = Wc_rnc;
common.mu_rnc = mu_rnc;
common.alpha_scale_enc = alpha_scale_enc;
common.Wc_rnc_random = build_random_rnc_init(Wc_rnc, 0);
common.rand_weight_table = build_random_enc_table(enc.weight_table, 0);

out_off = make_off_case(d_total);
out_zero_rnc = run_hybrid_case(common, true, false, 'Zero-init RNC only', 'zero', 'off');
out_rnc = run_hybrid_case(common, true,  false, 'MAML-init RNC only', 'maml', 'off');
out_enc = run_hybrid_case(common, false, true,  'ENC only',       'off',  'maml_residual');
out_zero_ernc = run_hybrid_case(common, true, true, 'Zero-init ERNC', 'zero', 'zero');
out_rand_ernc = run_hybrid_case(common, true, true, 'Random-init ERNC', 'random', 'random_residual');
out_ernc = run_hybrid_case(common, true, true,  'MAML-init ERNC', 'maml', 'maml_residual');

%% ===================== Metrics =========================================
p_total = mean(d_total.^2);
p_road  = mean(d_road.^2);
p_eng   = mean(d_engine.^2);

summary = [
    metric_row("Off",         out_off.e,  p_total)
    metric_row("Zero RNC only", out_zero_rnc.e, p_total)
    metric_row("MAML RNC only", out_rnc.e, p_total)
    metric_row("ENC only",    out_enc.e,  p_total)
    metric_row("Zero-init ERNC", out_zero_ernc.e, p_total)
    metric_row("Random-init ERNC", out_rand_ernc.e, p_total)
    metric_row("MAML-init ERNC", out_ernc.e, p_total)
];

fprintf('\n===== Full-band error/input ratio =====\n');
fprintf('%-14s | %10s | %10s\n', 'Case', 'Err/Input', 'Improve');
fprintf('%s\n', repmat('-', 1, 43));
off_db = summary(1).err_db;
for i = 1:numel(summary)
    fprintf('%-14s | %+9.2f dB | %+8.2f dB\n', ...
        summary(i).name, summary(i).err_db, off_db - summary(i).err_db);
end

%% ===================== Segment metrics =================================
segment_edges = unique([0, 1, 2, 3, 5, 7, T_sim]);
fprintf('\n===== Segment error/input ratio =====\n');
fprintf('%11s | %8s %8s %8s %8s %10s %10s %10s\n', ...
    'Time(s)', 'Off', 'ZeroRNC', 'MAMLRNC', 'ENC', 'ZeroERNC', 'RandERNC', 'MAMLERNC');
fprintf('%s\n', repmat('-', 1, 89));
for si = 1:(numel(segment_edges)-1)
    t0 = segment_edges(si);
    t1 = segment_edges(si+1);
    mask = (t >= t0) & (t < t1);
    if ~any(mask)
        continue;
    end
    p_seg = mean(d_total(mask).^2);
    fprintf('%4.1f-%4.1f | %+7.2f %+7.2f %+7.2f %+7.2f %+9.2f %+9.2f %+9.2f dB\n', ...
        t0, t1, ...
        10*log10(mean(d_total(mask).^2) / p_seg), ...
        10*log10(mean(out_zero_rnc.e(mask).^2) / p_seg), ...
        10*log10(mean(out_rnc.e(mask).^2) / p_seg), ...
        10*log10(mean(out_enc.e(mask).^2) / p_seg), ...
        10*log10(mean(out_zero_ernc.e(mask).^2) / p_seg), ...
        10*log10(mean(out_rand_ernc.e(mask).^2) / p_seg), ...
        10*log10(mean(out_ernc.e(mask).^2) / p_seg));
end

%% ===================== Component metrics ===============================
component_cases = {
    'Zero RNC only',  out_zero_rnc
    'MAML RNC only',  out_rnc
    'Zero-init ERNC', out_zero_ernc
    'Random-init ERNC', out_rand_ernc
    'MAML-init ERNC', out_ernc
};
fprintf('\n===== Component residual ratio =====\n');
fprintf('%-14s | %10s | %10s | %10s\n', ...
    'Case', 'Road part', 'Engine part', 'Total check');
fprintf('%s\n', repmat('-', 1, 57));
for ci = 1:size(component_cases, 1)
    cname = component_cases{ci, 1};
    cout = component_cases{ci, 2};
    road_res = d_road(:) - cout.y_rnc(:);
    engine_res = d_engine(:) + cout.y_enc(:);
    total_recon = road_res + engine_res;
    fprintf('%-14s | %+9.2f dB | %+9.2f dB | %+9.2f dB\n', ...
        cname, ...
        10*log10(mean(road_res.^2) / mean(d_road.^2)), ...
        10*log10(mean(engine_res.^2) / mean(d_engine.^2)), ...
        10*log10(mean(total_recon.^2) / mean(d_total.^2)));
end

[bin_centers, nr_zero_rnc, nr_rnc, nr_enc, nr_zero_ernc, nr_rand_ernc, nr_ernc] = per_bin_error_ratio( ...
    freq_inst, d_total, out_zero_rnc.e, out_rnc.e, out_enc.e, out_zero_ernc.e, out_rand_ernc.e, out_ernc.e, ...
    f_start, f_stop_actual, bin_width);
nr_off = zeros(size(nr_rnc));
nr_off(isnan(nr_zero_rnc) & isnan(nr_rnc) & isnan(nr_enc) & isnan(nr_zero_ernc) & isnan(nr_rand_ernc) & isnan(nr_ernc)) = NaN;

fprintf('\n===== Per-engine-frequency bin error/input ratio =====\n');
fprintf('%8s | %9s %9s %9s %9s %9s %9s %9s\n', 'Freq', 'Off', 'ZeroRNC', 'MAMLRNC', 'ENC', 'ZeroERNC', 'RandERNC', 'MAMLERNC');
fprintf('%s\n', repmat('-', 1, 86));
for i = 1:numel(bin_centers)
    if isnan(nr_ernc(i)); continue; end
    fprintf('%6.1fHz | %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f dB\n', ...
        bin_centers(i), nr_off(i), nr_zero_rnc(i), nr_rnc(i), nr_enc(i), nr_zero_ernc(i), nr_rand_ernc(i), nr_ernc(i));
end

%% ===================== Save ============================================
save('ERNC_singlechannel_results.mat', ...
    'out_off', 'out_zero_rnc', 'out_rnc', 'out_enc', 'out_zero_ernc', 'out_rand_ernc', 'out_ernc', ...
    'summary', 'bin_centers', 'nr_off', 'nr_zero_rnc', 'nr_rnc', 'nr_enc', 'nr_zero_ernc', 'nr_rand_ernc', 'nr_ernc', ...
    'd_road', 'd_engine', 'd_total', 'x_road', 'freq_inst', ...
    'fs', 'J', 'f_start', 'f_end', 'f_stop_actual', 'sweep_rate', 'bin_width', ...
    'road_gain', 'engine_gain', 'mu_rnc', 'alpha_scale_enc');
fprintf('\nSaved: ERNC_singlechannel_results.mat\n');

%% ===================== Plots ===========================================
win = round(0.05 * fs);
plot_t_end = min(5, T_sim);
[road_peak_power, idx_road_peak] = max(10*log10(movmean(d_road.^2, win) + eps));
fprintf('  largest road-noise moving-power segment: %.2f dB at t=%.3f s, f=%.3f Hz\n', ...
    road_peak_power, t(idx_road_peak), freq_inst(idx_road_peak));
enc_power_trace = 10*log10(movmean(out_enc.e.^2, win) + eps);
mid_mask = find(t > 0.45*T_sim & t < 0.60*T_sim);
[enc_mid_peak_power, rel_mid] = max(enc_power_trace(mid_mask));
idx_mid_peak = mid_mask(rel_mid);
fprintf('  middle ENC-only moving-power peak: %.2f dB at t=%.3f s, f=%.3f Hz\n', ...
    enc_mid_peak_power, t(idx_mid_peak), freq_inst(idx_mid_peak));

figure('Name', '30-500 Hz noise reduction');
[b_band, a_band] = butter(4, [30 500]/(fs/2), 'bandpass');
d_total_band = filtfilt(b_band, a_band, d_total);
zero_ernc_band = filtfilt(b_band, a_band, out_zero_ernc.e);
rand_ernc_band = filtfilt(b_band, a_band, out_rand_ernc.e);
ernc_band = filtfilt(b_band, a_band, out_ernc.e);
p_off_band = movmean(d_total_band.^2, win) + eps;
nr_zero_ernc_band = 10*log10(p_off_band ./ (movmean(zero_ernc_band.^2, win) + eps));
nr_rand_ernc_band = 10*log10(p_off_band ./ (movmean(rand_ernc_band.^2, win) + eps));
nr_ernc_band = 10*log10(p_off_band ./ (movmean(ernc_band.^2, win) + eps));
hold on;
plot(t, nr_rand_ernc_band, 'Color', [0.2 0.65 0.2], 'LineWidth', 0.65, ...
    'DisplayName', 'Random-init Broadband and Narrowband ANC');
plot(t, nr_zero_ernc_band, 'Color', [0.95 0.75 0.05], 'LineWidth', 0.65, ...
    'DisplayName', 'Zero-init Broadband and Narrowband ANC ');
plot(t, nr_ernc_band, 'r', 'LineWidth', 0.75, ...
    'DisplayName', 'MAML-init Broadband and Narrowband ANC');
xlabel('Time (s)'); ylabel('Noise reduction (dB)');
title('30-500 Hz Noise Reduction');
legend('Location', 'northwest');
xlim([0 plot_t_end]);
grid on;
format_current_figure();

figure('Name', 'Simulated Engine Noise and Road Noise Cancellation');
plot(t, d_total, 'Color', [.15 .15 .15], 'LineWidth', 0.75, ...
    'DisplayName', 'ANC Off'); hold on;
plot(t, out_rand_ernc.e, 'Color', [0.2 0.65 0.2], 'LineWidth', 0.65, ...
    'DisplayName', 'Random-init Broadband and Narrowband ANC');
plot(t, out_zero_ernc.e, 'Color', [0.95 0.75 0.05], 'LineWidth', 0.65, ...
    'DisplayName', 'Zero-init Broadband and Narrowband ANC ');
plot(t, out_ernc.e, 'r', 'LineWidth', 0.75, ...
    'DisplayName', 'MAML-init Broadband and Narrowband ANC');
xlabel('Time (s)');
ylabel('Error microphone signal');
title('Simulated Engine Noise and Road Noise Cancellation');
legend('Location', 'best');
xlim([0 plot_t_end]);
grid on;
format_current_figure();

tf_len = min(round(5*fs), numel(t));
tf_signals = {d_total(1:tf_len), out_rand_ernc.e(1:tf_len), out_zero_ernc.e(1:tf_len), out_ernc.e(1:tf_len)};
tf_titles = {'(a) ANC Off','(b) Random-init Broadband and Narrowband ANC','(c) Zero-init Broadband and Narrowband ANC','(d) MAML-init Broadband and Narrowband ANC'};
tf_window = hamming(1024);
tf_overlap = 768;
tf_nfft = 2048;
tf_power_db = cell(1,4);
tf_time = cell(1,4);
tf_freq = cell(1,4);
tf_clim = [inf, -inf];

for ii = 1:4
    [~, tf_freq{ii}, tf_time{ii}, ps] = spectrogram(tf_signals{ii}, tf_window, tf_overlap, tf_nfft, fs, 'yaxis');
    tf_power_db{ii} = 10*log10(ps + eps);
    freq_mask = tf_freq{ii} <= 500;
    tf_clim(1) = min(tf_clim(1), min(tf_power_db{ii}(freq_mask,:), [], 'all'));
    tf_clim(2) = max(tf_clim(2), max(tf_power_db{ii}(freq_mask,:), [], 'all'));
end

figure('Name', 'First 5 s time-frequency comparison');
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for ii = 1:4
    nexttile(tl);
    imagesc(tf_time{ii}, tf_freq{ii}, tf_power_db{ii});
    axis xy;
    title(tf_titles{ii});
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    ylim([0 500]);
    caxis(tf_clim);
    cb = colorbar;
    set(cb, 'FontName', 'Times New Roman', 'FontSize', 15);
end
format_current_figure();

fprintf('\n=== ERNC hybrid simulation done ===\n');
return;

figure('Name', 'ERNC hybrid time-domain waveforms');
subplot(3,1,1);
plot(t, d_road, 'Color', [.2 .45 .85], 'LineWidth', 0.6); hold on;
plot(t, d_engine, 'Color', [.9 .45 .1], 'LineWidth', 0.6);
plot(t, d_total, 'Color', [.35 .35 .35], 'LineWidth', 0.6);
ylabel('Primary');
title('Primary disturbance components');
legend('Road', 'Engine', 'Road + engine', 'Location', 'best');
xlim([0 plot_t_end]);
grid on;

subplot(3,1,2);
plot(t, d_total, 'Color', [.65 .65 .65], 'LineWidth', 0.6); hold on;
plot(t, out_rnc.e, 'b', 'LineWidth', 0.6);
plot(t, out_zero_ernc.e, 'Color', [0.95 0.75 0.05], 'LineWidth', 0.6);
plot(t, out_ernc.e, 'r', 'LineWidth', 0.7);
ylabel('Residual');
title('Residual error signals');
legend('Off', 'MAML-init RNC only', 'Zero-init ERNC', 'MAML-init ERNC', 'Location', 'best');
xlim([0 plot_t_end]);
grid on;

subplot(3,1,3);
plot(t, freq_inst, 'k', 'LineWidth', 0.8);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Engine frequency trajectory');
xlim([0 plot_t_end]);
grid on;
format_current_figure();

zoom_half = round(0.35 * fs);
idx_zoom = max(1, idx_road_peak - zoom_half) : min(numel(t), idx_road_peak + zoom_half);
figure('Name', 'ERNC hybrid time-domain zoom');
subplot(2,1,1);
plot(t(idx_zoom), d_road(idx_zoom), 'Color', [.2 .45 .85], 'LineWidth', 0.8); hold on;
plot(t(idx_zoom), d_engine(idx_zoom), 'Color', [.9 .45 .1], 'LineWidth', 0.8);
plot(t(idx_zoom), d_total(idx_zoom), 'Color', [.35 .35 .35], 'LineWidth', 0.8);
ylabel('Primary');
title(sprintf('Time-domain zoom around strongest road segment: t=%.3f s, f=%.2f Hz', ...
    t(idx_road_peak), freq_inst(idx_road_peak)));
legend('Road', 'Engine', 'Road + engine', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(t(idx_zoom), d_total(idx_zoom), 'Color', [.65 .65 .65], 'LineWidth', 0.8); hold on;
plot(t(idx_zoom), out_rnc.e(idx_zoom), 'b', 'LineWidth', 0.8);
plot(t(idx_zoom), out_zero_ernc.e(idx_zoom), 'Color', [0.95 0.75 0.05], 'LineWidth', 0.8);
plot(t(idx_zoom), out_ernc.e(idx_zoom), 'r', 'LineWidth', 0.9);
xlabel('Time (s)'); ylabel('Residual');
title('Residual errors in the same interval');
legend('Off', 'MAML-init RNC only', 'Zero-init ERNC', 'MAML-init ERNC', 'Location', 'best');
grid on;
format_current_figure();

mid_zoom_half = round(0.20 * fs);
idx_mid_zoom = max(1, idx_mid_peak - mid_zoom_half) : min(numel(t), idx_mid_peak + mid_zoom_half);
figure('Name', 'ERNC hybrid middle spike zoom');
subplot(3,1,1);
plot(t(idx_mid_zoom), d_road(idx_mid_zoom), 'Color', [.2 .45 .85], 'LineWidth', 0.8); hold on;
plot(t(idx_mid_zoom), d_engine(idx_mid_zoom), 'Color', [.9 .45 .1], 'LineWidth', 0.8);
plot(t(idx_mid_zoom), d_total(idx_mid_zoom), 'Color', [.35 .35 .35], 'LineWidth', 0.8);
ylabel('Primary');
title(sprintf('Middle zoom: t=%.3f s, f=%.2f Hz', t(idx_mid_peak), freq_inst(idx_mid_peak)));
legend('Road', 'Engine', 'Road + engine', 'Location', 'best');
grid on;

subplot(3,1,2);
plot(t(idx_mid_zoom), d_total(idx_mid_zoom), 'Color', [.65 .65 .65], 'LineWidth', 0.8); hold on;
plot(t(idx_mid_zoom), out_rnc.e(idx_mid_zoom), 'b', 'LineWidth', 0.8);
plot(t(idx_mid_zoom), out_zero_ernc.e(idx_mid_zoom), 'Color', [0.95 0.75 0.05], 'LineWidth', 0.8);
plot(t(idx_mid_zoom), out_ernc.e(idx_mid_zoom), 'r', 'LineWidth', 0.9);
ylabel('Residual');
legend('Off', 'MAML-init RNC only', 'Zero-init ERNC', 'MAML-init ERNC', 'Location', 'best');
grid on;

subplot(3,1,3);
plot(t(idx_mid_zoom), freq_inst(idx_mid_zoom), 'k', 'LineWidth', 0.9);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('Frequency is continuous in the spike interval');
grid on;
format_current_figure();

figure('Name', 'ERNC hybrid per-bin error ratio');
b = bar(bin_centers, [nr_off, nr_rnc, nr_zero_ernc, nr_ernc]);
b(1).FaceColor = [.65 .65 .65];
b(2).FaceColor = [0 0 1];
b(3).FaceColor = [0.95 0.75 0.05];
b(4).FaceColor = [1 0 0];
xlabel('Engine frequency (Hz)'); ylabel('Error/Input power (dB)');
title(sprintf('Per-bin error ratio, %.0f-Hz bins', bin_width));
legend('Off', 'MAML-init RNC only', 'Zero-init ERNC', 'MAML-init ERNC', 'Location', 'best');
grid on;
format_current_figure();

road_res_zero_rnc = d_road(:) - out_zero_rnc.y_rnc(:);
road_res_rnc = d_road(:) - out_rnc.y_rnc(:);
road_res_zero = d_road(:) - out_zero_ernc.y_rnc(:);
road_res_maml = d_road(:) - out_ernc.y_rnc(:);
eng_res_zero_rnc = d_engine(:) + out_zero_rnc.y_enc(:);
eng_res_rnc = d_engine(:) + out_rnc.y_enc(:);
eng_res_zero = d_engine(:) + out_zero_ernc.y_enc(:);
eng_res_maml = d_engine(:) + out_ernc.y_enc(:);

figure('Name', 'ERNC component residual powers');
subplot(2,1,1);
plot(t, 10*log10(movmean(d_road.^2, win) + eps), 'Color', [.65 .65 .65]); hold on;
plot(t, 10*log10(movmean(road_res_rnc.^2, win) + eps), 'b', 'LineWidth', 0.8);
plot(t, 10*log10(movmean(road_res_zero.^2, win) + eps), 'Color', [0.95 0.75 0.05], 'LineWidth', 0.8);
plot(t, 10*log10(movmean(road_res_maml.^2, win) + eps), 'r', 'LineWidth', 0.9);
ylabel('Road residual (dB)');
title('Road component residual: d_{road} - y_{RNC}');
legend('Road off', 'MAML-init RNC only', 'Zero-init ERNC', 'MAML-init ERNC', 'Location', 'best');
xlim([0 plot_t_end]);
grid on;
format_current_figure();

subplot(2,1,2);
plot(t, 10*log10(movmean(d_engine.^2, win) + eps), 'Color', [.65 .65 .65]); hold on;
plot(t, 10*log10(movmean(eng_res_rnc.^2, win) + eps), 'b', 'LineWidth', 0.8);
plot(t, 10*log10(movmean(eng_res_zero.^2, win) + eps), 'Color', [0.95 0.75 0.05], 'LineWidth', 0.8);
plot(t, 10*log10(movmean(eng_res_maml.^2, win) + eps), 'r', 'LineWidth', 0.9);
xlabel('Time (s)'); ylabel('Engine residual (dB)');
title('Engine component residual: d_{engine} + y_{ENC}');
legend('Engine off', 'MAML-init RNC only', 'Zero-init ERNC', 'MAML-init ERNC', 'Location', 'best');
xlim([0 plot_t_end]);
grid on;

figure('Name', 'ERNC control outputs');
subplot(3,1,1);
plot(t, out_ernc.u_rnc, 'b', 'LineWidth', 0.8);
ylabel('u_{RNC}'); title('RNC control output'); grid on;
xlim([0 plot_t_end]);
subplot(3,1,2);
plot(t, out_ernc.u_enc, 'r', 'LineWidth', 0.8);
ylabel('u_{ENC}'); title('ENC control output (code sign)'); grid on;
xlim([0 plot_t_end]);
subplot(3,1,3);
plot(t, out_ernc.u_rnc - out_ernc.u_enc, 'k', 'LineWidth', 0.8);
xlabel('Time (s)'); ylabel('u_{phys}');
title('Equivalent physical drive: u_{RNC} - u_{ENC}'); grid on;
xlim([0 plot_t_end]);
format_current_figure();

figure('Name', 'ERNC notch weights');
subplot(3,1,1);
plot(t, out_ernc.w_enc(:,1), 'r', 'LineWidth', 0.8); hold on;
plot(t, out_ernc.w_base(:,1), 'k--', 'LineWidth', 0.8);
ylabel('w_{sin}'); legend('online', 'interp base', 'Location', 'best');
title('ENC residual MAML-notch weights in hybrid ERNC'); grid on;
xlim([0 plot_t_end]);
subplot(3,1,2);
plot(t, out_ernc.w_enc(:,2), 'b', 'LineWidth', 0.8); hold on;
plot(t, out_ernc.w_base(:,2), 'k--', 'LineWidth', 0.8);
ylabel('w_{cos}'); legend('online', 'interp base', 'Location', 'best'); grid on;
xlim([0 plot_t_end]);
subplot(3,1,3);
plot(t, out_ernc.dw_enc(:,1), 'Color', [.8 .3 .1], 'LineWidth', 0.8); hold on;
plot(t, out_ernc.dw_enc(:,2), 'Color', [.1 .6 .3], 'LineWidth', 0.8);
xlabel('Time (s)'); ylabel('\delta w');
legend('\delta w_{sin}', '\delta w_{cos}', 'Location', 'best');
title('Residual correction'); grid on;
xlim([0 plot_t_end]);
format_current_figure();

fprintf('\n=== ERNC hybrid simulation done ===\n');

%% ===================== Local functions =================================
function out = run_hybrid_case(c, use_rnc, use_enc, case_name, rnc_init, enc_init)
    N = length(c.d_total);
    J = c.J;

    interp_w_sin = @(f) interp1(c.freq_list, c.weight_table(:,2), f, 'linear', 'extrap');
    interp_w_cos = @(f) interp1(c.freq_list, c.weight_table(:,3), f, 'linear', 'extrap');
    interp_alpha = @(f) interp1(c.freq_list, c.alpha_table,     f, 'linear', 'extrap');
    interp_rand_w_sin = @(f) interp1(c.freq_list, c.rand_weight_table(:,1), f, 'linear', 'extrap');
    interp_rand_w_cos = @(f) interp1(c.freq_list, c.rand_weight_table(:,2), f, 'linear', 'extrap');

    if use_rnc
        ctrl_rnc = Controller(c.mu_rnc, 1, 1, 1);
        ctrl_rnc.s = c.s_3d;
        if strcmpi(rnc_init, 'maml')
            ctrl_rnc.w = reshape(c.Wc_rnc, ctrl_rnc.I, ctrl_rnc.K, ctrl_rnc.M);
        elseif strcmpi(rnc_init, 'random')
            ctrl_rnc.w = reshape(c.Wc_rnc_random, ctrl_rnc.I, ctrl_rnc.K, ctrl_rnc.M);
        elseif strcmpi(rnc_init, 'zero')
            ctrl_rnc.w = zeros(ctrl_rnc.I, ctrl_rnc.K, ctrl_rnc.M);
        else
            error('Unknown RNC init mode: %s', rnc_init);
        end
    else
        ctrl_rnc = [];
    end

    f_init = c.freq_inst(1);
    if use_enc
        alpha_0 = c.alpha_scale_enc * interp_alpha(f_init);
        ctrl_enc = NotchController(c.fs, f_init, alpha_0, 1, 1, J);
        ctrl_enc.s = c.s_3d;

        if strcmpi(enc_init, 'maml_residual')
            ctrl_enc.w(1,1,1) = interp_w_sin(f_init);
            ctrl_enc.w(2,1,1) = interp_w_cos(f_init);
            w_base_prev = [interp_w_sin(f_init); interp_w_cos(f_init)];
        elseif strcmpi(enc_init, 'random_residual')
            ctrl_enc.w(1,1,1) = interp_rand_w_sin(f_init);
            ctrl_enc.w(2,1,1) = interp_rand_w_cos(f_init);
            w_base_prev = [interp_rand_w_sin(f_init); interp_rand_w_cos(f_init)];
        elseif strcmpi(enc_init, 'zero')
            ctrl_enc.w(:,1,1) = 0;
            w_base_prev = zeros(2,1);
        else
            error('Unknown ENC init mode: %s', enc_init);
        end

        if strcmpi(enc_init, 'maml_residual') || strcmpi(enc_init, 'random_residual')
            theta_pre = -(1:J)' * 2*pi * f_init / c.fs;
            ctrl_enc.sinBuffer(1,:) = sin(theta_pre)';
            ctrl_enc.cosBuffer(1,:) = cos(theta_pre)';
            ctrl_enc.theta(1) = 0;
        end
    else
        ctrl_enc = [];
        w_base_prev = zeros(2,1);
    end

    u_rnc_buf = zeros(1, J);
    u_enc_buf = zeros(1, J);
    if use_enc && (strcmpi(enc_init, 'maml_residual') || strcmpi(enc_init, 'random_residual'))
        u_enc_buf = prewarm_notch_u_buffer(ctrl_enc.w(:,1,1), f_init, c.fs, J);
    end

    e_prev = 0;
    e_buf_enc = zeros(1, c.d_err + 1);
    freq_buf = f_init * ones(c.d_ref + 1, 1);

    out = struct();
    out.name = case_name;
    out.e = zeros(N, 1);
    out.y_rnc = zeros(N, 1);
    out.y_enc = zeros(N, 1);
    out.u_rnc = zeros(N, 1);
    out.u_enc = zeros(N, 1);
    out.w_enc = nan(N, 2);
    out.w_base = nan(N, 2);
    out.dw_enc = nan(N, 2);

    fprintf('\nRunning %s ...\n', case_name);
    tic;
    for n = 1:N
        if use_rnc
            u_rnc = ctrl_rnc.iterate(c.x_road(n), e_prev);
        else
            u_rnc = 0;
        end

        if use_enc
            freq_buf = [c.freq_inst(n); freq_buf(1:end-1)];
            f_ctrl = freq_buf(end);
            alpha_now = c.alpha_scale_enc * interp_alpha(f_ctrl);

            ctrl_enc.freqs = f_ctrl;
            ctrl_enc.alpha = alpha_now;

            if strcmpi(enc_init, 'maml_residual')
                w_base_now = [interp_w_sin(f_ctrl); interp_w_cos(f_ctrl)];
                dw_prev = ctrl_enc.w(:,1,1) - w_base_prev;
                ctrl_enc.w(:,1,1) = w_base_now + dw_prev;
            elseif strcmpi(enc_init, 'random_residual')
                w_base_now = [interp_rand_w_sin(f_ctrl); interp_rand_w_cos(f_ctrl)];
                dw_prev = ctrl_enc.w(:,1,1) - w_base_prev;
                ctrl_enc.w(:,1,1) = w_base_now + dw_prev;
            else
                w_base_now = zeros(2,1);
            end

            u_enc = ctrl_enc.iterate(e_buf_enc(end));
        else
            u_enc = 0;
            w_base_now = zeros(2,1);
        end

        u_rnc_buf = [u_rnc, u_rnc_buf(1:end-1)];
        u_enc_buf = [u_enc, u_enc_buf(1:end-1)];

        y_rnc = c.s_vec' * u_rnc_buf';
        y_enc = c.s_vec' * u_enc_buf';
        e_now = c.d_total(n) - y_rnc + y_enc;

        e_prev = e_now;
        if use_enc
            e_buf_enc = [e_now, e_buf_enc(1:end-1)];
            out.w_enc(n,:) = ctrl_enc.w(:,1,1)';
            out.w_base(n,:) = w_base_now';
            out.dw_enc(n,:) = (ctrl_enc.w(:,1,1) - w_base_now)';
            if strcmpi(enc_init, 'maml_residual') || strcmpi(enc_init, 'random_residual')
                w_base_prev = w_base_now;
            end
        end

        out.e(n) = e_now;
        out.y_rnc(n) = y_rnc;
        out.y_enc(n) = y_enc;
        out.u_rnc(n) = u_rnc;
        out.u_enc(n) = u_enc;
    end
    fprintf('  Done %.1f s. Err/Input = %.2f dB\n', ...
        toc, 10*log10(mean(out.e.^2) / mean(c.d_total.^2)));
end

function out = make_off_case(d_total)
    out = struct();
    out.name = "Off";
    out.e = d_total(:);
    out.y_rnc = zeros(size(d_total(:)));
    out.y_enc = zeros(size(d_total(:)));
    out.u_rnc = zeros(size(d_total(:)));
    out.u_enc = zeros(size(d_total(:)));
    out.w_enc = nan(length(d_total), 2);
    out.w_base = nan(length(d_total), 2);
    out.dw_enc = nan(length(d_total), 2);
end

function row = metric_row(name, err, p_ref)
    row = struct();
    row.name = name;
    row.err_db = 10*log10(mean(err(:).^2) / p_ref);
end

function [centers, nr_zero_rnc, nr_rnc, nr_enc, nr_zero_ernc, nr_rand_ernc, nr_ernc] = per_bin_error_ratio( ...
        freq, d_ref, e_zero_rnc, e_rnc, e_enc, e_zero_ernc, e_rand_ernc, e_ernc, f_start, f_end, bin_width)
    edges = (f_start:bin_width:(f_end + bin_width))';
    centers = edges(1:end-1) + bin_width/2;
    nr_zero_rnc = nan(numel(centers), 1);
    nr_rnc = nan(numel(centers), 1);
    nr_enc = nan(numel(centers), 1);
    nr_zero_ernc = nan(numel(centers), 1);
    nr_rand_ernc = nan(numel(centers), 1);
    nr_ernc = nan(numel(centers), 1);
    idx = min(floor((freq - f_start) / bin_width) + 1, numel(centers));
    for bi = 1:numel(centers)
        mask = (idx == bi);
        if sum(mask) < 10
            continue;
        end
        p_d = mean(d_ref(mask).^2);
        nr_zero_rnc(bi) = 10*log10(mean(e_zero_rnc(mask).^2) / p_d);
        nr_rnc(bi) = 10*log10(mean(e_rnc(mask).^2) / p_d);
        nr_enc(bi) = 10*log10(mean(e_enc(mask).^2) / p_d);
        nr_zero_ernc(bi) = 10*log10(mean(e_zero_ernc(mask).^2) / p_d);
        nr_rand_ernc(bi) = 10*log10(mean(e_rand_ernc(mask).^2) / p_d);
        nr_ernc(bi) = 10*log10(mean(e_ernc(mask).^2) / p_d);
    end
end

function u_buf = prewarm_notch_u_buffer(w, f_init, fs, J)
    u_buf = zeros(1, J);
    dtheta = 2*pi * f_init / fs;
    for k = 1:J
        theta_k = -(k-1) * dtheta;
        u_buf(k) = w(1) * sin(theta_k) + w(2) * cos(theta_k);
    end
end

function value = getfield_default(s, name, default_value)
    if isfield(s, name)
        value = s.(name);
    else
        value = default_value;
    end
end

function Wc_random = build_random_rnc_init(Wc_ref, seed)
    rng(seed, 'twister');
    Wc_random = randn(size(Wc_ref));
    ref_norm = norm(Wc_ref);
    rand_norm = norm(Wc_random);
    if ref_norm > 0 && rand_norm > 0
        Wc_random = Wc_random / rand_norm * ref_norm;
    end
end

function rand_weight_table = build_random_enc_table(weight_table, seed)
    rng(seed, 'twister');
    rand_weight_table = randn(size(weight_table, 1), 2);
    ref_norm = vecnorm(weight_table(:, 2:3), 2, 2);
    rand_norm = vecnorm(rand_weight_table, 2, 2);
    valid = rand_norm > 0;
    rand_weight_table(valid, :) = rand_weight_table(valid, :) ./ rand_norm(valid) .* ref_norm(valid);
end

function format_current_figure()
    set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 15);
    set(findall(gcf, 'Type', 'legend'), 'FontSize', 15);
    set(findall(gcf, 'Type', 'ColorBar'), 'FontSize', 15, 'FontName', 'Times New Roman');
    set(findall(gcf, 'Type', 'text'), 'FontSize', 15);
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
