%% MAML-Notch: Multi-channel single-frequency validation
%
%  Loads MAML_weight_table_manual.mat produced by Main_maml_notch.m (M, L >= 1).
%
%  weight_table layout [N_freq x (1 + 2*M*L)]:
%    col 1     : frequency
%    remaining : [sin(m=1,l=1), cos(m=1,l=1),
%                 sin(m=2,l=1), cos(m=2,l=1), ...   <- l=1, m inner
%                 sin(m=1,l=2), cos(m=1,l=2), ...]  <- l=2, m inner
%    (l outer, m inner — same order as Main_maml_notch.m)
%
%  NotchController.w is [2, K, M].
%  Initialisation rule (K=1):
%    w(1, 1, m) = sum_l  Phi_sin(m, l)   (sum trained weights over mics)
%    w(2, 1, m) = sum_l  Phi_cos(m, l)
%
%  NR and convergence time are averaged across L mics.
%  Per-mic detail is shown in the "Samples" figure.
%
%  No primary path.
%  Files needed: NotchController.m, SimulationPlatform.m
%               + MAML_weight_table_manual.mat
%--------------------------------------------------------------------------
close all; clear; clc;

%% =====================  Load  ===========================================
load('MAML_weight_table_manual.mat');
fprintf('Loaded: %d freqs (%.0f - %.0f Hz)  M=%d  L=%d\n', ...
        length(freq_list), freq_list(1), freq_list(end), M, L);

N_freq = length(freq_list);

% Secondary path: training script saves full 's' [J, M, L]
% (older single-ch mat files may have s_coef [J,1] — handle both)
if exist('s', 'var') && isequal(size(s), [J, M, L])
    s_3d = s;
elseif exist('s_coef', 'var')
    s_3d = reshape(s_coef(:), [J, M, L]);
    warning('s_coef reshaped to [%d,%d,%d] — verify dimensions.', J, M, L);
else
    error('No secondary path variable found in mat file.');
end

%% =====================  Helper: read w_init from weight_table  ==========
% weight_table col layout: l outer, m inner
% col of (m, l): 2 + (l-1)*2*M + (m-1)*2   -> Phi_sin
%                2 + (l-1)*2*M + (m-1)*2+1  -> Phi_cos
% w(1,1,m) = sum_l Phi_sin(m,l)
% w(2,1,m) = sum_l Phi_cos(m,l)
function w_init = read_w_init(weight_table, fi, M, L)
    w_init = zeros(2, 1, M);   % [2, K=1, M]
    for l = 1:L
        for m = 1:M
            col = 2 + (l-1)*2*M + (m-1)*2;
            w_init(1, 1, m) = w_init(1, 1, m) + weight_table(fi, col);
            w_init(2, 1, m) = w_init(2, 1, m) + weight_table(fi, col+1);
        end
    end
end

%% =====================  Validation parameters  ==========================
T_test         = 2;
N_test         = round(T_test * fs);
t_test         = (0:N_test-1)' / fs;
conv_thresh_dB = -10;
smooth_win     = round(0.05 * fs);

%% =====================  Run  ============================================
% NR and Tc: averaged over L mics
NR_zero      = zeros(N_freq, 1);
NR_maml      = zeros(N_freq, 1);
T_conv_zero  = zeros(N_freq, 1);
T_conv_maml  = zeros(N_freq, 1);

% Store final w for speaker m=1 (representative)
w_final_zero = zeros(N_freq, 2);   % w(:,1,1) sin/cos
w_final_maml = zeros(N_freq, 2);

% Store w_init (sum_l Phi) for speaker m=1, for comparison plot
w_init_table = zeros(N_freq, 2);
for fi = 1:N_freq
    wi = read_w_init(weight_table, fi, M, L);
    w_init_table(fi, :) = [wi(1,1,1), wi(2,1,1)];
end

fprintf('\n========== Per-frequency validation (M=%d, L=%d) ==========\n', M, L);
tic;

for fi = 1:N_freq
    f0      = freq_list(fi);
    alpha_f = 50 * alpha_table(fi);

    % Disturbance: same sine on all L mics [N_test x L]
    theta_t = 0;
    d_test  = zeros(N_test, L);
    for n = 1:N_test
        theta_t    = mod(theta_t + 2*pi*f0/fs, 2*pi);
        d_test(n,:) = sin(theta_t);
    end
    d_pow = mean(d_test(:).^2);   % scalar, averaged over time and mics

    % ── Zero-init ─────────────────────────────────────────────────────────
    ctrl0 = NotchController(fs, f0, alpha_f, M, L, J);
    ctrl0.s = s_3d;
    plat0   = SimulationPlatform(d_test, s_3d, M, L);
    e0      = zeros(N_test, L);
    for n = 1:N_test
        u = ctrl0.iterate(plat0.e);
        plat0.sim(u);
        e0(n,:) = plat0.e';
    end
    w_final_zero(fi,:) = ctrl0.w(:, 1, 1)';   % speaker m=1

    e0_avg = mean(e0.^2, 2);                   % [N_test x 1] avg over mics
    NR_zero(fi) = log10(mean(e0_avg) / d_pow);
    nr_i = 10*log10(movmean(e0_avg, smooth_win) / d_pow + eps);
    ic = find(nr_i < conv_thresh_dB, 1);
    T_conv_zero(fi) = ifelse_scalar(isempty(ic), T_test, ic/fs);

    % ── MAML-init ─────────────────────────────────────────────────────────
    ctrlM = NotchController(fs, f0, alpha_f, M, L, J);
    ctrlM.s = s_3d;
    % Initialise w: w(i,1,m) = sum_l Phi(m,l)
    ctrlM.w = read_w_init(weight_table, fi, M, L);

    platM = SimulationPlatform(d_test, s_3d, M, L);
    eM    = zeros(N_test, L);
    for n = 1:N_test
        u = ctrlM.iterate(platM.e);
        platM.sim(u);
        eM(n,:) = platM.e';
    end
    w_final_maml(fi,:) = ctrlM.w(:, 1, 1)';

    eM_avg = mean(eM.^2, 2);
    NR_maml(fi) = 10*log10(mean(eM_avg) / d_pow);
    nr_i = 10*log10(movmean(eM_avg, smooth_win) / d_pow + eps);
    ic = find(nr_i < conv_thresh_dB, 1);
    T_conv_maml(fi) = ifelse_scalar(isempty(ic), T_test, ic/fs);

    if mod(fi,50)==0 || fi==1 || fi==N_freq
        fprintf('  [%3d/%3d] %4dHz  alpha=%.5f  NR %+.1f/%+.1f dB  Tc %.3f/%.3f s\n', ...
            fi, N_freq, f0, alpha_f, NR_zero(fi), NR_maml(fi), ...
            T_conv_zero(fi), T_conv_maml(fi));
    end
end
fprintf('Done. %.1f s\n', toc);

%% =====================  Print  ==========================================
fprintf('\n================== Results (avg over %d mics) ==================\n', L);
fprintf('%5s | %7s %7s | %7s %7s | %8s\n', ...
    'Freq','NR_0','NR_M','Tc_0','Tc_M','alpha');
fprintf('%s\n', repmat('-',1,52));
for fi = 1:N_freq
    fprintf('%4.0f  | %+6.1f %+6.1f | %6.3f %6.3f | %8.5f\n', ...
        freq_list(fi), NR_zero(fi), NR_maml(fi), ...
        T_conv_zero(fi), T_conv_maml(fi), alpha_table(fi));
end

%% =====================  Plots  ==========================================

figure('Name','NR (avg mics)');
plot(freq_list, NR_zero, 'b.-', freq_list, NR_maml, 'r.-', 'MarkerSize',4);
xlabel('Freq (Hz)'); ylabel('NR (dB)');
title(sprintf('Noise Reduction (avg %d mics)', L));
legend('Zero-init','MAML-init'); grid on;

figure('Name','MAML advantage');
bar(freq_list, NR_zero - NR_maml, 'FaceColor',[.2 .7 .3]);
xlabel('Freq (Hz)'); ylabel('\DeltaNR (dB)');
title('MAML Advantage  (positive = MAML better)'); grid on;

figure('Name','Convergence Time');
plot(freq_list, T_conv_zero*1e3, 'b.-', freq_list, T_conv_maml*1e3, 'r.-', 'MarkerSize',4);
xlabel('Freq (Hz)'); ylabel('ms');
title(sprintf('Convergence time to %d dB (avg mics)', conv_thresh_dB));
legend('Zero-init','MAML-init'); grid on;

figure('Name','Speed-up');
bar(freq_list, T_conv_zero ./ max(T_conv_maml, 1/fs), 'FaceColor',[.9 .4 .1]);
xlabel('Freq (Hz)'); ylabel('x'); title('Convergence Speed-up');
yline(1,'k--'); grid on;

% Weights for speaker m=1
figure('Name',sprintf('Weights — speaker m=1 (init w = sum_l Phi)'));
subplot(2,1,1);
plot(freq_list, w_final_zero(:,1),'b.-', ...
     freq_list, w_final_maml(:,1),'r.-', ...
     freq_list, w_init_table(:,1),'k--', 'MarkerSize',3);
ylabel('w_{sin}');
title('w_{sin}  (m=1,  k=1)');
legend('Zero final','MAML final','MAML init (sum_l\Phi_{sin})'); grid on;
subplot(2,1,2);
plot(freq_list, w_final_zero(:,2),'b.-', ...
     freq_list, w_final_maml(:,2),'r.-', ...
     freq_list, w_init_table(:,2),'k--', 'MarkerSize',3);
xlabel('Freq (Hz)'); ylabel('w_{cos}');
title('w_{cos}  (m=1,  k=1)');
legend('Zero final','MAML final','MAML init (sum_l\Phi_{cos})'); grid on;

% Phi per (m, l) from weight_table
figure('Name','Phi per (m,l) from weight_table');
panel = 0;
for l = 1:L
    for m = 1:M
        panel = panel + 1;
        col   = 2 + (l-1)*2*M + (m-1)*2;
        subplot(M*L, 2, 2*panel-1);
        plot(freq_list, weight_table(:,col),   'b.-', 'MarkerSize',3);
        ylabel('\Phi_{sin}');
        title(sprintf('m=%d, l=%d  \\Phi_{sin}', m, l)); grid on;
        subplot(M*L, 2, 2*panel);
        plot(freq_list, weight_table(:,col+1), 'r.-', 'MarkerSize',3);
        ylabel('\Phi_{cos}');
        title(sprintf('m=%d, l=%d  \\Phi_{cos}', m, l)); grid on;
    end
end
xlabel('Frequency (Hz)');

% Sample time-domain detail — per-mic, per-speaker subplots
sample_f = [20 50 100 150 200 250 300];
sample_f = sample_f(sample_f >= freq_list(1) & sample_f <= freq_list(end));

if ~isempty(sample_f)
    w2 = round(0.02 * fs);

    for si = 1:length(sample_f)
        f0 = sample_f(si);
        fi = find(freq_list == f0, 1);
        if isempty(fi); continue; end
        af = alpha_table(fi);

        % Disturbance
        th = 0; dt = zeros(N_test, L);
        for n = 1:N_test
            th = mod(th + 2*pi*f0/fs, 2*pi);
            dt(n,:) = sin(th);
        end

        % Zero-init run
        c0 = NotchController(fs, f0, af, M, L, J); c0.s = s_3d;
        p0 = SimulationPlatform(dt, s_3d, M, L);
        e0t = zeros(N_test, L);
        for n = 1:N_test; u=c0.iterate(p0.e); p0.sim(u); e0t(n,:)=p0.e'; end

        % MAML-init run
        cM = NotchController(fs, f0, af, M, L, J); cM.s = s_3d;
        cM.w = read_w_init(weight_table, fi, M, L);
        pM = SimulationPlatform(dt, s_3d, M, L);
        eMt = zeros(N_test, L);
        for n = 1:N_test; u=cM.iterate(pM.e); pM.sim(u); eMt(n,:)=pM.e'; end

        % One figure per sample frequency, L subplots (one per mic)
        figure('Name', sprintf('Sample %dHz', f0));
        for l = 1:L
            subplot(L, 1, l);
            plot(t_test, 10*log10(movmean(dt(:,l).^2,   w2)+eps), 'Color',[.7 .7 .7]); hold on;
            plot(t_test, 10*log10(movmean(e0t(:,l).^2,  w2)+eps), 'b', 'LineWidth',.8);
            plot(t_test, 10*log10(movmean(eMt(:,l).^2,  w2)+eps), 'r', 'LineWidth',.8);
            ylabel('dB');
            title(sprintf('%dHz — Mic l=%d  (\\alpha=%.5f)', f0, l, af));
            grid on;
            if l == 1; legend('Off','Zero-init','MAML-init','Location','best'); end
        end
        xlabel('Time (s)');
    end
end

fprintf('\n=== Done ===\n');

%% =====================  Local helper  ===================================
function v = ifelse_scalar(cond, a, b)
    if cond; v = a; else; v = b; end
end