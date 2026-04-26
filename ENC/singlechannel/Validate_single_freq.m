%% MAML-Notch: Single-channel single-frequency validation
%
%  M=1 loudspeaker, L=1 error mic.
%  Loads MAML_weight_table_manual.mat and validates per frequency:
%    - Zero-init FxLMS vs MAML-init FxLMS
%    - NR (dB), convergence time, final weights
%
%  No primary path.
%  Files needed: NotchController.m, SimulationPlatform.m
%               + MAML_weight_table_manual.mat
%--------------------------------------------------------------------------
close all; clear; clc;

%% =====================  Load  ===========================================
load('MAML_weight_table_manual.mat');
fprintf('Loaded: %d freqs (%.0f - %.0f Hz)\n', ...
        length(freq_list), freq_list(1), freq_list(end));

M = 1;  L = 1;
% Accept s_coef [J x 1] or s_3d [J,1,1]
if exist('s_3d','var') && isequal(size(s_3d),[J,1,1])
    % already correct
elseif exist('s_coef','var')
    s_3d = reshape(s_coef(:), [J, 1, 1]);
else
    error('No secondary path found in mat file.');
end
N_freq = length(freq_list);

%% =====================  Validation parameters  ==========================
T_test         = 2;
N_test         = round(T_test * fs);
t_test         = (0:N_test-1)' / fs;
conv_thresh_dB = -10;
smooth_win     = round(0.05 * fs);

%% =====================  Run  ============================================
NR_zero      = zeros(N_freq, 1);
NR_maml      = zeros(N_freq, 1);
T_conv_zero  = zeros(N_freq, 1);
T_conv_maml  = zeros(N_freq, 1);
w_final_zero = zeros(N_freq, 2);
w_final_maml = zeros(N_freq, 2);

fprintf('\n========== Per-frequency validation (M=1, L=1) ==========\n');
tic;

for fi = 1:N_freq
    f0      = freq_list(fi);
    alpha_f =  0.1*alpha_table(fi);

    % Disturbance [N_test x 1]
    theta_t = 0;
    d_test  = zeros(N_test, 1);
    for n = 1:N_test
        theta_t   = mod(theta_t + 2*pi*f0/fs, 2*pi);
        d_test(n) = sin(theta_t);
    end
    d_pow = mean(d_test.^2);

    % ── Zero-init ─────────────────────────────────────────────────────────
    ctrl0 = NotchController(fs, f0, alpha_f, M, L, J);
    ctrl0.s = s_3d;
    plat0   = SimulationPlatform(d_test, s_3d, M, L);
    e0 = zeros(N_test, 1);
    for n = 1:N_test
        u = ctrl0.iterate(plat0.e); plat0.sim(u); e0(n) = plat0.e;
    end
    w_final_zero(fi,:) = ctrl0.w(:,1,1)';
    NR_zero(fi) = 10*log10(mean(e0.^2) / d_pow);
    nr_i = 10*log10(movmean(e0.^2, smooth_win) / d_pow + eps);
    ic = find(nr_i < conv_thresh_dB, 1);
    T_conv_zero(fi) = ifelse_val(isempty(ic), T_test, ic/fs);

    % ── MAML-init ─────────────────────────────────────────────────────────
    ctrlM = NotchController(fs, f0, alpha_f, M, L, J);
    ctrlM.s = s_3d;
    ctrlM.w(1,1,1) = weight_table(fi, 2);   % Phi_sin
    ctrlM.w(2,1,1) = weight_table(fi, 3);   % Phi_cos
    platM  = SimulationPlatform(d_test, s_3d, M, L);
    eM = zeros(N_test, 1);
    for n = 1:N_test
        u = ctrlM.iterate(platM.e); platM.sim(u); eM(n) = platM.e;
    end
    w_final_maml(fi,:) = ctrlM.w(:,1,1)';
    NR_maml(fi) = 10*log10(mean(eM.^2) / d_pow);
    nr_i = 10*log10(movmean(eM.^2, smooth_win) / d_pow + eps);
    ic = find(nr_i < conv_thresh_dB, 1);
    T_conv_maml(fi) = ifelse_val(isempty(ic), T_test, ic/fs);

    if mod(fi,50)==0 || fi==1 || fi==N_freq
        fprintf('  [%3d/%3d] %4dHz  alpha=%.5f  NR %+.1f/%+.1f dB  Tc %.3f/%.3f s\n', ...
            fi, N_freq, f0, alpha_f, NR_zero(fi), NR_maml(fi), ...
            T_conv_zero(fi), T_conv_maml(fi));
    end
end
fprintf('Done. %.1f s\n', toc);

%% =====================  Print  ==========================================
fprintf('\n================== Results ==================\n');
fprintf('%5s | %7s %7s | %7s %7s | %8s\n', ...
    'Freq','NR_0','NR_M','Tc_0','Tc_M','alpha');
fprintf('%s\n', repmat('-',1,52));
for fi = 1:N_freq
    fprintf('%4.0f  | %+6.1f %+6.1f | %6.3f %6.3f | %8.5f\n', ...
        freq_list(fi), NR_zero(fi), NR_maml(fi), ...
        T_conv_zero(fi), T_conv_maml(fi), alpha_table(fi));
end

%% =====================  Plots  ==========================================
figure('Name','NR');
plot(freq_list, NR_zero, 'b.-', freq_list, NR_maml, 'r.-', 'MarkerSize',4);
xlabel('Freq (Hz)'); ylabel('NR (dB)');
title('Noise Reduction'); legend('Zero-init','MAML-init'); grid on;

figure('Name','MAML advantage');
bar(freq_list, NR_zero - NR_maml, 'FaceColor',[.2 .7 .3]);
xlabel('Freq (Hz)'); ylabel('\DeltaNR (dB)');
title('MAML Advantage (positive = MAML better)'); grid on;

figure('Name','Convergence Time');
plot(freq_list, T_conv_zero*1e3, 'b.-', freq_list, T_conv_maml*1e3, 'r.-', 'MarkerSize',4);
xlabel('Freq (Hz)'); ylabel('ms');
title(sprintf('Convergence time to %d dB', conv_thresh_dB));
legend('Zero-init','MAML-init'); grid on;

figure('Name','Speed-up');
bar(freq_list, T_conv_zero ./ max(T_conv_maml, 1/fs), 'FaceColor',[.9 .4 .1]);
xlabel('Freq (Hz)'); ylabel('x'); title('Convergence Speed-up');
yline(1,'k--'); grid on;

figure('Name','Weights');
subplot(2,1,1);
plot(freq_list, w_final_zero(:,1),'b.-', freq_list, w_final_maml(:,1),'r.-', ...
     freq_list, weight_table(:,2),'k--', 'MarkerSize',3);
ylabel('w_{sin}'); legend('Zero final','MAML final','MAML init'); grid on;
subplot(2,1,2);
plot(freq_list, w_final_zero(:,2),'b.-', freq_list, w_final_maml(:,2),'r.-', ...
     freq_list, weight_table(:,3),'k--', 'MarkerSize',3);
xlabel('Freq (Hz)'); ylabel('w_{cos}'); legend('Zero final','MAML final','MAML init'); grid on;

% Sample time-domain detail
sample_f = [20 50 100 150 200 250 300];
sample_f = sample_f(sample_f >= freq_list(1) & sample_f <= freq_list(end));
if ~isempty(sample_f)
    w2 = round(0.02*fs);
    figure('Name','Samples');
    for si = 1:length(sample_f)
        f0 = sample_f(si);
        fi = find(freq_list == f0, 1); if isempty(fi); continue; end
        af = alpha_table(fi);
        th = 0; dt = zeros(N_test,1);
        for n=1:N_test; th=mod(th+2*pi*f0/fs,2*pi); dt(n)=sin(th); end

        c0=NotchController(fs,f0,af,M,L,J); c0.s=s_3d;
        p0=SimulationPlatform(dt,s_3d,M,L); e0t=zeros(N_test,1);
        for n=1:N_test; u=c0.iterate(p0.e); p0.sim(u); e0t(n)=p0.e; end

        cM=NotchController(fs,f0,af,M,L,J); cM.s=s_3d;
        cM.w(1,1,1)=weight_table(fi,2); cM.w(2,1,1)=weight_table(fi,3);
        pM=SimulationPlatform(dt,s_3d,M,L); eMt=zeros(N_test,1);
        for n=1:N_test; u=cM.iterate(pM.e); pM.sim(u); eMt(n)=pM.e; end

        subplot(length(sample_f),1,si);
        plot(t_test, 10*log10(movmean(dt.^2,  w2)+eps), 'Color',[.7 .7 .7]); hold on;
        plot(t_test, 10*log10(movmean(e0t.^2, w2)+eps), 'b', 'LineWidth',.8);
        plot(t_test, 10*log10(movmean(eMt.^2, w2)+eps), 'r', 'LineWidth',.8);
        ylabel('dB'); title(sprintf('%dHz  (alpha=%.5f)',f0,af)); grid on;
        if si==1; legend('Off','Zero-init','MAML-init','Location','best'); end
        if si==length(sample_f); xlabel('Time (s)'); end
    end
end

fprintf('\n=== Done ===\n');

%% =====================  Local helper  ===================================
function v = ifelse_val(cond, a, b)
    if cond; v = a; else; v = b; end
end
