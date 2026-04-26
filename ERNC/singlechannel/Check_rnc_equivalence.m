%% Check RNC branch equivalence between Main_tst_function style and ERNC style
close all; clear; clc;

this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(fileparts(this_dir));
rnc_dir  = fullfile(root_dir, 'Rnc', 'singlechannel');

addpath(fullfile(rnc_dir, 'class'));

air = load(fullfile(rnc_dir, 'data', '707_Sound_for_Simulation.mat'));
pri = load(fullfile(rnc_dir, 'path', 'PrimaryPath_256order.mat'));
sec = load(fullfile(rnc_dir, 'path', 'SecondaryPath_2x2_256order.mat'));
rnc = load(fullfile(rnc_dir, 'Weigth_initiate_Nstep_forget.mat'));

fs = 16000;
J = 256;
Len_N = 512;
muw = 1e-5;
N = min(numel(air.PilingNoise), 16000 * 3);

pri_path = conv(pri.P1(:), sec.S11(:));
x_raw = air.PilingNoise(1:N);
d_raw = filter(pri_path, 1, x_raw);
scale = 1 / max(sqrt(mean(d_raw.^2)), eps);
x = x_raw * scale;
d = d_raw * scale;
s_vec = sec.S11(:);
s_3d = reshape(s_vec, [J, 1, 1]);

fprintf('Checking RNC equivalence on road-only signal, N=%d samples.\n', N);

%% Main_tst_function style: SimulationPlatform owns the secondary path state.
sim0 = SimulationPlatform(x, d, s_vec);
ctrl0 = Controller(muw, 1, 1, 1);
ctrl0.s = sim0.s;
ctrl0.w = zeros(Len_N, 1, 1);
e_main_zero = zeros(N, 1);
for n = 1:N
    ecurrent = sim0.e;
    e_main_zero(n) = ecurrent;
    u = ctrl0.iterate(sim0.x(n,:), ecurrent);
    sim0.sim(u);
end

simM = SimulationPlatform(x, d, s_vec);
ctrlM = Controller(muw, 1, 1, 1);
ctrlM.s = simM.s;
ctrlM.w = reshape(rnc.Wc(:), Len_N, 1, 1);
e_main_maml = zeros(N, 1);
for n = 1:N
    ecurrent = simM.e;
    e_main_maml(n) = ecurrent;
    u = ctrlM.iterate(simM.x(n,:), ecurrent);
    simM.sim(u);
end

%% ERNC style: manually maintain secondary path buffer and record post-control error.
e_ernc_zero = run_manual_rnc(x, d, s_vec, s_3d, zeros(Len_N,1), muw);
e_ernc_maml = run_manual_rnc(x, d, s_vec, s_3d, rnc.Wc(:), muw);

%% Align Main_tst_function error by one sample because it records e before sim().
e_main_zero_post = [e_main_zero(2:end); sim0.e];
e_main_maml_post = [e_main_maml(2:end); simM.e];

fprintf('\nFull error/input ratio, post-control alignment:\n');
fprintf('  Main zero: %+7.3f dB\n', 10*log10(mean(e_main_zero_post.^2) / mean(d.^2)));
fprintf('  ERNC zero: %+7.3f dB\n', 10*log10(mean(e_ernc_zero.^2) / mean(d.^2)));
fprintf('  Main MAML: %+7.3f dB\n', 10*log10(mean(e_main_maml_post.^2) / mean(d.^2)));
fprintf('  ERNC MAML: %+7.3f dB\n', 10*log10(mean(e_ernc_maml.^2) / mean(d.^2)));

fprintf('\nMax abs post-control difference:\n');
fprintf('  zero: %.3e\n', max(abs(e_main_zero_post - e_ernc_zero)));
fprintf('  MAML: %.3e\n', max(abs(e_main_maml_post - e_ernc_maml)));

figure('Name', 'RNC equivalence check');
t = (0:N-1)' / fs;
subplot(2,1,1);
plot(t, e_main_zero_post, 'b'); hold on;
plot(t, e_ernc_zero, 'r--');
title('Zero-init RNC: Main style vs ERNC manual style');
legend('Main post-control', 'ERNC manual'); grid on;
subplot(2,1,2);
plot(t, e_main_maml_post, 'b'); hold on;
plot(t, e_ernc_maml, 'r--');
title('MAML-init RNC: Main style vs ERNC manual style');
legend('Main post-control', 'ERNC manual'); grid on;

function e_out = run_manual_rnc(x, d, s_vec, s_3d, w_init, muw)
    N = numel(d);
    J = numel(s_vec);
    ctrl = Controller(muw, 1, 1, 1);
    ctrl.s = s_3d;
    ctrl.w = reshape(w_init, ctrl.I, ctrl.K, ctrl.M);

    u_buf = zeros(1, J);
    e_prev = 0;
    e_out = zeros(N, 1);
    for n = 1:N
        u = ctrl.iterate(x(n), e_prev);
        u_buf = [u, u_buf(1:end-1)];
        y = s_vec' * u_buf';
        e_now = d(n) - y;
        e_out(n) = e_now;
        e_prev = e_now;
    end
end

