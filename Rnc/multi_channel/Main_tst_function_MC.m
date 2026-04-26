%% Title: Testing multi-channel MAML-FxLMS (1x2x2)
%  K=1 reference, M=2 speakers, L=2 error microphones
%  Secondary paths: S11, S12, S21, S22
%  Primary paths: P1, P2
%  Author: Adapted from Dongyan Shi's single-channel version

%% Clean workspace
close all;
clear;
clc;
addpath("class\");
addpath("data\");
addpath("path\");
%% System configuration
fs      = 16000;        % Sampling rate
T       = 3;            % Simulation duration (seconds)
t       = 0:1/fs:T;
N_total = length(t);

Len_N   = 512;          % Control filter length (per sub-filter)
K = 1;                  % Number of references
M = 2;                  % Number of speakers
L = 2;                  % Number of error mics

%% Load acoustic paths
% Modify these paths according to your file locations
load('path\PrimaryPath_256order.mat');       % Contains P1, P2 (256x1 each)
load('path\SecondaryPath_2x2_256order.mat'); % Contains S11, S12, S21, S22 (256x1 each)

J_path = length(S11);  % Secondary path order = 256

% Assemble secondary path matrix [J, M, L]
% s(:, m, l) = path from speaker m to mic l
s_path = zeros(J_path, M, L);
s_path(:, 1, 1) = S11;   % Speaker 1 -> Mic 1
s_path(:, 1, 2) = S12;   % Speaker 1 -> Mic 2
s_path(:, 2, 1) = S21;   % Speaker 2 -> Mic 1
s_path(:, 2, 2) = S22;   % Speaker 2 -> Mic 2

% Assemble primary path matrix [O, K, L]
% p(:, k, l) = path from noise source k to mic l
O_path = length(P1);
p_path = zeros(O_path, K, L);
p_path(:, 1, 1) = P1;    % Noise source -> Mic 1
p_path(:, 1, 2) = P2;    % Noise source -> Mic 2

% Combined primary path (for generating disturbance in training)
% Not needed if using SimulationPlatformOSPM_MC directly

% Use the physical secondary path as the estimated one (perfect estimation)
s_hat = s_path;

%% Progress bar
f = waitbar(0, 'Please wait...');
pause(0.5);

%% Build broadband noise for training
waitbar(0.1, f, 'Generating training noise...');

Track_num = 3;  % Number of training noise tracks

% Generate broadband noise
Noise = randn(N_total, 1);

% Bandpass filters for different frequency bands
filter_1 = fir1(512, [0.05 0.25]);
filter_2 = fir1(512, [0.20 0.55]);
filter_3 = fir1(512, [0.5  0.75]);

% Primary noise (reference signals)
Pri_1 = filter(filter_1, 1, Noise);
Pri_2 = filter(filter_2, 1, Noise);
Pri_3 = filter(filter_3, 1, Noise);

% Generate disturbance at each mic through primary paths
% d_l(n) = sum_k p(:,k,l) * x_k(n)
% Here K=1, so d_l(n) = p(:,1,l) * x(n)
Dis_1_L1 = filter(P1, 1, Pri_1);  % Track1, Mic1
Dis_1_L2 = filter(P2, 1, Pri_1);  % Track1, Mic2
Dis_2_L1 = filter(P1, 1, Pri_2);  % Track2, Mic1
Dis_2_L2 = filter(P2, 1, Pri_2);  % Track2, Mic2
Dis_3_L1 = filter(P1, 1, Pri_3);  % Track3, Mic1
Dis_3_L2 = filter(P2, 1, Pri_3);  % Track3, Mic2

% Generate filtered-reference signals through estimated secondary paths
% fx_{mlk}(n) = s_hat(:,m,l) * x_k(n)
% For K=1, we have fx_{ml}(n) for each (m,l) pair
% Track 1
Rf_1_11 = filter(S11, 1, Pri_1);  % s_hat(1,1) * x, Track1
Rf_1_12 = filter(S12, 1, Pri_1);  % s_hat(1,2) * x, Track1
Rf_1_21 = filter(S21, 1, Pri_1);  % s_hat(2,1) * x, Track1
Rf_1_22 = filter(S22, 1, Pri_1);  % s_hat(2,2) * x, Track1
% Track 2
Rf_2_11 = filter(S11, 1, Pri_2);
Rf_2_12 = filter(S12, 1, Pri_2);
Rf_2_21 = filter(S21, 1, Pri_2);
Rf_2_22 = filter(S22, 1, Pri_2);
% Track 3
Rf_3_11 = filter(S11, 1, Pri_3);
Rf_3_12 = filter(S12, 1, Pri_3);
Rf_3_21 = filter(S21, 1, Pri_3);
Rf_3_22 = filter(S22, 1, Pri_3);

%% Randomly sample noise segments to build MAML training dataset
waitbar(0.3, f, 'Building MAML training dataset...');

N_epoch  = 4096 * 80;                      % Number of training epochs
Trac     = randi(Track_num, [N_epoch, 1]);  % Random track selection
Len_data = length(Dis_1_L1);

len = 2 * Len_N - 1;  % Minimum starting index

% Pre-allocate: filtered-reference segments [Len_N, M, L, K, N_epoch]
% and disturbance segments [Len_N, L, N_epoch]
Fx_data = zeros(Len_N, M, L, K, N_epoch);
Di_data = zeros(Len_N, L, N_epoch);

% Organize data by track for easy indexing
% Disturbance: [N_total, L, Track_num]
Dis_all = zeros(Len_data, L, Track_num);
Dis_all(:, 1, 1) = Dis_1_L1; Dis_all(:, 2, 1) = Dis_1_L2;
Dis_all(:, 1, 2) = Dis_2_L1; Dis_all(:, 2, 2) = Dis_2_L2;
Dis_all(:, 1, 3) = Dis_3_L1; Dis_all(:, 2, 3) = Dis_3_L2;

% Filtered-reference: [N_total, M, L, Track_num] (K=1 omitted for indexing)
Rf_all = zeros(Len_data, M, L, Track_num);
Rf_all(:, 1, 1, 1) = Rf_1_11; Rf_all(:, 1, 2, 1) = Rf_1_12;
Rf_all(:, 2, 1, 1) = Rf_1_21; Rf_all(:, 2, 2, 1) = Rf_1_22;
Rf_all(:, 1, 1, 2) = Rf_2_11; Rf_all(:, 1, 2, 2) = Rf_2_12;
Rf_all(:, 2, 1, 2) = Rf_2_21; Rf_all(:, 2, 2, 2) = Rf_2_22;
Rf_all(:, 1, 1, 3) = Rf_3_11; Rf_all(:, 1, 2, 3) = Rf_3_12;
Rf_all(:, 2, 1, 3) = Rf_3_21; Rf_all(:, 2, 2, 3) = Rf_3_22;

for jj = 1:N_epoch
    tr = Trac(jj);
    End_idx = randi([len, Len_data]);
    Start_idx = End_idx - Len_N + 1;

    % Extract disturbance segment [Len_N, L]
    Di_data(:, :, jj) = Dis_all(Start_idx:End_idx, :, tr);

    % Extract filtered-reference segment [Len_N, M, L, K=1]
    for m = 1:M
        for l = 1:L
            Fx_data(:, m, l, 1, jj) = Rf_all(Start_idx:End_idx, m, l, tr);
        end
    end
end

%% Run Modified MAML algorithm to compute optimal initial control filter
waitbar(0.5, f, 'Running MAML training...');

% Create MAML object
maml = MAML_Nstep_forget_MC(Len_N, K, M, L);

N_train = size(Di_data, 3);   % Number of training samples
Er_train = zeros(N_train, 1);

% MAML hyperparameters
mu     = 0.0003;    % Step size for inner FxLMS
lamda  = 0.99;      % Forgetting factor
epslon = 0.5;       % MAML outer learning rate

fprintf('Starting MAML training with %d epochs...\n', N_train);
for jj = 1:N_train
    if mod(jj, 10000) == 0
        waitbar(0.5 + 0.3 * jj / N_train, f, ...
            sprintf('MAML training: epoch %d / %d', jj, N_train));
    end

    Fx_seg = Fx_data(:, :, :, :, jj);  % [Len_N, M, L, K]
    Di_seg = Di_data(:, :, jj);         % [Len_N, L]

    [maml, Er_train(jj)] = maml.MAML_initial(Fx_seg, Di_seg, s_hat, mu, lamda, epslon);
end

% Plot MAML learning curve
figure;
plot(Er_train);
grid on;
title('Learning curve of multi-channel MAML algorithm');
xlabel('Epoch');
ylabel('Residual error (sum of squared errors)');

% Get the optimal initial control filter
Wc_maml = maml.Phi;

%% Test: Aircraft noise cancellation
waitbar(0.85, f, 'Testing aircraft noise cancellation...');

% Load aircraft noise
aircrat = load('707_Sound_for_Simulation.mat');
Pri_test = aircrat.PilingNoise(1:153945);

% Reference signal [N_test, K]
x_test = Pri_test(:);

N_test = length(x_test);

% Step size for testing
muw = 0.00001;

% --- Test 1: FxLMS with zero initialization ---
fprintf('Running McFxLMS with zero initialization...\n');
Wc_zero = zeros(Len_N * M * K, 1);
[Er_zero, ~] = McFxLMS(Len_N, Wc_zero, x_test, s_path, p_path, s_hat, muw, N_test);

% --- Test 2: FxLMS with MAML initialization ---
fprintf('Running McFxLMS with MAML initialization...\n');
[Er_maml_test, platform_maml] = McFxLMS(Len_N, Wc_maml, x_test, s_path, p_path, s_hat, muw, N_test);

% --- Generate disturbance (ANC off) for reference ---
Dis_test_L1 = filter(P1, 1, Pri_test);
Dis_test_L2 = filter(P2, 1, Pri_test);

%% Plot results
time_axis = (0:N_test-1) / fs;

% --- Mic 1 ---
figure;
plot(time_axis, Dis_test_L1, 'b', ...
     time_axis, Er_zero(:,1), 'r', ...
     time_axis, Er_maml_test(:,1), 'g');
title('Aircraft noise cancellation - Microphone 1');
xlabel('Time (seconds)');
ylabel('Error signal');
legend({'ANC off', 'McFxLMS zero-init', 'McFxLMS MAML-init'});
grid on;

% --- Mic 2 ---
figure;
plot(time_axis, Dis_test_L2, 'b', ...
     time_axis, Er_zero(:,2), 'r', ...
     time_axis, Er_maml_test(:,2), 'g');
title('Aircraft noise cancellation - Microphone 2');
xlabel('Time (seconds)');
ylabel('Error signal');
legend({'ANC off', 'McFxLMS zero-init', 'McFxLMS MAML-init'});
grid on;

% --- Total error power ---
figure;
Er_zero_power = sum(Er_zero.^2, 2);
Er_maml_power = sum(Er_maml_test.^2, 2);
Dis_power = Dis_test_L1.^2 + Dis_test_L2.^2;

% Smoothing for visualization
win_len = 256;
Er_zero_smooth = movmean(Er_zero_power, win_len);
Er_maml_smooth = movmean(Er_maml_power, win_len);
Dis_smooth = movmean(Dis_power, win_len);

plot(time_axis, 10*log10(Dis_smooth), 'b', ...
     time_axis, 10*log10(Er_zero_smooth), 'r', ...
     time_axis, 10*log10(Er_maml_smooth), 'g');
title('Total error power comparison');
xlabel('Time (seconds)');
ylabel('Error power (dB)');
legend({'ANC off', 'McFxLMS zero-init', 'McFxLMS MAML-init'});
grid on;

waitbar(1, f, 'Finished!');
pause(1);
close(f);

fprintf('Simulation completed!\n');
%-------------------------------end----------------------------------------
