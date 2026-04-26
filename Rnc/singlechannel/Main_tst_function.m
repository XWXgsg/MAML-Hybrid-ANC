%   ____________________________   ________   __  ______     __  _____
%  /_  __/ ____/ ___/_  __/  _/ | / / ____/  /  |/  /   |   /  |/  / /
%   / / / __/  \__ \ / /  / //  |/ / / __   / /|_/ / /| |  / /|_/ / /
%  / / / /___ ___/ // / _/ // /|  / /_/ /  / /  / / ___ | / /  / / /___
% /_/ /_____//____//_/ /___/_/ |_/\____/  /_/  /_/_/  |_|/_/  /_/_____/

%% Title: Testing the MAML algorithm to accelerate the convergence of the FxLMS algorithm
% Author: DONGYAN SHI(DSHI003@ntu.edu.sg)
% Date  : 2019-9-1
%% Introduction
% In this numerical simulaiton, the MAML algorithm is used to improve the
% convergence of the FxLMS algorithm. The MAML algorithm uses three
% different brodband noise to compute the initial control filter for the
% FxLMS alogrithm. Then, this MAML-initial control filter is used to control 
% the aricraft noise. 
%% Clean the memory and worksapace 
close all ;
clear     ;
clc       ;
addpath("class\");
addpath("data\");
addpath("path\");
%% Configure the system simulation condition 
fs  =   16000    ; % The system sampling rate.
T   =   3        ; % The duration of the simulation.
t   =   0:1/fs:T ; 
N   =   length(t); % The number of the data.

Len_N = 512      ; % Seting the length of the control filter.
%<<===Progress bar===>> 
f = waitbar(0,'Please wait...');
pause(.5)

%% Build the broad band noise for training set
%<<===Progress bar===>> 
waitbar(0.25,f,'Build the broad band noise for training set');
pause(1)
%<<===Progress bar===>> 
% Loading path 
load('path\PrimaryPath_256order.mat')    ;
load('path\SecondaryPath_2x2_256order.mat')   ;
Pri_path = conv(P1,S11);
% Pri_path = P1;

Track_num = 3         ; % Seting the number of the track for the trainning noise. 
if exist('Primary_noise.mat', 'file') == 2
    disp('Primary_noise exists in the current path.\n');
    % Loading the primary noise 
    load('Primary_noise.mat');
    load('Disturbance.mat')  ;
    load('Reference.mat')    ;
else
Noise     = randn(N,1);
% filter 
% filter_1 = fir1(512,[0.05 0.25]);
% filter_2 = fir1(512,[0.20 0.55]) ;
% filter_3 = fir1(512,[0.5,0.75]) ;
filter_1 = fir1(512,[20 150]/(fs/2));
filter_2 = fir1(512,[80 200]/(fs/2));
filter_3 = fir1(512,[150 500]/(fs/2));
% Primary noise 
Pri_1 = filter(filter_1,1,Noise) ;
Pri_2 = filter(filter_2,1,Noise) ;
Pri_3 = filter(filter_3,1,Noise) ;
% Drawing fiture 
data = [Pri_1,Pri_2,Pri_3];
figure ;
len_fft = length(Pri_1)   ;
len_hal = round(len_fft/2);
title('Frequency spectrum of primary noises')
for ii = 1:3
    freq = 20*log(abs(fft(data(:,ii))));
    subplot(3,1,ii);
    plot(0:(fs/len_fft):(len_hal-1)*(fs/len_fft), freq(1:len_hal));
    grid on   ;
    title("The "+num2str(ii)+"th primary noise")
    xlabel('Frequency (Hz)')
end
% Save primary noise into workspace 
save('Primary_noise.mat','Pri_1','Pri_2','Pri_3');
% Generating Distrubance 
Dis_1 = filter(Pri_path,1,Pri_1);
Dis_2 = filter(Pri_path,1,Pri_2);
Dis_3 = filter(Pri_path,1,Pri_3);
% Save distrubancec into workspace 
save('Disturbance.mat','Dis_1','Dis_2','Dis_3');
% Genrating Filtered reference signal 
Rf_1 = filter(S11,1,Pri_1);
Rf_2 = filter(S11,1,Pri_2);
Rf_3 = filter(S11,1,Pri_3);
% Save filter reference signal into workspace 
save('Reference.mat','Rf_1','Rf_2','Rf_3');
end

%% Radomly sampling the noise tracks to build dataset for the MAML algorithm
%<<===Progress bar===>> 
waitbar(0.5,f,'Radomly sampling the noise tracks to build dataset for the MAML algorithm');
pause(1)
%<<===Progress bar===>> 
if exist('Sampe_data_N_set.mat', 'file') == 2
    disp('Sampe_data_N_set in the current path.\n');
    load('Sampe_data_N_set.mat');
else
N_epcho  = 4096 * 80                   ; % Setting the number of the epcho 
Trac     = randi(Track_num,[N_epcho,1]); % Randomly choosing the different tracks. 
Len_data = length(Dis_1)               ;
% Seting the N steps 
len   = 2*Len_N -1 ;
Fx_data = zeros(Len_N,N_epcho);
Di_data = zeros(Len_N,N_epcho);
Ref_data = [Rf_1,Rf_2,Rf_3]   ;
Dis_data = [Dis_1,Dis_2,Dis_3];
for jj = 1:N_epcho
    End = randi([len,Len_data]);
    Di_data(:,jj) = Dis_data(End-511:End,Trac(jj));
    Fx_data(:,jj) = Ref_data(End-511:End,Trac(jj));
end
save('Sampe_data_N_set.mat','Di_data','Fx_data');
end

%% Using Modified MAML algorithm to get the best initial control filter
%<<===Progress bar===>> 
waitbar(0.75,f,'Using Modified MAML algorithm to get the best initial control filter');
pause(1)
%<<===Progress bar===>> 
if exist('Weigth_initiate_Nstep_forget.mat', 'file') == 2
    disp('Weigth_initiate_Nstep_forget in the current path.\n');
    load('Weigth_initiate_Nstep_forget.mat');
else
% Create a MAML algorithm
a  = MAML_Nstep_forget(Len_N);
N  = size(Di_data,2)         ; % The number of the sample in training set.
Er = zeros(N,1)              ; % Residual error vector
% Seting the step size for the embeded FxLMS algorithm 
mu    = 0.0003              ;
% Seting the forget factor 
lamda = 0.99                ;
% Seting the learning for MAML 
epslon = 0.5 ;
% Runing the MAML algorithm 
for jj = 1:N
    [a, Er(jj)] = a.MAML_initial(Fx_data(:,jj),Di_data(:,jj),mu,lamda,epslon);
end
% Drawing the residual error of the Modified MAML algorihtm 
figure    ;
plot(Er)  ;
grid on   ;
title('Leanring curve of the modified MAML algorithm');
xlabel('Epoch');
ylabel('Residual error');
% Getting the optimal intial control filter
Wc = a.Phi ;
% Saving the best initial control filter into workspace
save('Weigth_initiate_Nstep_forget.mat','Wc');
end

%% Testing aircraft noise cancellation by using MAML initial control filter 
%<<===Progress bar===>> 
waitbar(0.95,f,'Testing aircraft noise cancellation by using MAML initial control filter');
pause(1)
%<<===Progress bar===>> 
% Loading aricrat noise data 
% aircrat = load('707_Sound_for_Simulation.mat');
roadnoise = load('CCLQ_60_1.mat');
% Building primary noise
% Pri_1   = aircrat.PilingNoise(1:153945)  
Pri_ori = roadnoise.Signal_0.y_values.values; 
Pri_1 = resample(Pri_ori(1:256000,2),fs,25600);
% Generating the disturbacne 
Dis_1   = filter(Pri_path,1,Pri_1)                  ;
% Generating the filter reference               
Rf_1    = Pri_1                            ;

muw = 0.000001 ; % Step size of All FxLMS algorithms. 
Num_samples = size(Rf_1,1);

%% Runging the FxLMS with the zero-initialization control filter
Wc_initial = zeros(Len_N,1);

simulation = SimulationPlatform(Rf_1, Dis_1, S11);
controller = Controller(muw, 1, 1, 1);
controller.s = simulation.s;
% [FIX] 原来直接赋值 Wc_initial([512,1]) 给 controller.w([512,1,1])
% MATLAB 赋值时形状不匹配会静默 reshape，为保险显式 reshape 成 [I,K,M]
controller.w = reshape(Wc_initial, controller.I, controller.K, controller.M);

Er = zeros(Num_samples, 1);
for i = 1:Num_samples
    xcurrent = simulation.x(i,:);   % [1,K] 行向量，iterate内部会转置
    ecurrent = simulation.e;        % [L,1] 上一时刻误差
    % [FIX] 先记录误差再仿真，与FxLMS中 Er(tt)=e 在 y 输出之后、w 更新之前的时序一致
    Er(i) = ecurrent;
    ucurrent = controller.iterate(xcurrent, ecurrent);
    simulation.sim(ucurrent);
end

%% Runging the FxLMS with the MAML-initialization control filter
Wc_initial_maml = Wc;

simulation_maml = SimulationPlatform(Rf_1, Dis_1, S11);
controller_maml = Controller(muw, 1, 1, 1);
controller_maml.s = simulation_maml.s;
% [FIX] 同上，显式 reshape 保证维度正确
controller_maml.w = reshape(Wc_initial_maml, controller_maml.I, controller_maml.K, controller_maml.M);

Er1 = zeros(Num_samples, 1);
for i = 1:Num_samples
    xcurrent1 = simulation_maml.x(i,:);
    ecurrent1 = simulation_maml.e;
    % [FIX] 先记录误差再仿真
    Er1(i) = ecurrent1;
    ucurrent1 = controller_maml.iterate(xcurrent1, ecurrent1);
    simulation_maml.sim(ucurrent1);
end

%% Runging the FxLMS with the random-initialization control filter
rng(0,'twister');
Wc_initial_random = randn(Len_N,1);
Wc_maml_norm = norm(Wc_initial_maml);
if Wc_maml_norm > 0
    Wc_initial_random = Wc_initial_random / norm(Wc_initial_random) * Wc_maml_norm;
end

simulation_random = SimulationPlatform(Rf_1, Dis_1, S11);
controller_random = Controller(muw, 1, 1, 1);
controller_random.s = simulation_random.s;
controller_random.w = reshape(Wc_initial_random, controller_random.I, controller_random.K, controller_random.M);

Er2 = zeros(Num_samples, 1);
for i = 1:Num_samples
    xcurrent2 = simulation_random.x(i,:);
    ecurrent2 = simulation_random.e;
    Er2(i) = ecurrent2;
    ucurrent2 = controller_random.iterate(xcurrent2, ecurrent2);
    simulation_random.sim(ucurrent2);
end

%% Drawing the figures of the MAML and FxLMS 
figure
t_axis = (0:Num_samples-1) * (1/fs);
plot(t_axis, Dis_1(1:Num_samples), ...
     t_axis, Er2, ...
     t_axis, Er,  ...
     t_axis, Er1);
set(gca, 'FontName', 'Times New Roman');
title('Road Noise Cancellation', 'FontSize', 18, 'FontName', 'Times New Roman')
xlim([0 5])
xlabel('Time (seconds)', 'FontSize', 15, 'FontName', 'Times New Roman')
ylabel('Error signal', 'FontSize', 15, 'FontName', 'Times New Roman')
legend({'ANC off','Random initialization','Zero initialization','MAML initialization'}, 'FontSize', 15, 'FontName', 'Times New Roman')
grid on;

%% Time-frequency comparison for the first 5 seconds
tf_len = min(round(5*fs), Num_samples);
tf_signals = {Dis_1(1:tf_len), Er2(1:tf_len), Er(1:tf_len), Er1(1:tf_len)};
tf_titles = {'ANC off','Random-initialization','Zero-initialization','MAML-initialization'};
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
    tf_clim(1) = min(tf_clim(1), min(tf_power_db{ii}(:)));
    tf_clim(2) = max(tf_clim(2), max(tf_power_db{ii}(:)));
end

figure
for ii = 1:4
    subplot(2,2,ii);
    imagesc(tf_time{ii}, tf_freq{ii}/1000, tf_power_db{ii});
    axis xy;
    set(gca, 'FontName', 'Times New Roman');
    title([tf_titles{ii}, ' - first 5 s'], 'FontSize', 18, 'FontName', 'Times New Roman');
    xlabel('Time (seconds)', 'FontSize', 15, 'FontName', 'Times New Roman');
    ylabel('Frequency (kHz)', 'FontSize', 15, 'FontName', 'Times New Roman');
    ylim([0 1]);
    caxis(tf_clim);
    cb = colorbar;
    set(cb, 'FontName', 'Times New Roman');
end

%<<===Progress bar===>> 
waitbar(1,f,'Finishing');
pause(1)
close(f)
%-------------------------------end----------------------------------------
