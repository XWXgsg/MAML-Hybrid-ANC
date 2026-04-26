close all; clc; clear;

%% ===== 参数设置 =====
fs    = 16000;
M     = 256;
mu    = 0.3;
leak  = 0;
N_repeat = 5;
sensitivity6 = 0.0519409;
sensitivity7  = 0.0493182;
idx = 1000:400000;

%% ===== 加载扬声器1数据 =====
fprintf('加载扬声器1数据...\n');
load('Project2_Section1_20251210_headLocation_02_speaker_l_01.hdf.mat');
fs_ori = floor(1/Signal_1.x_values.increment);

un1   = resample(Signal_0.y_values.values(idx,1), fs, fs_ori);
dn6_1 = resample(sensitivity6 * Signal_1.y_values.values(idx,6), fs, fs_ori);
dn7_1 = resample(sensitivity7 * Signal_1.y_values.values(idx,7), fs, fs_ori);

L1    = min([length(un1), length(dn6_1), length(dn7_1)]);
un1   = repmat(un1(1:L1),   N_repeat, 1);
dn6_1 = repmat(dn6_1(1:L1), N_repeat, 1);
dn7_1 = repmat(dn7_1(1:L1), N_repeat, 1);
fprintf('Spk1 训练样本: %d (%.1f min)\n', length(un1), length(un1)/fs/60);

%% 辨识 S11, S12（扬声器1 -> 麦克风1, 麦克风2）
% S11: 扬声器1 -> 麦克风1(CH6)
% S12: 扬声器1 -> 麦克风2(CH7)
fprintf('辨识 S11 (Spk1->CH6)...\n');
[~, en_s11, ~, S11] = NLMSadapt(un1, dn6_1, NLMSinit(zeros(M,1),mu,leak));

fprintf('辨识 S12 (Spk1->CH7)...\n');
[~, en_s12, ~, S12] = NLMSadapt(un1, dn7_1, NLMSinit(zeros(M,1),mu,leak));

%% ===== 加载扬声器2数据 =====
fprintf('\n加载扬声器2数据...\n');
clear Signal_0 Signal_1
load('Project2_Section1_20251210_headLocation_02_speaker_r_01.hdf.mat');
fs_ori2 = floor(1/Signal_1.x_values.increment);

un2   = resample(Signal_0.y_values.values(idx,1), fs, fs_ori2);
dn6_2 = resample(sensitivity6 * Signal_1.y_values.values(idx,6), fs, fs_ori2);
dn7_2 = resample(sensitivity7 * Signal_1.y_values.values(idx,7), fs, fs_ori2);

L2    = min([length(un2), length(dn6_2), length(dn7_2)]);
un2   = repmat(un2(1:L2),   N_repeat, 1);
dn6_2 = repmat(dn6_2(1:L2), N_repeat, 1);
dn7_2 = repmat(dn7_2(1:L2), N_repeat, 1);
fprintf('Spk2 训练样本: %d (%.1f min)\n', length(un2), length(un2)/fs/60);

%% 辨识 S21, S22（扬声器2 -> 麦克风1, 麦克风2）
% S21: 扬声器2 -> 麦克风1(CH6)
% S22: 扬声器2 -> 麦克风2(CH7)
fprintf('辨识 S21 (Spk2->CH6)...\n');
[~, en_s21, ~, S21] = NLMSadapt(un2, dn6_2, NLMSinit(zeros(M,1),mu,leak));

fprintf('辨识 S22 (Spk2->CH7)...\n');
[~, en_s22, ~, S22] = NLMSadapt(un2, dn7_2, NLMSinit(zeros(M,1),mu,leak));

%% ===== 收敛分析 & 画图 =====
% 矩阵排列:
%        Spk1   Spk2
%  CH6 [ S11    S21 ]
%  CH7 [ S12    S22 ]
window_size = 1000;
all_en  = {en_s11, en_s12, en_s21, en_s22};
all_W   = {S11,    S12,    S21,    S22};
labels  = {'S11 Spk1->CH6', 'S12 Spk1->CH7', ...
           'S21 Spk2->CH6', 'S22 Spk2->CH7'};
colors  = {'b','r','g','m'};

fprintf('\n===== 收敛统计 =====\n');
figure('Name','Secondary Path 2x2','Position',[100 100 1400 1000]);

for pp = 1:4
    en    = all_en{pp};
    W     = all_W{pp};
    n_seg = floor(length(en)/window_size);
    seg_err = zeros(n_seg,1);
    for k = 1:n_seg
        seg = (k-1)*window_size+1 : k*window_size;
        seg_err(k) = mean(en(seg).^2);
    end
    seg_t   = (1:n_seg)*window_size/fs;
    improve = 10*log10(seg_err(1)+1e-12) - 10*log10(seg_err(end)+1e-12);

    fprintf('%s: %.2f dB -> %.2f dB，改善 %.2f dB\n', labels{pp}, ...
        10*log10(seg_err(1)+1e-12), 10*log10(seg_err(end)+1e-12), improve);

    subplot(4,3,(pp-1)*3+1);
    plot(seg_t, 10*log10(seg_err+1e-12), colors{pp}, 'LineWidth',1.5);
    title([labels{pp} ' 收敛曲线']);
    xlabel('Time(s)'); ylabel('dB'); grid on;

    subplot(4,3,(pp-1)*3+2);
    [~,pk] = max(abs(W));
    stem(W,'filled','MarkerSize',2);
    title(sprintf('%s 冲激响应\n主峰tap%d=%.2fms', labels{pp}, pk, pk/fs*1000));
    xlabel('Tap'); ylabel('Amplitude'); grid on;

    subplot(4,3,(pp-1)*3+3);
    [H,f] = freqz(W, 1, 1024, fs);
    plot(f, 20*log10(abs(H)+1e-12), colors{pp}, 'LineWidth',1.5);
    title([labels{pp} ' 频率响应']);
    xlabel('Hz'); ylabel('dB'); grid on;
end

%% ===== 保存 =====
save('SecondaryPath_2x2_256order.mat', 'S11','S12','S21','S22');
fprintf('\n已保存 SecondaryPath_2x2_256order.mat\n');
fprintf('矩阵定义:\n');
fprintf('  S11: Spk1->CH6(麦克风1)\n');
fprintf('  S12: Spk1->CH7(麦克风2)\n');
fprintf('  S21: Spk2->CH6(麦克风1)\n');
fprintf('  S22: Spk2->CH7(麦克风2)\n');