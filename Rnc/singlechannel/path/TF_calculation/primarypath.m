close all; clc; clear;
load Project1_Section1_20251210_headLocation_02_speaker_f_01.hdf.mat

fs_ori = floor(1/Signal_1.x_values.increment);
fs = 16000;
sensitivity1 = 0.0519409;   % CH6 误差麦克风灵敏度
sensitivity2 = 0.0493182;   % CH7 误差麦克风灵敏度

%% 提取原始单段数据
idx = 1000:400000;
un_raw  = Signal_0.y_values.values(idx, 1);                   % Signal0 CH1 噪声源
dn1_raw = sensitivity1 * Signal_1.y_values.values(idx, 6);    % CH6 误差麦克风1
dn2_raw = sensitivity2 * Signal_1.y_values.values(idx, 7);    % CH7 误差麦克风2

% 降采样
un_seg  = resample(un_raw,  fs, fs_ori);
dn1_seg = resample(dn1_raw, fs, fs_ori);
dn2_seg = resample(dn2_raw, fs, fs_ori);

%% 多次循环拼接
N_repeat = 1;
un  = repmat(un_seg,  N_repeat, 1);
dn1 = repmat(dn1_seg, N_repeat, 1);
dn2 = repmat(dn2_seg, N_repeat, 1);

fprintf('总训练样本数: %d (约 %.1f 分钟)\n', length(un), length(un)/fs/60);

%% NLMS 参数
M    = 256;
mu   = 0.2;
leak = 0;

%% 辨识主级路径
fprintf('Identifying Primary Path 1 (Signal0->CH6)...\n');
S1 = NLMSinit(zeros(M,1), mu, leak);
[yn1, en1, S1, P1] = NLMSadapt(un, dn1, S1);

fprintf('Identifying Primary Path 2 (Signal0->CH7)...\n');
S2 = NLMSinit(zeros(M,1), mu, leak);
[yn2, en2, S2, P2] = NLMSadapt(un, dn2, S2);

%% 收敛分析
window_size = 1000;
n_seg = floor(length(en1)/window_size);
seg_err1 = zeros(n_seg,1);
seg_err2 = zeros(n_seg,1);
for k = 1:n_seg
    seg = (k-1)*window_size+1 : k*window_size;
    seg_err1(k) = mean(en1(seg).^2);
    seg_err2(k) = mean(en2(seg).^2);
end
seg_t = (1:n_seg)*window_size/fs;

loop_len = length(un_seg);
loop_boundaries = (1:N_repeat-1) * loop_len / fs;

gain1 = 10*log10(seg_err1(1)+1e-12) - 10*log10(seg_err1+1e-12);
gain2 = 10*log10(seg_err2(1)+1e-12) - 10*log10(seg_err2+1e-12);

fprintf('\n===== 收敛统计 =====\n');
fprintf('Path1 (Signal0->CH6): %.2f dB -> %.2f dB，共减小 %.2f dB\n', ...
    10*log10(seg_err1(1)+1e-12), 10*log10(seg_err1(end)+1e-12), gain1(end));
fprintf('Path2 (Signal0->CH7): %.2f dB -> %.2f dB，共减小 %.2f dB\n', ...
    10*log10(seg_err2(1)+1e-12), 10*log10(seg_err2(end)+1e-12), gain2(end));

%% 画图
figure('Name','Primary Path','Position',[100 100 1400 900]);

subplot(3,4,1);
plot(seg_t, 10*log10(seg_err1+1e-12), 'b-', 'LineWidth', 1.5); hold on;
for b = loop_boundaries, xline(b,'r--','LineWidth',1,'Label','Loop'); end
title('Path1(S0->CH6) 分段平均误差'); xlabel('Time(s)'); ylabel('dB'); grid on;

subplot(3,4,2);
plot(seg_t, 10*log10(seg_err2+1e-12), 'r-', 'LineWidth', 1.5); hold on;
for b = loop_boundaries, xline(b,'r--','LineWidth',1,'Label','Loop'); end
title('Path2(S0->CH7) 分段平均误差'); xlabel('Time(s)'); ylabel('dB'); grid on;

subplot(3,4,3);
plot(seg_t, gain1, 'b-', 'LineWidth', 1.5); hold on;
for b = loop_boundaries, xline(b,'r--'); end
title('Path1 误差减小量'); xlabel('Time(s)'); ylabel('dB'); grid on;

subplot(3,4,4);
plot(seg_t, gain2, 'r-', 'LineWidth', 1.5); hold on;
for b = loop_boundaries, xline(b,'r--'); end
title('Path2 误差减小量'); xlabel('Time(s)'); ylabel('dB'); grid on;

subplot(3,4,5);
t_ax = (1:length(en1))/fs;
plot(t_ax, dn1, 'b', 'LineWidth', 0.3); hold on;
plot(t_ax, en1, 'r', 'LineWidth', 0.3);
title('Path1: dn1(蓝) vs en1(红)'); xlabel('Time(s)'); grid on;
legend('dn1','en1');

subplot(3,4,6);
t_ax = (1:length(en2))/fs;
plot(t_ax, dn2, 'b', 'LineWidth', 0.3); hold on;
plot(t_ax, en2, 'r', 'LineWidth', 0.3);
title('Path2: dn2(蓝) vs en2(红)'); xlabel('Time(s)'); grid on;
legend('dn2','en2');

subplot(3,4,7);
loop_err1 = seg_err1(round((1:N_repeat)*loop_len/fs/window_size*fs));
plot(1:N_repeat, 10*log10(loop_err1+1e-12), 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
title('Path1 每轮结束误差'); xlabel('Loop #'); ylabel('dB'); grid on;

subplot(3,4,8);
loop_err2 = seg_err2(round((1:N_repeat)*loop_len/fs/window_size*fs));
plot(1:N_repeat, 10*log10(loop_err2+1e-12), 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
title('Path2 每轮结束误差'); xlabel('Loop #'); ylabel('dB'); grid on;

subplot(3,4,9);
stem(P1, 'filled', 'MarkerSize', 2);
title('Path1(S0->CH6) 冲激响应'); xlabel('Tap'); ylabel('Amplitude'); grid on;

subplot(3,4,10);
stem(P2, 'filled', 'MarkerSize', 2);
title('Path2(S0->CH7) 冲激响应'); xlabel('Tap'); ylabel('Amplitude'); grid on;

subplot(3,4,11);
[H1,f1] = freqz(P1, 1, 1024, fs);
plot(f1, 20*log10(abs(H1)+1e-12));
title('Path1(S0->CH6) 频率响应'); xlabel('Hz'); ylabel('dB'); grid on;

subplot(3,4,12);
[H2,f2] = freqz(P2, 1, 1024, fs);
plot(f2, 20*log10(abs(H2)+1e-12));
title('Path2(S0->CH7) 频率响应'); xlabel('Hz'); ylabel('dB'); grid on;

%% 保存
save('PrimaryPath_256order.mat', 'P1', 'P2');
fprintf('已保存 PrimaryPath_256order.mat\n');