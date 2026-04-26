%% Title: Single-Tone Noise Generator for ANC Simulation
%  生成单频正弦噪声信号，用于陷波器 ANC 算法验证
%
%  Author: Zhicheng Zhang
%  Date  : 2026-2-26
%--------------------------------------------------------------------------
clc; clear; close all;

%% =================== 参数设置 ===================

fs = 2000;              % 采样率 (Hz)
T  = 10;                % 信号时长 (s)
N  = fs * T;            % 总采样点数
t  = (0:N-1)' / fs;     % 时间向量

L  = 2;                 % 通道数 (误差麦克风数量)
M  = 2;                 % 控制扬声器数量

%% =================== 单频信号参数 ===================

f0  = 100;              % 目标频率 (Hz)
A0  = 1.0;              % 幅值
phi = 0;                % 初始相位 (rad)

%% =================== 信号生成 ===================

% 纯净单频信号
signal_clean = A0 * sin(2*pi*f0*t + phi);

% 添加少量宽带背景噪声
SNR_dB = 30;                                             % 信噪比 (dB)
noise_power = rms(signal_clean)^2 / (10^(SNR_dB/10));
bg_noise = sqrt(noise_power) * randn(N, 1);

signal = signal_clean + bg_noise;

%% =================== 多通道生成 ===================

channel_gains  = [1.0, 0.85];       % 各通道增益
channel_delays = [0, 3];            % 各通道延迟 (采样点)

dn = zeros(N, L);
for ch = 1:L
    delayed = circshift(signal, channel_delays(ch));
    if channel_delays(ch) > 0
        delayed(1:channel_delays(ch)) = 0;
    end
    dn(:, ch) = channel_gains(ch) * delayed;
end

%% =================== 次级路径模型生成 ===================

J = 128;                % 次级路径滤波器阶数

% 生成简化的次级路径脉冲响应 (指数衰减 + 延迟)
s = zeros(J, M, L);
delay_s = 10;           % 次级路径延迟 (采样点)
for m = 1:M
    for l = 1:L
        imp = zeros(J, 1);
        imp(delay_s) = 0.8 + 0.1 * randn;
        % 加一些衰减尾部模拟房间反射
        for j = delay_s+1:J
            imp(j) = 0.02 * exp(-(j - delay_s) / 20) * randn;
        end
        s(:, m, l) = imp;
    end
end

%% =================== 可视化 ===================

figure('Name', 'Single-Tone Noise', 'Position', [150, 150, 1000, 600], 'Color', 'w');

% 时域
subplot(2, 2, 1);
plot(t, dn(:,1), 'b');
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('时域波形 (f_0 = %d Hz)', f0));
grid on; xlim([0, 0.1]);  % 显示前 100ms

% 全段时域
subplot(2, 2, 2);
plot(t, dn(:,1), 'b'); hold on;
plot(t, dn(:,2), 'r'); hold off;
xlabel('Time (s)'); ylabel('Amplitude');
title('双通道全段波形');
legend('Ch 1', 'Ch 2'); grid on; xlim([0, T]);

% PSD
subplot(2, 2, 3);
nfft = 2048;
[pxx, f_psd] = pwelch(dn(:,1), hann(nfft), nfft/2, nfft, fs);
plot(f_psd, 10*log10(pxx), 'b', 'LineWidth', 1.2);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
title('功率谱密度');
grid on; xlim([0, 500]);
xline(f0, 'r--', sprintf('%d Hz', f0), 'LineWidth', 1.2, 'LabelOrientation', 'horizontal');

% 次级路径
subplot(2, 2, 4);
stem((0:J-1)/fs*1000, s(:,1,1), 'b', 'MarkerSize', 3);
xlabel('Time (ms)'); ylabel('Amplitude');
title('次级路径脉冲响应 S_{1,1}');
grid on;

sgtitle(sprintf('单频噪声信号: f_0 = %d Hz, fs = %d Hz, SNR = %d dB', f0, fs, SNR_dB), ...
    'FontSize', 13, 'FontWeight', 'bold');

%% =================== 保存 ===================

save('sim_single_tone.mat', 'dn', 's', 'fs', 'T', 'N', 'f0', 'M', 'L', 'J');
fprintf('已保存 sim_single_tone.mat\n');
fprintf('  频率: %d Hz | 采样率: %d Hz | 时长: %d s | SNR: %d dB\n', f0, fs, T, SNR_dB);
