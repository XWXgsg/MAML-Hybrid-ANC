%% Title: Variable-Frequency Noise Generator for ANC Simulation
%  生成模拟变频率噪声信号（如发动机升降速工况）
%  
%  功能:
%    1. 生成基频随时间变化的周期性噪声（模拟转速变化）
%    2. 自动叠加多阶谐波
%    3. 添加宽带背景噪声
%    4. 可选：通过次级路径模型生成多通道误差信号
%
%  Author: Zhicheng Zhang
%  Date  : 2026-2-26
%--------------------------------------------------------------------------
clc; clear; close all;

%% =================== 参数设置 ===================

fs = 2000;              % 采样率 (Hz)
T  = 10;                % 信号总时长 (s)
N  = fs * T;            % 总采样点数
t  = (0:N-1)' / fs;     % 时间向量

L  = 2;                 % 通道数 (误差麦克风数量)

%% =================== 基频曲线定义 ===================
% 定义基频随时间的变化规律，模拟不同工况
% 可选模式: 'sweep_linear', 'sweep_sine', 'step', 'accel_decel', 'custom'

mode = 'accel_decel';

switch mode
    case 'sweep_linear'
        % --- 线性扫频 ---
        f0_start = 30;      % 起始基频 (Hz)
        f0_end   = 120;     % 终止基频 (Hz)
        f0 = linspace(f0_start, f0_end, N)';

    case 'sweep_sine'
        % --- 正弦调频 (模拟周期性转速波动) ---
        f0_center = 80;     % 中心频率 (Hz)
        f0_amp    = 40;     % 频率波动幅度 (Hz)
        f_mod     = 0.2;    % 调制频率 (Hz), 即转速波动周期
        f0 = f0_center + f0_amp * sin(2*pi*f_mod*t);

    case 'step'
        % --- 阶梯变频 (模拟离散转速工况) ---
        f_steps = [40, 60, 80, 100, 120];           % 各段频率
        step_dur = T / length(f_steps);              % 每段持续时间
        f0 = zeros(N, 1);
        for idx = 1:length(f_steps)
            n_start = round((idx-1) * step_dur * fs) + 1;
            n_end   = min(round(idx * step_dur * fs), N);
            f0(n_start:n_end) = f_steps(idx);
        end

    case 'accel_decel'
        % --- 加速-匀速-减速 (典型行驶工况) ---
        % 分三段: 加速 0~3s, 匀速 3~7s, 减速 7~10s
        f0 = zeros(N, 1);
        t1 = 3; t2 = 7;  % 分段时间点
        f_low = 30; f_high = 110;
        for i = 1:N
            if t(i) < t1
                % 加速段: 线性升频
                f0(i) = f_low + (f_high - f_low) * (t(i) / t1);
            elseif t(i) < t2
                % 匀速段: 恒定频率 + 小幅随机抖动
                f0(i) = f_high + 2 * randn;
            else
                % 减速段: 线性降频
                f0(i) = f_high - (f_high - f_low) * ((t(i) - t2) / (T - t2));
            end
        end
        % 平滑处理，去除抖动造成的突变
        f0 = smoothdata(f0, 'movmean', round(fs * 0.05));

    case 'custom'
        % --- 自定义: 用插值定义任意曲线 ---
        t_knots = [0, 1, 3, 5, 7, 9, 10];          % 时间节点 (s)
        f_knots = [40, 60, 100, 100, 80, 50, 50];   % 对应频率 (Hz)
        f0 = interp1(t_knots, f_knots, t, 'pchip');

    otherwise
        error('未知模式: %s', mode);
end

%% =================== 谐波信号生成 ===================

harmonics    = [1, 2, 3, 4, 5];                % 谐波阶次 (1=基频)
harmonic_amp = [1.0, 0.6, 0.35, 0.2, 0.1];    % 各阶幅值 (相对)
harmonic_phase_offset = [0, 0.3, 0.8, 1.2, 2.0]; % 各阶初始相位偏移

% 瞬时相位累积 (对瞬时频率积分)
phase_base = cumtrapz(t, 2*pi*f0);

% 合成各阶谐波
signal_clean = zeros(N, 1);
for h = 1:length(harmonics)
    order = harmonics(h);
    amp   = harmonic_amp(h);
    phi0  = harmonic_phase_offset(h);
    signal_clean = signal_clean + amp * sin(order * phase_base + phi0);
end

%% =================== 添加噪声 ===================

% 宽带高斯白噪声
SNR_dB = 20;                                    % 信噪比 (dB)
noise_power = rms(signal_clean)^2 / (10^(SNR_dB/10));
wideband_noise = sqrt(noise_power) * randn(N, 1);

% 合成单通道含噪信号
signal = signal_clean + wideband_noise;

%% =================== 多通道生成 ===================
% 模拟 L 个误差麦克风，各通道有不同增益和延迟

channel_gains  = [1.0, 0.8];        % 各通道增益
channel_delays = [0, 5];            % 各通道延迟 (采样点)

dn = zeros(N, L);
for ch = 1:L
    delayed = circshift(signal, channel_delays(ch));
    % 清除循环移位引入的首端伪数据
    if channel_delays(ch) > 0
        delayed(1:channel_delays(ch)) = 0;
    end
    dn(:, ch) = channel_gains(ch) * delayed + 0.02 * randn(N, 1);
end

%% =================== 可视化 ===================

figure('Name', 'Variable-Frequency Noise Generator', ...
       'Position', [100, 100, 1200, 800], 'Color', 'w');

% --- 1. 基频曲线 ---
subplot(3, 2, 1);
plot(t, f0, 'b', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('基频变化曲线 (Fundamental Frequency)');
grid on; xlim([0, T]);

% --- 2. 时域波形 ---
subplot(3, 2, 2);
plot(t, dn(:,1), 'Color', [0.2, 0.4, 0.8]);
xlabel('Time (s)'); ylabel('Amplitude');
title('通道 1 时域波形');
grid on; xlim([0, T]);

% --- 3. 频谱图 (Spectrogram) ---
subplot(3, 2, [3, 4]);
nfft_spec = 512;
noverlap  = round(nfft_spec * 0.9);
spectrogram(dn(:,1), hann(nfft_spec), noverlap, nfft_spec, fs, 'yaxis');
title('短时傅里叶变换频谱图 (STFT Spectrogram)');
colormap('jet');
clim_val = max(abs(get(gca,'CLim')));
caxis([-clim_val, clim_val] * 0.6);
ylim([0, min(500, fs/2)]);
colorbar;
hold on;
% 在频谱图上叠加理论基频及谐波线
for h = 1:length(harmonics)
    plot(t, harmonics(h) * f0 / 1000, 'w--', 'LineWidth', 1);
end
hold off;

% --- 4. 功率谱密度 (整段) ---
subplot(3, 2, 5);
nfft_psd = 2048;
[pxx, f_psd] = pwelch(dn(:,1), hann(nfft_psd), nfft_psd/2, nfft_psd, fs);
plot(f_psd, 10*log10(pxx), 'b', 'LineWidth', 1);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
title('功率谱密度 (Welch PSD)');
grid on; xlim([0, 500]);

% --- 5. 双通道对比 ---
subplot(3, 2, 6);
plot(t, dn(:,1), 'b'); hold on;
plot(t, dn(:,2), 'r'); hold off;
xlabel('Time (s)'); ylabel('Amplitude');
title('双通道时域对比');
legend('Ch 1', 'Ch 2'); grid on; xlim([0, T]);

sgtitle(sprintf('模拟变频率噪声 [模式: %s, fs=%d Hz, T=%d s]', mode, fs, T), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% =================== 保存数据 ===================
% 保存为 .mat 文件，供 ANC 仿真主程序使用

save('sim_noise.mat', 'dn', 'f0', 'fs', 'T', 'N', 't', 'harmonics', 'mode');
fprintf('数据已保存至 sim_noise.mat\n');
fprintf('  采样率: %d Hz\n', fs);
fprintf('  时长:   %d s (%d 点)\n', T, N);
fprintf('  通道数: %d\n', L);
fprintf('  基频范围: %.1f ~ %.1f Hz\n', min(f0), max(f0));
fprintf('  谐波阶次: %s\n', num2str(harmonics));
fprintf('  模式: %s\n', mode);
