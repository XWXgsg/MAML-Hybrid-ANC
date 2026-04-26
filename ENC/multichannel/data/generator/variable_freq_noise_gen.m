%% Title: Variable-Frequency Noise Generator for ANC Simulation (Extended)
%  生成模拟变频率噪声信号 — 长时间、宽频段、随机变化版本
%  
%  修改说明:
%    1. 信号时长大幅延长 (默认60s，可调)
%    2. 新增 'random_walk' 模式：基频在宽频段内随机游走
%    3. 新增 'random_segments' 模式：随机分段，每段不同频率特征
%    4. 新增 'multi_event' 模式：模拟多次随机加减速事件
%    5. 频率覆盖范围扩展至 20~300 Hz
%
%  Author: Zhicheng Zhang (modified)
%  Date  : 2026-2-26
%--------------------------------------------------------------------------
clc; clear; close all;

%% =================== 参数设置 ===================

fs = 2000;              % 采样率 (Hz)
T  = 60;                % 信号总时长 (s)  ← 延长至60s
N  = fs * T;            % 总采样点数
t  = (0:N-1)' / fs;     % 时间向量

L  = 2;                 % 通道数 (误差麦克风数量)

% 频率范围 (扩大覆盖)
f_min = 20;             % 最低基频 (Hz)
f_max = 300;            % 最高基频 (Hz)

%% =================== 基频曲线定义 ===================
% 可选模式: 'sweep_linear', 'sweep_sine', 'step', 'accel_decel', 'custom',
%           'random_walk', 'random_segments', 'multi_event'

mode = 'random_walk';

switch mode
    case 'sweep_linear'
        % --- 线性扫频 (全频段) ---
        f0 = linspace(f_min, f_max, N)';

    case 'sweep_sine'
        % --- 正弦调频 (多周期波动) ---
        f0_center = (f_min + f_max) / 2;
        f0_amp    = (f_max - f_min) / 2 * 0.8;
        f_mod     = 0.1;    % 慢速调制
        f0 = f0_center + f0_amp * sin(2*pi*f_mod*t);

    case 'step'
        % --- 阶梯变频 (覆盖更多频率段) ---
        f_steps = linspace(f_min, f_max, 12);    % 12个频率台阶
        f_steps = f_steps(randperm(length(f_steps))); % 随机打乱顺序
        step_dur = T / length(f_steps);
        f0 = zeros(N, 1);
        for idx = 1:length(f_steps)
            n_start = round((idx-1) * step_dur * fs) + 1;
            n_end   = min(round(idx * step_dur * fs), N);
            f0(n_start:n_end) = f_steps(idx);
        end
        % 台阶间平滑过渡
        f0 = smoothdata(f0, 'gaussian', round(fs * 0.3));

    case 'accel_decel'
        % --- 多次加速-匀速-减速循环 ---
        n_cycles = 4;
        T_cycle = T / n_cycles;
        f0 = zeros(N, 1);
        for c = 1:n_cycles
            f_low  = f_min + (f_max - f_min) * 0.1 * rand;
            f_high = f_max - (f_max - f_min) * 0.1 * rand;
            t_off  = (c - 1) * T_cycle;
            t1 = t_off + T_cycle * 0.3;
            t2 = t_off + T_cycle * 0.7;
            for i = 1:N
                if t(i) >= t_off && t(i) < t1
                    f0(i) = f_low + (f_high - f_low) * ((t(i) - t_off) / (t1 - t_off));
                elseif t(i) >= t1 && t(i) < t2
                    f0(i) = f_high + 3 * randn;
                elseif t(i) >= t2 && t(i) < t_off + T_cycle
                    f0(i) = f_high - (f_high - f_low) * ((t(i) - t2) / (t_off + T_cycle - t2));
                end
            end
        end
        f0 = smoothdata(f0, 'movmean', round(fs * 0.05));

    case 'random_walk'
        % --- 随机游走 (宽频段内自然漫游) ---
        % 基频以布朗运动方式在 [f_min, f_max] 间随机变化
        % 通过调节 drift_std 控制变化速率
        
        drift_std = 15;          % 每秒频率变化标准差 (Hz/s)
        f0 = zeros(N, 1);
        f0(1) = f_min + (f_max - f_min) * rand;  % 随机起始频率
        
        dt = 1 / fs;
        for i = 2:N
            % 随机增量 + 微弱回中力 (防止长时间卡在边界)
            center = (f_min + f_max) / 2;
            revert_force = -0.5 * (f0(i-1) - center) / (f_max - f_min) * drift_std;
            f0(i) = f0(i-1) + (drift_std * randn + revert_force) * sqrt(dt);
        end
        
        % 硬限幅 + 平滑
        f0 = max(f_min, min(f_max, f0));
        f0 = smoothdata(f0, 'gaussian', round(fs * 0.1));

    case 'random_segments'
        % --- 随机分段模式 ---
        % 将总时长随机切分为若干段，每段有不同的频率行为
        % (恒频、线性扫频、正弦调频等随机组合)
        
        seg_min_dur = 2.0;     % 最短段时长 (s)
        seg_max_dur = 8.0;     % 最长段时长 (s)
        
        f0 = zeros(N, 1);
        pos = 1;               % 当前写入位置
        
        while pos <= N
            % 随机段长
            seg_dur = seg_min_dur + (seg_max_dur - seg_min_dur) * rand;
            seg_len = round(seg_dur * fs);
            seg_end = min(pos + seg_len - 1, N);
            seg_n   = seg_end - pos + 1;
            t_seg   = (0:seg_n-1)' / fs;
            
            % 随机选择本段频率行为
            behavior = randi(4);
            f_start = f_min + (f_max - f_min) * rand;
            f_end_  = f_min + (f_max - f_min) * rand;
            
            switch behavior
                case 1  % 恒频 (带微小抖动)
                    f_seg = f_start + 1.5 * randn(seg_n, 1);
                case 2  % 线性扫频
                    f_seg = linspace(f_start, f_end_, seg_n)';
                case 3  % 正弦调频
                    f_center = (f_start + f_end_) / 2;
                    f_amp = abs(f_start - f_end_) / 2;
                    f_mod_local = 0.3 + 0.7 * rand;
                    f_seg = f_center + f_amp * sin(2*pi*f_mod_local*t_seg);
                case 4  % 指数变化 (快速升/降频)
                    tau = seg_dur * (0.2 + 0.6 * rand);
                    if rand > 0.5
                        f_seg = f_start + (f_end_ - f_start) * (1 - exp(-t_seg / tau));
                    else
                        f_seg = f_end_ + (f_start - f_end_) * exp(-t_seg / tau);
                    end
            end
            
            f0(pos:seg_end) = f_seg;
            pos = seg_end + 1;
        end
        
        % 限幅 + 段间平滑
        f0 = max(f_min, min(f_max, f0));
        f0 = smoothdata(f0, 'gaussian', round(fs * 0.15));

    case 'multi_event'
        % --- 多事件模式 ---
        % 模拟多次独立的加减速事件，事件间有随机怠速段
        
        n_events = randi([5, 10]);       % 随机事件数
        event_times = sort(rand(n_events, 1) * T * 0.9);  % 事件起始时间
        event_durs  = 1.5 + 3.5 * rand(n_events, 1);      % 事件持续时间
        
        f0 = ones(N, 1) * (f_min + 10);  % 背景怠速频率
        
        for ev = 1:n_events
            ev_start = round(event_times(ev) * fs) + 1;
            ev_len   = round(event_durs(ev) * fs);
            ev_end   = min(ev_start + ev_len - 1, N);
            ev_n     = ev_end - ev_start + 1;
            t_ev     = linspace(0, 1, ev_n)';
            
            f_peak = f_min + (f_max - f_min) * (0.3 + 0.7 * rand);
            
            % 钟形包络: 快速上升，缓慢下降
            rise_rate = 2 + 4 * rand;
            envelope = (t_ev .^ (1/rise_rate)) .* ((1 - t_ev) .^ 0.5);
            envelope = envelope / max(envelope);
            
            f_event = f0(ev_start:ev_end) + (f_peak - f0(ev_start:ev_end)) .* envelope;
            f0(ev_start:ev_end) = max(f0(ev_start:ev_end), f_event);
        end
        
        % 全局平滑
        f0 = smoothdata(f0, 'gaussian', round(fs * 0.08));
        f0 = max(f_min, min(f_max, f0));

    case 'custom'
        % --- 自定义: 用插值定义任意曲线 ---
        t_knots = linspace(0, T, 15);
        f_knots = f_min + (f_max - f_min) * rand(1, 15);
        f0 = interp1(t_knots, f_knots, t, 'pchip');

    otherwise
        error('未知模式: %s', mode);
end

%% =================== 谐波信号生成 ===================

harmonics    = [1, 2, 3, 4, 5, 6];                      % 谐波阶次 (增加到6阶)
harmonic_amp = [1.0, 0.6, 0.35, 0.2, 0.1, 0.05];       % 各阶幅值
harmonic_phase_offset = [0, 0.3, 0.8, 1.2, 2.0, 2.8];  % 各阶初始相位偏移

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
SNR_dB = 20;
noise_power = rms(signal_clean)^2 / (10^(SNR_dB/10));
wideband_noise = sqrt(noise_power) * randn(N, 1);

% 添加有色噪声 (低频隆隆声)
[b_lp, a_lp] = butter(4, 100 / (fs/2), 'low');
colored_noise = filter(b_lp, a_lp, randn(N, 1)) * sqrt(noise_power) * 0.5;

% 合成单通道含噪信号
signal = signal_clean + wideband_noise + colored_noise;

%% =================== 多通道生成 ===================

channel_gains  = [1.0, 0.8];
channel_delays = [0, 5];

dn = zeros(N, L);
for ch = 1:L
    delayed = circshift(signal, channel_delays(ch));
    if channel_delays(ch) > 0
        delayed(1:channel_delays(ch)) = 0;
    end
    dn(:, ch) = channel_gains(ch) * delayed + 0.02 * randn(N, 1);
end

%% =================== 可视化 ===================

figure('Name', 'Variable-Frequency Noise Generator (Extended)', ...
       'Position', [50, 50, 1400, 900], 'Color', 'w');

% --- 1. 基频曲线 ---
subplot(3, 2, 1);
plot(t, f0, 'b', 'LineWidth', 1.0);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
title('基频变化曲线 (Fundamental Frequency)');
grid on; xlim([0, T]); ylim([f_min - 10, f_max + 10]);

% --- 2. 时域波形 ---
subplot(3, 2, 2);
plot(t, dn(:,1), 'Color', [0.2, 0.4, 0.8], 'LineWidth', 0.3);
xlabel('Time (s)'); ylabel('Amplitude');
title('通道 1 时域波形');
grid on; xlim([0, T]);

% --- 3. 频谱图 (Spectrogram) ---
subplot(3, 2, [3, 4]);
nfft_spec = 1024;
noverlap  = round(nfft_spec * 0.92);
spectrogram(dn(:,1), hann(nfft_spec), noverlap, nfft_spec, fs, 'yaxis');
title('短时傅里叶变换频谱图 (STFT Spectrogram)');
colormap('jet');
clim_val = max(abs(get(gca,'CLim')));
caxis([-clim_val, clim_val] * 0.6);
ylim([0, min(500, fs/2)]);
colorbar;
hold on;
for h = 1:length(harmonics)
    plot(t, harmonics(h) * f0 / 1000, 'w--', 'LineWidth', 0.8);
end
hold off;

% --- 4. 功率谱密度 ---
subplot(3, 2, 5);
nfft_psd = 4096;
[pxx, f_psd] = pwelch(dn(:,1), hann(nfft_psd), nfft_psd/2, nfft_psd, fs);
plot(f_psd, 10*log10(pxx), 'b', 'LineWidth', 1);
xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
title('功率谱密度 (Welch PSD)');
grid on; xlim([0, 500]);

% --- 5. 基频直方图 (频率覆盖分布) ---
subplot(3, 2, 6);
histogram(f0, 50, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'none');
xlabel('Frequency (Hz)'); ylabel('Count');
title('基频分布直方图 (Frequency Coverage)');
grid on; xlim([f_min - 10, f_max + 10]);

sgtitle(sprintf('模拟变频率噪声 [模式: %s, fs=%d Hz, T=%d s, f: %d~%d Hz]', ...
    mode, fs, T, f_min, f_max), 'FontSize', 14, 'FontWeight', 'bold');

%% =================== 保存数据 ===================

save('sim_noise.mat', 'dn', 'f0', 'fs', 'T', 'N', 't', 'harmonics', 'mode', 'f_min', 'f_max');
fprintf('数据已保存至 sim_noise.mat\n');
fprintf('  采样率: %d Hz\n', fs);
fprintf('  时长:   %d s (%d 点)\n', T, N);
fprintf('  通道数: %d\n', L);
fprintf('  基频范围: %.1f ~ %.1f Hz (设定: %d ~ %d Hz)\n', min(f0), max(f0), f_min, f_max);
fprintf('  谐波阶次: %s\n', num2str(harmonics));
fprintf('  模式: %s\n', mode);
