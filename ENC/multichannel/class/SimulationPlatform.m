classdef SimulationPlatform < handle
%  仿真平台：模拟物理次级声学路径及声波叠加
%  
%  功能:
%    1. 维护控制信号 u 的缓冲区
%    2. 通过次级路径模型 S(z) 将 u 滤波为次级声波 y
%    3. 将初级噪声 d 与次级声波 y 叠加，得到误差信号 e
%
%  Author: Zhicheng Zhang
%  Date  : 2026-2-6
%--------------------------------------------------------------------------

    properties
        M;        % 控制扬声器数量
        L;        % 误差麦克风数量
        d;        % 初级噪声信号 [N, L]
        s;        % 次级路径模型 [J, M, L]
        e;        % 误差信号 [L, 1]
        y;        % 次级声波 [L, 1]
        u;        % 当前控制信号 [M, 1]
        n;        % 当前时间步
        uBuffer;  % 控制信号缓冲区 [M, J]
        state = 0;
    end

    properties (Constant)
        J = 128;  % 次级路径模型滤波器阶数
    end

    methods
        function obj = SimulationPlatform(d, s, M, L)
            %  构造函数
            %  输入:
            %    d - 初级噪声 (期望信号) [N, L]
            %    s - 次级路径模型 [J, M, L]
            %    M - 控制扬声器数量
            %    L - 误差麦克风数量
            
            obj.d = d;
            obj.M = M;
            obj.L = L;

            if isequal(size(s,1), obj.J) && isequal(size(s,2), obj.M) && isequal(size(s,3), obj.L)
                obj.s = s;
                obj.state = 1;
            else
                error("次级路径矩阵 s 维度必须为 [%d, %d, %d]，当前为 [%s]", ...
                    obj.J, obj.M, obj.L, num2str(size(s)));
            end

            obj.uBuffer = zeros(obj.M, obj.J);
            obj.y = zeros(obj.L, 1);
            obj.e = zeros(obj.L, 1);
            obj.n = 1;
        end

        function updateuBuffer(obj)
            % 更新控制信号缓冲区 [M, J] (FIFO)
            obj.uBuffer = [obj.u, obj.uBuffer(:, 1:end-1)];
        end

        function ufilter2y(obj)
            % 通过次级路径模型计算次级声波
            % s: [J, M, L] -> [J*M, L],  uBuffer: [M, J] -> [J*M, 1]
            ss = reshape(obj.s, [], obj.L);
            uu = reshape(obj.uBuffer', [], 1);
            obj.y = ss' * uu;
        end

        function SoundSuperposition(obj, n)
            % 声波叠加: e = d + y
            obj.e = obj.d(n, :)' + obj.y;
        end

        function sim(obj, u)
            % 单步仿真
            obj.u = u;
            obj.updateuBuffer();
            obj.ufilter2y();
            obj.SoundSuperposition(obj.n);
            obj.n = obj.n + 1;
        end
    end
end
