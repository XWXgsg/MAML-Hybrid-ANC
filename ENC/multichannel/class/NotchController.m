classdef NotchController < handle
%  Notch Filter (Narrowband Feedforward ANC) Controller
%  Filtered-x LMS algorithm based on internally generated reference signals (sin/cos)
%  Suitable for active noise control of periodic noise (e.g., engine order noise)
%
%  Principle:
%    For each target frequency f_k, sin(theta_k) and cos(theta_k) are used as references.
%    The control signal u is synthesized through adaptive weights w, and the reference
%    signals are filtered through the secondary path model S(z) (Filtered-X) to ensure
%    LMS convergence.
%
%  Author: Wenxuan Xu
%  Date  : 2026-3-2
%--------------------------------------------------------------------------

    properties (Constant)
        I = 2;          % Number of weights per frequency point (sin and cos)
    end

    properties
        K;              % Number of target frequencies (number of frequency components)
        M;              % Number of control loudspeakers
        L;              % Number of error microphones
        fs;             % Sampling rate
        J;              % Order of secondary path filters

        w;              % Controller weights [I, K, M]
        s;              % Secondary path model [J, M, L]
        alpha;          % Step size

        % Internal reference signals
        theta;          % Phase accumulator [K, 1]
        freqs;          % Target frequency list [K, 1]

        % Buffers for Filtered-X
        sinBuffer;      % Sin reference signal history [K, J]
        cosBuffer;      % Cos reference signal history [K, J]
    end

    methods
        function obj = NotchController(fs, freqs, alpha, NumSpeaker, NumErrorMic, J)
            %  Constructor
            %  Input:
            %    fs          - Sampling rate (Hz)
            %    freqs       - Target frequency vector [K x 1] (Hz)
            %    alpha       - Adaptive step size
            %    NumSpeaker  - Number of control loudspeakers M
            %    NumErrorMic - Number of error microphones L
            %    J           - Order of secondary path filters
            
            obj.fs = fs;
            obj.freqs = freqs(:);
            obj.K = length(freqs);
            obj.M = NumSpeaker;
            obj.L = NumErrorMic;
            obj.J = J;
            obj.alpha = alpha;

            obj.w = zeros(obj.I, obj.K, obj.M);
            obj.s = zeros(obj.J, obj.M, obj.L);
            obj.theta = zeros(obj.K, 1);

            % Sin/cos reference signal buffers, length J (secondary path order)
            obj.sinBuffer = zeros(obj.K, obj.J);
            obj.cosBuffer = zeros(obj.K, obj.J);
        end

        function u = iterate(obj, en)
            %  Single-step iteration
            %  Input:
            %    en - Current error signal [L, 1]
            %  Output:
            %    u  - Control signal [M, 1]
            
            %  Update phase and generate internal reference signals ===
            obj.theta = obj.theta + 2*pi*obj.freqs / obj.fs;
            obj.theta = mod(obj.theta, 2*pi);
            
            x_sin = sin(obj.theta);   % [K, 1]
            x_cos = cos(obj.theta);   % [K, 1]

            %  Update reference signal buffers (FIFO) 
            obj.sinBuffer = [x_sin, obj.sinBuffer(:, 1:end-1)];
            obj.cosBuffer = [x_cos, obj.cosBuffer(:, 1:end-1)];

            %  Calculate Filtered-X signals r(n) 
            %  For each combination (m, l, k):
            %    r_sin(m,l,k) = sum_j s(j,m,l) * sinBuffer(k,j)
            %    r_cos(m,l,k) = sum_j s(j,m,l) * cosBuffer(k,j)
            %  i.e., sin/cos references filtered through secondary path S(z)
            
            r_sin = zeros(obj.M, obj.L, obj.K);  % Filtered sin
            r_cos = zeros(obj.M, obj.L, obj.K);  % Filtered cos

            for m = 1:obj.M
                for l = 1:obj.L
                    s_ml = obj.s(:, m, l);          % [J, 1]
                    for k = 1:obj.K
                        r_sin(m, l, k) = s_ml' * obj.sinBuffer(k, :)';
                        r_cos(m, l, k) = s_ml' * obj.cosBuffer(k, :)';
                    end
                end
            end

            %  Adaptive weight update (Filtered-X LMS) 
            %  w(i,k,m) = w(i,k,m) - alpha * sum_l [ r(m,l,k,i) * e(l) ]
            for m = 1:obj.M
                for k = 1:obj.K
                    grad_sin = 0;
                    grad_cos = 0;
                    for l = 1:obj.L
                        grad_sin = grad_sin + r_sin(m, l, k) * en(l);
                        grad_cos = grad_cos + r_cos(m, l, k) * en(l);
                    end
                    obj.w(1, k, m) = obj.w(1, k, m) - obj.alpha(1,k) * grad_sin;
                    obj.w(2, k, m) = obj.w(2, k, m) - obj.alpha(1,k) * grad_cos;
                end
            end

            %  Synthesize control signal 
            %  u(m) = sum_k [ w(1,k,m)*sin(theta_k) + w(2,k,m)*cos(theta_k) ]
            u = zeros(obj.M, 1);
            for m = 1:obj.M
                for k = 1:obj.K
                    u(m) = u(m) + obj.w(1, k, m) * x_sin(k) ...
                                + obj.w(2, k, m) * x_cos(k);
                end
            end
        end
    end
end