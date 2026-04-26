classdef SimulationPlatformOSPM_MC < handle
    % Multi-channel ANC simulation platform
    % Configuration: K=1 reference, M=2 control speakers, L=2 error microphones
    % Secondary paths: S11, S12, S21, S22 (speaker m -> mic l)
    % Primary paths:   P1, P2           (noise source -> mic l)

    properties
        d;          % Disturbance signal at error mics [L,1]
        x;          % Reference signal matrix [N_total, K]
        s;          % Secondary path filters [J, M, L]
        p;          % Primary path filters   [O, K, L]
        e;          % Error signal [L,1]
        y;          % Secondary sound [L,1]
        u;          % Control signal [M,1]
        n;          % Current sample index
        uBffer;     % Control signal buffer [M, J]
        xBffer;     % Reference signal buffer [K, O]
        state;
    end

    properties (Constant)
        K = 1;        % Number of References
        M = 2;        % Number of Control Speakers
        L = 2;        % Number of Error Microphones
        J = 256;      % Order of Secondary path model filter
        O = 256;      % Order of Primary path model filter
        I = 512;      % Order of control filter
    end

    methods
        function obj = SimulationPlatformOSPM_MC(x, s, p)
            % x: reference signal [N_total, K]
            % s: secondary path [J, M, L]  — s(:,m,l) is speaker m -> mic l
            % p: primary path   [O, K, L]  — p(:,k,l) is ref k -> mic l
            obj.x = x;
            obj.s = s;
            obj.p = p;
            obj.state = 1;
            obj.uBffer = zeros(obj.M, obj.J);
            obj.xBffer = zeros(obj.K, obj.O);
            obj.y = zeros(obj.L, 1);
            obj.e = zeros(obj.L, 1);
            obj.d = zeros(obj.L, 1);
            obj.n = 1;
        end

        % Update control signal buffer [M, J]
        function updatauBuffer(obj)
            obj.uBffer = [obj.u, obj.uBffer(:, 1:end-1)];
        end

        % Update reference signal buffer [K, O]
        function updataxBuffer(obj)
            obj.xBffer = [obj.x(obj.n,:)', obj.xBffer(:, 1:end-1)];
        end

        % u -> y through secondary path: y(l) = sum_m s(:,m,l)' * uBffer(m,:)'
        function ufilter2y(obj)
            obj.y = zeros(obj.L, 1);
            for l = 1:obj.L
                for m = 1:obj.M
                    obj.y(l) = obj.y(l) + obj.s(:,m,l)' * obj.uBffer(m,:)';
                end
            end
        end

        % x -> d through primary path: d(l) = sum_k p(:,k,l)' * xBffer(k,:)'
        function xfilter2d(obj)
            obj.d = zeros(obj.L, 1);
            for l = 1:obj.L
                for k = 1:obj.K
                    obj.d(l) = obj.d(l) + obj.p(:,k,l)' * obj.xBffer(k,:)';
                end
            end
        end

        % Sound superposition: e = d + y
        function SoundSuperposition(obj)
            obj.e = obj.d + obj.y;
        end

        % Run one time step
        function sim(obj, u)
            obj.u = u;    % u: [M, 1]
            obj.updatauBuffer;
            obj.updataxBuffer;
            obj.ufilter2y;
            obj.xfilter2d;
            obj.SoundSuperposition;
            obj.n = obj.n + 1;
        end
    end
end
