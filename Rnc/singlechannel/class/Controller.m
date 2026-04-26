classdef Controller < handle
    %UNTITLED8 此处显示有关此类的摘要

    properties (Constant)
        positive = 1e-5;
        I = 512;      %Order of control filter
        J = 256;      %Orde of Secondary path model filter
    end
    %real-time data
    properties
        K;        %Number of References
        M;        %Number of Control Speakers
        L;        %Number of Error Micphones
        x;          %reference signals
        w;          %controller filter coeffs[I,K,M]
        u           %drive for spks;control signals from controller
        e;          %erroe signals
        r;          %filtered references
        s;          %secondary path filter loaded from PlantMoedl
        alpha;      %step for adapt controller weights
        MeanPower;
        beta;       %leakage
        d;          %noise    if simulation
        conlSwtich = 1;
        resetSwtich = 0;
    end
    %Buffer data
    properties
        xBuffer;
        xBuffer2;
        rBuffer;
        uBuffer;    %if simulation
    end
    %contructor
    methods
        function obj = Controller(alpha,NumRef,NumSpeaker,NumErrorMic)
            obj.K = NumRef;        %Number of References
            obj.M = NumSpeaker;        %Number of Control Speakers
            obj.L = NumErrorMic;        %Number of Error Micphones
            obj.x = zeros(obj.K,1);
            obj.w = zeros(obj.I,obj.K,obj.M);   %[I,K,M]
            obj.u = zeros(obj.M,1);
            obj.e = zeros(obj.L,1);
            obj.r = zeros(obj.M,obj.L,obj.K);   %[M,L,K]
            obj.s = zeros(obj.J,obj.M,obj.L);   %[J,M,L]
            obj.alpha = alpha;
            obj.beta = 0;
            obj.d = zeros(obj.L,1);
            obj.xBuffer = zeros(obj.K,obj.I);
            obj.rBuffer = zeros(obj.M,obj.L,obj.K,obj.I);
            obj.uBuffer = zeros(obj.M,obj.J);
            obj.xBuffer2 = zeros(obj.K,obj.J);
        end


        function updataxBufferI(obj)
            obj.xBuffer = [obj.x,obj.xBuffer(:,1:end-1)];
        end
        function updataxBufferJ(obj)
            obj.xBuffer2 = [obj.x,obj.xBuffer2(:,1:end-1)];
        end

        function xfilter2r(obj)
            %pagemtimes([K][J],[J][M][L])->[K][M][L]
            obj.r = pagemtimes(obj.xBuffer2,obj.s);
            %[K][M][L]->[M][L][K]
            obj.r = permute(obj.r,[2,3,1]);
        end
        function nomalizeAlpha(obj)
            obj.MeanPower = sum(obj.xBuffer(:).^2) / (obj.I*obj.K);
        end

        function updatarBuffer(obj)
            newr = obj.r(:,:,:,1);
            obj.rBuffer = cat(4,newr,obj.rBuffer(:,:,:,1:end-1));
        end

        function adaptw(obj)
            % alpha_n=obj.alpha/(obj.MeanPower+obj.positive);
            alpha_n = obj.alpha;
            %rBuffer:[M][L][K][I]->[I*K*M,L]
            rBuffer_permute  = permute(obj.rBuffer,[4,3,1,2]);
            rBuffer_reshaped = reshape(rBuffer_permute,[],obj.L);
            r_e = rBuffer_reshaped * obj.e;       %[I*K*M,1]
            %reshape r_e[I*K*M,1]->[I][K][M]
            r_e = reshape(r_e, obj.I, obj.K, obj.M);
            % [FIX] 原来是 + alpha_n*r_e，符号与FxLMS的 Wc = Wc + muw*e*Rf 一致，保持不变。
            % 但原代码注释掉的循环版本是 -= alpha_n*grad，两者矛盾。
            % 这里以FxLMS为基准：e = d - y，梯度下降方向为 +mu*e*r，保留加号。
            obj.w = (1 - obj.beta) * obj.w + alpha_n * r_e;
            if obj.resetSwtich
                obj.w = zeros(obj.I, obj.K, obj.M);
            end
        end

        function xfilter2u(obj)
            %xBuffer:[K][I]->[I*K][1]
            xBuffer_reshaped = reshape(obj.xBuffer',[],1);
            %w:[I][K][M]->[M][I*K]
            w_reshaped = reshape(obj.w,[],obj.M)';
            obj.u = w_reshaped * xBuffer_reshaped;
        end

        % [FIX] 修正 iterate() 内部执行顺序，与 FxLMS 对齐：
        %   FxLMS 顺序：移位存x → 计算y(u) → 计算e → 更新w
        %   原代码顺序：更新w → 移位存x → 计算r → 存rBuffer → 计算u  （错误）
        %   修正顺序：移位存x → 计算r → 存rBuffer → 计算u → 更新w    （正确）
        function u = iterate(obj, xn, en)
            obj.x = xn';    %[K,1]
            obj.e = en;     %[L,1]

            % Step 1: 先移位，将当前x存入buffer（对应FxLMS的 XD=[Rf(tt);XD(1:end-1)]）
            obj.updataxBufferI();   % update xBuffer [K,I]
            obj.updataxBufferJ();   % update xBuffer2 [K,J]

            % Step 2: 用当前xBuffer2计算filtered-reference r
            obj.xfilter2r();        % calculate r [M,L,K]

            % Step 3: 将当前r存入rBuffer（移位）
            obj.updatarBuffer();    % update rBuffer [M,L,K,I]

            % Step 4: 用当前w和xBuffer计算控制输出u（对应FxLMS的 y=Wc'*XD）
            obj.xfilter2u();        % calculate u [M,1]
            u = obj.u * obj.conlSwtich;

            % Step 5: 最后用当前rBuffer和e更新w（对应FxLMS的 Wc=Wc+muw*e*Rf_i）
            % 注意：此处的e是从外部传入的，即上一时刻SimulationPlatform输出的误差，
            % 与FxLMS中 e=Dis(tt)-y_t 在时序上等价（都是当前输出前的误差）。
            obj.nomalizeAlpha();    % compute reference power (for optional normalized step)
            obj.adaptw();           % update w
        end
    end
end