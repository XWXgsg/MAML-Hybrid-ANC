classdef SimulationPlatform < handle
    %UNTITLED9 此处显示有关此类的摘要
    %   此处显示详细说明

    properties
        d;
        x;
        s;
        e;
        y;
        u;
        n;
        uBffer;
        state = 0;
    end
    properties (Constant)
        K = 1;        %Number of References
        M = 1;        %Number of Control Speakers
        L = 1;        %Number of Error Micphones
        J = 256;      %Orde of Secondary path model filter
    end
    methods
        function obj = SimulationPlatform(x,d,s)
            obj.x = x;
            obj.d = d;
            if isequal([obj.L,obj.M],[1,1]) 
               if isequal(size(s), [obj.J,1])
                obj.s = s;
                obj.state = 1;
                else
                print("error input, The dimension of matrix s must be [a, b, c]")
               end
            else
            if isequal(size(s), [obj.J,obj.M,obj.L])
                obj.s = s;
                obj.state = 1;
            else
                print("error input, The dimension of matrix s must be [a, b, c]")
            end
            end
            obj.uBffer = zeros(obj.M,obj.J);
            obj.y = zeros(obj.L,1);
            obj.e = zeros(obj.L,1);
            obj.n = 1;
        end
        %Update the control signal u buffer [M,J] for secondary path simulation.
        function updatauBuffer(obj)
            obj.uBffer = [obj.u,obj.uBffer(:,1:end-1)];
        end
        %Simulate the process from the driving signal u [M,1] to the secondary sound wave y [L,1] in the physical path.
        function ufilter2y(obj)
            obj.y = zeros(obj.L,1);
            % s:[J,M,L]->[J*M,L]
            ss = reshape(obj.s,[],obj.L);
            % uBuffer:[M,J]->[J*M,1]
            uu = reshape(obj.uBffer',[],1);
            obj.y = ss'*uu;
        end
        %Simulate the process of sound wave superposition
        function SoundSuperposition(obj,n)
            obj.e = obj.d(n,:)' - obj.y;
        end
        function sim(obj,u)
            obj.u = u;
            obj.updatauBuffer;
            obj.ufilter2y;
            obj.SoundSuperposition(obj.n);
            obj.n = obj.n + 1;
        end
    end
end