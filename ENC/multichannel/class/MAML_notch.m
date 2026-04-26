%% Modified MAML for multi-channel Notch FxLMS initialization
%--------------------------------------------------------------------------
%  Phi_sin [M x L] / Phi_cos [M x L]
%  Each (m, l) channel pair is trained independently:
%
%    e(l) = Di(l) + sum_m [ Phi_sin(m,l)*Fx_sin(m,l) + Phi_cos(m,l)*Fx_cos(m,l) ]
%
%    grad_sin(m,l) = -epslon*(mu/Li) * e(l) * Fx_sin(m,l) * lamda^(jj-1)
%
%  This matches the per-(m,l) structure of NotchController.iterate():
%    grad_sin(k,m) = sum_l [ r_sin(m,l,k) * e(l) ]
%    w(1,k,m)     -= alpha * grad_sin(k,m)
%
%  To initialise NotchController.w(1,k,m) from Phi_sin [M x L]:
%    w(1,k,m) = sum_l Phi_sin(m,l)
%
%  Sign convention (SimulationPlatform: e = d + y):
%    e = d + y   ->   gradient descent: w -= alpha * e * r
%
%  handle class — call obj.reset() to re-initialise for each frequency.
%--------------------------------------------------------------------------
classdef MAML_notch < handle
    properties
        Phi_sin   % [M x L]  initial sin weights
        Phi_cos   % [M x L]  initial cos weights
        M         % number of loudspeakers
        L         % number of error microphones
    end

    methods
        function obj = MAML_notch(M, L)
            if nargin < 1; M = 1; end
            if nargin < 2; L = 1; end
            obj.M = M;
            obj.L = L;
            obj.Phi_sin = zeros(M, L);
            obj.Phi_cos = zeros(M, L);
        end

        function reset(obj)
            obj.Phi_sin = zeros(obj.M, obj.L);
            obj.Phi_cos = zeros(obj.M, obj.L);
        end

        function Er = MAML_initial(obj, Fx_sin, Fx_cos, Di, mu, lamda, epslon)
            % One MAML epoch — updates Phi_sin / Phi_cos in-place.
            %
            %  Fx_sin  [Len_ep x M x L]   filtered-x sin  S(z)*sin(theta)
            %  Fx_cos  [Len_ep x M x L]   filtered-x cos  S(z)*cos(theta)
            %  Di      [Len_ep x L]        disturbance per mic
            %  mu      scalar              inner-loop step size
            %  lamda   scalar              forgetting factor  (0 < lamda <= 1)
            %  epslon  scalar              meta learning rate
            %
            %  Returns:
            %  Er      scalar              mean aggregated error at first step

            Len_ep = size(Di, 1);

            % Flip time axis (index 1 = most recent, matching MAML convention)
            Di_f     = flipud(Di);            % [Len_ep x L]
            Fx_sin_f = flip(Fx_sin, 1);       % [Len_ep x M x L]
            Fx_cos_f = flip(Fx_cos, 1);       % [Len_ep x M x L]

            % ── Step 4: error per mic with current Phi ────────────────────
            % e(l) = Di(l) + sum_m [ Phi_sin(m,l)*Fx_sin(m,l) + Phi_cos(m,l)*Fx_cos(m,l) ]
            % e_vec: [1 x L]
            e_vec = zeros(1, obj.L);
            for l = 1:obj.L
                for m = 1:obj.M
                    e_vec(l) = e_vec(l) ...
                        + obj.Phi_sin(m,l) * Fx_sin_f(1,m,l) ...
                        + obj.Phi_cos(m,l) * Fx_cos_f(1,m,l);
                end
                e_vec(l) = e_vec(l) + Di_f(1, l);
            end

            % ── Step 5: one-step inner update for each (m, l) ────────────
            % Wo_sin(m,l) = Phi_sin(m,l) - mu * e(l) * Fx_sin(1,m,l)
            Wo_sin = zeros(obj.M, obj.L);
            Wo_cos = zeros(obj.M, obj.L);
            for l = 1:obj.L
                for m = 1:obj.M
                    Wo_sin(m,l) = obj.Phi_sin(m,l) - mu * e_vec(l) * Fx_sin_f(1,m,l);
                    Wo_cos(m,l) = obj.Phi_cos(m,l) - mu * e_vec(l) * Fx_cos_f(1,m,l);
                end
            end

            % ── Step 6: accumulate meta-gradient over episode ─────────────
            Grad_sin = zeros(obj.M, obj.L);
            Grad_cos = zeros(obj.M, obj.L);
            Er_sum   = 0;

            for jj = 1:Len_ep
                % Error per mic with adapted weights Wo at step jj
                e_jj = zeros(1, obj.L);
                for l = 1:obj.L
                    for m = 1:obj.M
                        e_jj(l) = e_jj(l) ...
                            + Wo_sin(m,l) * Fx_sin_f(jj,m,l) ...
                            + Wo_cos(m,l) * Fx_cos_f(jj,m,l);
                    end
                    e_jj(l) = e_jj(l) + Di_f(jj, l);
                end

                % Gradient for each (m, l) independently
                for l = 1:obj.L
                    for m = 1:obj.M
                        Grad_sin(m,l) = Grad_sin(m,l) ...
                            - epslon * (mu/Len_ep) * e_jj(l) ...
                              * Fx_sin_f(jj,m,l) * (lamda^(jj-1));
                        Grad_cos(m,l) = Grad_cos(m,l) ...
                            - epslon * (mu/Len_ep) * e_jj(l) ...
                              * Fx_cos_f(jj,m,l) * (lamda^(jj-1));
                    end
                end

                if jj == 1
                    Er_sum = mean(abs(e_jj));   % mean over mics
                end
            end

            % ── Step 7: meta-update (in-place) ───────────────────────────
            obj.Phi_sin = obj.Phi_sin + Grad_sin;
            obj.Phi_cos = obj.Phi_cos + Grad_cos;

            Er = Er_sum;
        end
    end
end