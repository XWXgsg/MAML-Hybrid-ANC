%% Modified MAML for single-channel Notch FxLMS initialization
%--------------------------------------------------------------------------
%  Single channel: M=1 loudspeaker, L=1 error mic.
%  Phi_sin, Phi_cos: scalars — initial sin/cos weights for the controller.
%
%  MAML inner loop:
%    e  = Di + Phi_sin*Fx_sin + Phi_cos*Fx_cos
%    Wo_sin = Phi_sin - mu * e * Fx_sin      (one-step inner update)
%    Wo_cos = Phi_cos - mu * e * Fx_cos
%    Grad_sin += -epslon*(mu/Li) * e_jj * Fx_sin(jj) * lamda^(jj-1)
%    Phi_sin  += Grad_sin                    (meta-update)
%
%  Sign convention matches SimulationPlatform (e = d + y):
%    gradient descent: w -= alpha * e * r
%
%  handle class — call obj.reset() to re-initialise for each frequency.
%--------------------------------------------------------------------------
classdef MAML_notch < handle
    properties
        Phi_sin   % scalar — initial sin weight
        Phi_cos   % scalar — initial cos weight
    end

    methods
        function obj = MAML_notch()
            obj.Phi_sin = 0;
            obj.Phi_cos = 0;
        end

        function reset(obj)
            obj.Phi_sin = 0;
            obj.Phi_cos = 0;
        end

        function Er = MAML_initial(obj, Fx_sin, Fx_cos, Di, mu, lamda, epslon)
            % One MAML epoch — updates Phi_sin / Phi_cos in-place.
            %
            %  Fx_sin  [Len_ep x 1]  filtered-x sin  S(z)*sin(theta)
            %  Fx_cos  [Len_ep x 1]  filtered-x cos  S(z)*cos(theta)
            %  Di      [Len_ep x 1]  disturbance
            %  mu      scalar        inner-loop step size
            %  lamda   scalar        forgetting factor  (0 < lamda <= 1)
            %  epslon  scalar        meta learning rate
            %
            %  Returns:
            %  Er      scalar        error at first adapted step

            Fx_sin = flipud(Fx_sin(:));
            Fx_cos = flipud(Fx_cos(:));
            Dis    = flipud(Di(:));
            Len_ep = length(Dis);

            % Step 4: error with current Phi
            e = Dis(1) + obj.Phi_sin * Fx_sin(1) + obj.Phi_cos * Fx_cos(1);

            % Step 5: one-step inner update
            Wo_sin = obj.Phi_sin - mu * e * Fx_sin(1);
            Wo_cos = obj.Phi_cos - mu * e * Fx_cos(1);

            % Step 6: accumulate meta-gradient
            Grad_sin = 0;
            Grad_cos = 0;
            Er = 0;

            for jj = 1:Len_ep
                e_jj = Dis(jj) + Wo_sin * Fx_sin(jj) + Wo_cos * Fx_cos(jj);
                Grad_sin = Grad_sin ...
                         - epslon * (mu/Len_ep) * e_jj * Fx_sin(jj) * (lamda^(jj-1));
                Grad_cos = Grad_cos ...
                         - epslon * (mu/Len_ep) * e_jj * Fx_cos(jj) * (lamda^(jj-1));
                if jj == 1
                    Er = e_jj;
                end
            end

            % Step 7: meta-update (in-place)
            obj.Phi_sin = obj.Phi_sin + Grad_sin;
            obj.Phi_cos = obj.Phi_cos + Grad_cos;
        end
    end
end
