%% Title: Multi-channel Modified MAML algorithm (1x2x2)
%  K=1 reference, M=2 speakers, L=2 error microphones
%  Used to compute the initial control filter for multi-channel FxLMS
%  Based on Shi et al., IEEE SPL 2021

classdef MAML_Nstep_forget_MC
    properties
        Phi    % Initial control filter [I*M*K, 1]
        I      % Length of each sub-filter
        K      % Number of references
        M      % Number of speakers
        L      % Number of error mics
    end

    methods
        function obj = MAML_Nstep_forget_MC(len_c, K, M, L)
            % len_c : length of each control sub-filter
            % K, M, L : system dimensions
            obj.I = len_c;
            obj.K = K;
            obj.M = M;
            obj.L = L;
            obj.Phi = zeros(len_c * M * K, 1);
        end

        function [obj, Er] = MAML_initial(obj, Fx_seg, Di_seg, s_hat, mu, lamda, epslon)
            % Fx_seg : filtered-reference data segment [I, M, L, K] 
            %          Pre-computed filtered-reference for this training sample.
            %          Fx_seg(:, m, l, k) = s_hat_{ml} * x_k, length I segment
            % Di_seg : disturbance segment [I, L]
            %          Di_seg(:, l) is disturbance at mic l, length I
            % s_hat  : estimated secondary path [J, M, L] (not directly used here 
            %          since Fx_seg is pre-computed)
            % mu     : step size for inner FxLMS update
            % lamda  : forgetting factor
            % epslon : MAML outer learning rate

            I = obj.I;
            K = obj.K;
            M = obj.M;
            L = obj.L;

            % Flip time order (as in original code)
            Di_seg = flipud(Di_seg);  % [I, L]
            % Flip each filtered-reference along time dimension
            for m = 1:M
                for l = 1:L
                    for k = 1:K
                        Fx_seg(:, m, l, k) = flipud(Fx_seg(:, m, l, k));
                    end
                end
            end

            % --- Step 4: Compute error with current Phi ---
            % e_l(1) = d_l(1) - sum_{m,k} w_mk' * fx_{mlk}(:) at time index 1
            e_init = zeros(L, 1);
            for l = 1:L
                e_init(l) = Di_seg(1, l);
                for m = 1:M
                    for k = 1:K
                        idx_s = ((m-1)*K + (k-1)) * I + 1;
                        idx_e = idx_s + I - 1;
                        e_init(l) = e_init(l) - obj.Phi(idx_s:idx_e)' * Fx_seg(:, m, l, k);
                    end
                end
            end

            % --- Step 5: One-step update -> Wo (assumed optimal) ---
            Wo = obj.Phi;
            for m = 1:M
                for k = 1:K
                    idx_s = ((m-1)*K + (k-1)) * I + 1;
                    idx_e = idx_s + I - 1;
                    grad = zeros(I, 1);
                    for l = 1:L
                        grad = grad + e_init(l) * Fx_seg(:, m, l, k);
                    end
                    Wo(idx_s:idx_e) = Wo(idx_s:idx_e) + mu * grad;
                end
            end

            % --- Step 6: Compute N-step gradient with forgetting ---
            Grad = zeros(length(obj.Phi), 1);
            Er = 0;

            for jj = 1:I
                % Build shifted filtered-reference for time index jj
                % Shift: pad with zeros at the end, shift forward
                e_jj = zeros(L, 1);
                for l = 1:L
                    e_jj(l) = Di_seg(jj, l);
                    for m = 1:M
                        for k = 1:K
                            idx_s = ((m-1)*K + (k-1)) * I + 1;
                            idx_e = idx_s + I - 1;
                            % Shifted filtered-reference
                            Fd = [Fx_seg(jj:end, m, l, k); zeros(jj-1, 1)];
                            e_jj(l) = e_jj(l) - Wo(idx_s:idx_e)' * Fd;
                        end
                    end
                end

                % Accumulate gradient
                for m = 1:M
                    for k = 1:K
                        idx_s = ((m-1)*K + (k-1)) * I + 1;
                        idx_e = idx_s + I - 1;
                        grad_mk = zeros(I, 1);
                        for l = 1:L
                            Fd = [Fx_seg(jj:end, m, l, k); zeros(jj-1, 1)];
                            grad_mk = grad_mk + e_jj(l) * Fd;
                        end
                        Grad(idx_s:idx_e) = Grad(idx_s:idx_e) + ...
                            epslon * (mu / I) * grad_mk * (lamda^(jj-1));
                    end
                end

                if jj == 1
                    Er = sum(e_jj.^2);  % Total squared error as training metric
                end
            end

            % --- Step 7: Update Phi ---
            obj.Phi = obj.Phi + Grad;
        end
    end
end
