function S = NLMSinit(w0,mu,leak)
S.coeffs = w0(:);
S.step = mu;
S.leakage = leak;
S.iter = 0;
S.AdaptStart = length(w0);
S.alpha = 1e-5;
