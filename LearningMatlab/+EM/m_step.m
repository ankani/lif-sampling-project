function params = m_step(params, data_inner, mu_z, stim_mu, outer_z)
%EM.M_STEP using the expectations computed in EM.e_step, compute updated model parameters that
%maximize the EM objective.

% Update prior. Note: since we have only a single shared prior for all latents, prior becomes the
% mean across all latents rather than keeping it separate for each latent.
params.prior = sum(mu_z) / (params.H * params.N);

% Update projective fields (note: this must happen before updating sigma since sigma depends on G)
params.G = stim_mu / outer_z;
GG = params.G' * params.G;

% Update sigma.
expected_xGz = params.G(:)' * stim_mu(:);
expected_zGGz = GG(:)' * outer_z(:);
sigma2 = (data_inner - 2 * expected_xGz + expected_zGGz) / (params.N * params.pix);
params.sigma = sqrt(sigma2);

end