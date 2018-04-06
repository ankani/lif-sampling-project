function params = m_step(params, data_inner, mu_z, stim_mu, stim_G_mu, outer_z, G_z)
%EM.M_STEP using the expectations computed in EM.e_step, compute updated model parameters that
%maximize the EM objective.

% Update prior. Note: since we have only a single shared prior for all latents, prior becomes the
% mean across all latents rather than keeping it separate for each latent.
params.prior = sum(mu_z) / (params.H * params.N);

% Update sigma.
sigma2 = (data_inner - 2 * stim_G_mu + G_z) / (params.N * params.pix);
params.sigma = sqrt(sigma2);

% Update projective fields.
params.G = stim_mu / outer_z;

end