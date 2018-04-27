function params = m_step(params, data_inner, mu_z, stim_mu, outer_z)
%EM.M_STEP using the expectations computed in EM.e_step, compute updated model parameters that
%maximize the EM objective.

% Update prior. Note: since we have only a single shared prior for all latents, prior becomes the
% mean across all latents rather than keeping it separate for each latent.
if ~any(strcmpi('prior', params.fixed))
    params.prior = sum(mu_z) / (params.H * params.N);
end

% Update projective fields (note: this must happen before updating sigma since sigma depends on G)
if ~any(strcmpi('G', params.fixed))
    params.G = stim_mu / outer_z;
end
GG = params.G' * params.G;

% Update sigma.
expected_xGz = params.G(:)' * stim_mu(:);
expected_zGGz = GG(:)' * outer_z(:);
sigma2 = (data_inner - 2 * expected_xGz + expected_zGGz) / (params.N * params.pix);
if ~any(strcmpi('sigma', params.fixed))
    params.sigma = sqrt(sigma2);
end

end