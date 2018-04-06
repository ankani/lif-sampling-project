function log_p = log_joint(params, stim, zs)
%EM.JOINT compute log joint probability of a given stimulus value for potentially many values of z,
%vectorized. If vectorizing over multiple values of z, shape should be [H x K]

% TODO allow stim'*stim, stim'*params.G, and params.G'*params.G to be precomputed and passed in

projection = params.G * zs;

log_prior = params.H * log(1 - params.prior) + sum(zs, 1) * (log(params.prior) - log(1 - params.prior));
log_like = -(stim' * stim - 2 * stim' * params.G * zs + sum(projection .* projection, 1)) / (2 * params.sigma^2) - params.pix * log(params.sigma);

log_p = log_prior + log_like;

end