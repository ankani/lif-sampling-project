function [mu_z, stim_mu, outer_z, Q, z_samples, z_posteriors, num_new_samples] = ...
    e_step_tvi(params, data, z_samples, z_posteriors)
%EM.E_STEP_TVI propose new samples of z and compute relevant expectations of z with respect to the
%given data and new set of samples. Also compute value of EM objective 'Q'
%
%In the paper, stim_mu is 'S', outer_z is 'Z', G_z is 'Gamma', and stim_G_mu is 'Delta'

H = params.H;
P = params.pix;

% mu_z will contain expected value of z summed over data points
mu_z = zeros(H, 1);
% stim_mu will contain stim*mu_z' summed over data points
stim_mu = zeros(P, H);
% outer_z will contain expected value of zz' summed over data points
outer_z = zeros(H, H);

% EM objective Q is a lower bound on the data log likelihood. First compute terms that do not depend
% on data.
Q = params.N * (H * log(1 - params.prior) - P * log(params.sigma) - log(2*pi) / 2);

GG = params.G' * params.G;
log_prior = log(params.prior) - log(1 - params.prior);

num_new_samples = zeros(params.N, 1);

for n=1:params.N
    %% Propose new samples of z(n, :) using VB estimate of marginals
    stim = data(:, n);
    marginals = EM.variational_bayes(params, stim, 1e-6);
    new_samples = rand(1, params.H, params.tvi_propose) < reshape(marginals, 1, params.H, 1);
    new_posteriors = EM.log_joint(params, stim, squeeze(new_samples));
    
    % Keep best (unique) samples only
    all_samples = cat(3, z_samples(n, :, :), new_samples);
    all_posteriors = horzcat(z_posteriors(n, :), new_posteriors);
    [~, unqIdx] = unique(squeeze(all_samples)', 'rows');
    all_samples = all_samples(:, :, unqIdx);
    all_posteriors = all_posteriors(unqIdx);
    [~, sort_idx] = sort(all_posteriors, 'descend');
    z_samples(n, :, :) = all_samples(:, :, sort_idx(1:params.tvi_samples));
    z_posteriors(n, :) = all_posteriors(sort_idx(1:params.tvi_samples));
    
    % Count number of 'new' samples; these are any samples in the 'tail' that have been sorted to
    % now be among the top tvi_samples
    num_new_samples(n) = sum(sort_idx(params.tvi_samples+1:end) <= params.tvi_samples);
    
    %% Compute expectations

    T = params.tvi_samples;
    z = squeeze(z_samples(n, :, :));
    log_p = z_posteriors(n, :);
    
    % Normalize p for the truncated set of states we are using. p is a column vector of T
    % probabilities
    p = exp(log_p(:) - max(log_p));
    p = p / sum(p);
    
    % Compute mean of z for this data point only
    mu = z * p;
    
    % zzT has size [H H T] and contains z*z' weighted by p for each of the T states
    z1 = reshape(z, [H 1 T]);
    z2 = reshape(z, [1 H T]);
    zzT = (z1 .* z2) .* reshape(p, [1 1 T]);
    outer = sum(zzT, 3);
    
    % Accumulate results
    mu_z = mu_z + mu;
    stim_mu = stim_mu + stim * mu';
    outer_z = outer_z + outer;
    
    % Update Q with terms that depend on each data point
    expected_zGGz = sum(sum(sum(zzT .* GG)));
    Q = Q + sum(mu) * log_prior - (stim'*stim - 2 * stim' * params.G * mu + expected_zGGz) / (2 * params.sigma^2);
end

end