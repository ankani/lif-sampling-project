function [mu_z, stim_mu, outer_z, Q] = e_step(params, data)
%EM.E_STEP compute relevant expectations of z with respect to the given data and current model
%parameters. Also compute value of EM objective 'Q'
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

for n=1:params.N
    %% Get posterior over truncated Z
    stim = data(:, n);
    z = EM.truncate(params, stim)'; % z has size [H x T] where T depends on params.truncate
    T = size(z, 2);
    log_p = EM.log_joint(params, stim, z);
    
    % Normalize p for the truncated set of states we are using. p is a column vector of T
    % probabilities
    p = exp(log_p(:) - max(log_p));
    p = p / sum(p);
    
    %% Compute expectations
    
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