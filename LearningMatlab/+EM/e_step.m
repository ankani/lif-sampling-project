function [mu_z, stim_mu, stim_G_mu, outer_z, G_z, Q] = e_step(params, data)
%EM.E_STEP compute relevant expectations of z with respect to the given data and current model
%parameters. Also compute value of EM objective 'Q'
%
%In the paper, stim_mu is 'S', outer_z is 'Z', G_z is 'Gamma', and stim_G_mu is 'Delta'

% mu_z will contain expected value of z summed over data points
mu_z = zeros(params.H, 1);
% stim_mu will contain stim*mu_z' summed over data points
stim_mu = zeros(params.pix, params.H);
% stim_G_mu will contain expected value of stim G mu_z summed over data points
stim_G_mu = 0;
% outer_z will contain expected value of zz' summed over data points
outer_z = zeros(params.H, params.H);
% G_z will contain expected value of z'G'Gz summed over data points
G_z = 0;

% EM objective Q is a lower bound on the data log likelihood. First compute terms that do not depend
% on data.
Q = params.N * (params.H * log(1 - params.prior) - params.pix * log(params.sigma) - log(2*pi) / 2);

log_prior = log(params.prior) - log(1 - params.prior);
GG = reshape(params.G' * params.G, [params.H params.H 1]);

for n=1:params.N
    stim = data(:, n);
    z = EM.truncate(params, stim)'; % z has size [H x T] where T depends on params.truncate
    T = size(z, 2);
    log_p = EM.log_joint(params, stim, z);
    
    % Normalize p for the truncated set of states we are using. p is a column vector of
    % T probabilities
    log_p = log_p - max(log_p);
    p = exp(log_p(:));
    p = p / sum(p);
    
    % zzT has size [H H T] and contains z*z' weighted by p for each of the T states
    z1 = reshape(z, [params.H 1 T]);
    z2 = reshape(z, [1 params.H T]);
    zzT = (z1 .* z2) .* reshape(p, [1 1 T]);
    
    % Compute expectations
    mu_z = mu_z + z * p;
    stim_mu = stim_mu + stim * mu_z';
    stim_G_mu = stim_G_mu + stim' * params.G * mu_z;
    outer_z = outer_z + sum(zzT, 3);
    G_z = G_z + sum(sum(sum(GG .* zzT)));
    
    % Update Q with terms that depend on each data point
    Q = Q + sum(mu_z) * log_prior - (stim'*stim - 2*stim_G_mu + G_z) / (2 * params.sigma^2);
end

end