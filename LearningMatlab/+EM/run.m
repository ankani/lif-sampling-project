function [params, Q] = run(params, data)
%EM.RUN run EM algorithm until convergence or up to params.max_iter iterations.

data_inner = sum(sum(data .* data));

Q = zeros(params.max_iter + 1);
timing = zeros(params.max_iter);

for itr=1:params.max_iter
    prev_params = params;
    
    iter_start = tic;

    % E Step
    if params.debug, fprintf('%d\tE-Step', itr); end
    [mu_z, stim_mu, stim_G_mu, outer_z, G_z, Q(itr)] = EM.e_step(params, data);
    
    % M Step
    if params.debug, fprintf('\tM-Step\n'); end
    params = EM.m_step(params, data_inner, mu_z, stim_mu, stim_G_mu, outer_z, G_z);
    
    % Diagnostics
    timing(itr) = toc(iter_start);

    if params.debug
        fprintf('\t%.2fs per iteration\n', mean(timing(1:itr)));
        fprintf('\tQ = %.2e\n', Q(itr));
    end
    
    % Check for convergence
    prior_diff = abs(params.prior - prev_params.prior);
    sigma_diff = abs(params.sigma - prev_params.sigma);
    G_diff = max(abs(params.G(:) - prev_params.G(:)));
    
    if all([prior_diff sigma_diff G_diff] < params.tol)
        break;
    end
end

[~, ~, ~, ~, ~, Q(itr+1)] = EM.e_step(params, data);
Q(itr+2:end) = [];

end