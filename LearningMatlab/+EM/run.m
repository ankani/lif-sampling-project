function [params, Q, params_history] = run(params, data)
%EM.RUN run EM algorithm until convergence or up to params.max_iter iterations.

data_inner = sum(sum(data .* data));

Q = zeros(params.max_iter + 1);
timing = zeros(params.max_iter);

if nargout > 2, params_history = params; end

T = params.anneal_init;
params.sigma = T * params.sigma;

lastQ = -inf;

for itr=1:params.max_iter
    prev_params = params;
    
    iter_start = tic;

    % E Step
    if params.debug, fprintf('%d/%d\t', itr, params.max_iter); end
    [mu_z, stim_mu, outer_z, Q(itr)] = EM.e_step(params, data);
    
    deltaQ = Q(itr) - lastQ;
    lastQ = Q(itr);
    
    if isnan(Q(itr))
        error('matlab:EM:run:diverged', 'Learning diverged! NaN value of Q');
    end
    
    % M Step
    params = EM.m_step(params, data_inner, mu_z, stim_mu, outer_z);
    deltaParams = deltaStruct(params, prev_params, {'sigma', 'prior', 'G'});
    params.sigma = T * params.sigma;
    
    if nargout > 2
        params_history(itr+1) = params;
    end
    
    % Diagnostics
    timing(itr) = toc(iter_start);

    if params.debug
        fprintf('\tQ = %.2e', Q(itr));
        fprintf('\tdQ = %.2e', deltaQ);
        fprintf('\tdParams = %.2e', deltaParams);
        fprintf('\t%.2fs per iteration\n', mean(timing(1:itr)));
    end
    
    % Check for convergence
    if deltaParams < params.tol
        break;
    end
    
    % Update annealing
    T = 1 + (T - 1) * params.anneal_decay;
end

% Once converged, undo scaling of sigma by T.
params.sigma = params.sigma / T;

[~, ~, ~, Q(itr+1)] = EM.e_step(params, data);
Q(itr+2:end) = [];

end

function delta = deltaStruct(s1, s2, fields)
delta2 = 0;
for iField = 1:length(fields)
    deltaField = s1.(fields{iField}) - s2.(fields{iField});
    delta2 = delta2 + sum(deltaField(:).^2);
end
delta = sqrt(delta2);
end