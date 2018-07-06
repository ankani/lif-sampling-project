function [params, Q, params_history] = runTVI(params, data)
%EM.RUNTVI run Truncated Variational Inference algorithm until convergence or up to params.max_iter
%iterations. Keeps in memory of 'params.tvi_samples' samples of the latents per data point, which
%may make significant memory pressure; a total of [params.H x params.N x params.tvi_samples] samples
%will be tracked at all times.

assert(params.N == size(data, 2), 'mismatch between params.N and size of data');

data_inner = sum(sum(data .* data));

Q = zeros(params.max_iter + 1);
timing = zeros(params.max_iter);

if nargout > 2, params_history = params; end

T = params.anneal_init;
params.sigma = T * params.sigma;

lastQ = -inf;

% Initialize samples
samplesSet = zeros(params.N, params.H, params.tvi_samples);
samplesPosteriors = zeros(params.N, params.tvi_samples);
params.trunctate = ceil(log2(params.tvi_samples));
for n=1:params.N
    stim = data(:, n);
    z_trunc = EM.truncate(params, stim)';
    % By setting params.trunctate to ceil(log2(tvi_samples)), params.tvi_samples is smaller than (or
    % equal to) what we got from truncate(); keep only the top values
    posteriors = EM.log_joint(params, stim, z_trunc);
    [~, sort_idx] = sort(posteriors, 'descend');
    samplesSet(n, :, :) = z_trunc(:, sort_idx(1:params.tvi_samples));
    samplesPosteriors(n, :) = posteriors(sort_idx(1:params.tvi_samples));

    assert(size(unique(squeeze(samplesSet(n, :, :))', 'rows'), 1) == params.tvi_samples);
end

for itr=1:params.max_iter
    prev_params = params;
    
    iter_start = tic;

    % E Step
    if params.debug, fprintf('%d/%d\t', itr, params.max_iter); end
    [mu_z, stim_mu, outer_z, Q(itr), samplesSet, samplesPosteriors, new_samples] = ...
        EM.e_step_tvi(params, data, samplesSet, samplesPosteriors);
    
    deltaQ = Q(itr) - lastQ;
    lastQ = Q(itr);
    
    if isnan(Q(itr))
        error('matlab:EM:run:diverged', 'Learning diverged! NaN value of Q');
    end
    
    % M Step
    params = EM.m_step(params, data_inner, mu_z, stim_mu, outer_z, T / 100);
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
        fprintf('\t%.2f +/- %.2f samples/datapt changed', mean(new_samples), std(new_samples));
        fprintf('\t%.2fs per iteration\n', mean(timing(1:itr)));
    end
    
    % Check for convergence
    if deltaParams < params.tol || deltaQ < 0
        break;
    end
    
    % Update annealing
    T = 1 + (T - 1) * params.anneal_decay;
end

[mu_z, stim_mu, outer_z, Q(itr+1), ~, ~] = EM.e_step_tvi(params, data, samplesSet, samplesPosteriors);
Q(itr+2:end) = [];

% Once converged, perform a final M-step without annealing
params = EM.m_step(params, data_inner, mu_z, stim_mu, outer_z, 0);

end

function delta = deltaStruct(s1, s2, fields)
delta2 = 0;
for iField = 1:length(fields)
    deltaField = s1.(fields{iField}) - s2.(fields{iField});
    delta2 = delta2 + sum(deltaField(:).^2);
end
delta = sqrt(delta2);
end