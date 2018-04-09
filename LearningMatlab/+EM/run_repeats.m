function [best_params, best_Q, fits, Qs] = run_repeats(params, data, repeats, init_method, max_retries)
%EM.run_repeats re-initialize and re-fit multiple times and keep the best result.
%
% best_params = EM.run_repeats(params, data, repeats) re-initialize and re-fit 'repeats' times and
% return the best parameters.
%
% [best_params, fits, Qs] = EM.run_repeats(...) additionally return the fitted parameters and values
% of Q (the log likelihood lower bound per iteration) for each of the runs.
%
% ... = EM.run_repeats(..., init_method) specifies how to initialize projective fields from data
%
% ... = EM.run_repeats(..., init_method, max_retries) specifies how many times to retry per repeat
% if the fit diverges. The maximum total attempts is repeats*max_retries.

if ~exist('init_method', 'var'), init_method = 'pca'; end
if ~exist('max_retries', 'var'), max_retries = 10; end

fits = cell(1, repeats);
Qs = cell(1, repeats);

parfor r=1:repeats
    [fits{r}, Qs{r}] = loadOrRunFit(params, data, r, init_method, max_retries);
end

Q_endpoints = cellfun(@max, Qs);
[~, max_idx] = max(Q_endpoints);
best_params = fits{max_idx};
best_Q = Qs{max_idx};

end

function [fit, Q] = loadOrRunFit(params, data, idx, init_method, max_retries)
savename = sprintf('%s_H%03d_%s_run%03d.mat', params.dataset, params.H, init_method, idx);

if exist(savename, 'file')
    contents = load(savename);
    fit = contents.fit;
    Q = contents.Q;
else
    Q = nan;
    tries = 1;
    while any(isnan(Q))
        try
            init = EM.initialize(params, data, init_method);
            [fit, Q] = EM.run(init, data);
            break;
        catch
            warning('Run %d : attempt %d failed', idx, tries);
            tries = tries + 1;
            if tries > max_retries
                fit = params;
                fit.prior = nan;
                fit.sigma = nan;
                fit.G = nan(size(params.G));
                Q = nan;
                break;
            end
        end
    end
    save(savename, 'fit', 'Q');
end

end