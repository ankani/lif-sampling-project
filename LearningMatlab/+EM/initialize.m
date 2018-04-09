function params = initialize(params, data, method)

% Ensure that params.pix matches size of data. If not, both params.pix and params.size must be
% fixed.
assert(params.pix == size(data, 1));

% When prior is too extreme, learning is slower. Initialize randomly between .2 and .8
params.prior = .2 + rand * 6;

scale = std(data(:));

% Initialize projective fields
switch method
    case 'pca'
        params.G = zeros(params.pix, params.H);
        C = cov(data');
        [V, D] = eig(C);
        [~, sort_idx] = sort(diag(D), 'descend');
        V = V(:, sort_idx);
        % Note: because latents are binary and cannot be negative, each PC is added twice, once with
        % each sign.
        for h=1:min(params.H, 2*params.pix)
            eig_idx = floor((h-1)/2) + 1;
            sgn = sign(mod(h, 2) - .5);
            params.G(:, h) = V(:, eig_idx) * sgn * scale;
        end
        
        % If very overcomplete, fill in remainder randomly
        if params.H > params.pix
            params.G(:, h+1:end) = randn(params.pix, params.H-2*params.pix) * scale;
        end
    case 'rand'
        params.G = randn(params.pix, params.H) * scale;
    otherwise
        error('Method must be one of ''pca'' or ''rand''');
end

% Initialize pixel noise based on residuals from projective fields
bestfit = pinv(params.G) * data;
errors = data - params.G * bestfit;
sigma2 = mean(sum(errors.^2, 1));
params.sigma = sqrt(sigma2);

end