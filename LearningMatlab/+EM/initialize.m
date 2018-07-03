function params = initialize(params, data)

% Ensure that params.pix matches size of data. If not, both params.pix and params.size must be
% fixed.
assert(params.pix == size(data, 1));

% When prior is too extreme, learning is slower. Initialize randomly between .001 and .1
if ~any(strcmpi('prior', params.fixed))
    log_lower = log(.0001);
    log_upper = log(.01);
    u = rand * (log_upper - log_lower) + log_lower;
    params.prior = exp(u);
end

scale = std(data(:));

% Initialize projective fields
if ~any(strcmpi('G', params.fixed))
    switch lower(params.init_method)
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
            if params.H > 2 * params.pix
                params.G(:, h+1:end) = randn(params.pix, params.H-2*params.pix) * scale;
            end
        case 'rand'
            params.G = randn(params.pix, params.H) * scale;
        case 'orban'
            orban_size = [16 16];
            contents = load('OrbanPFs.mat');
            A = contents.A;
            params.G = zeros(params.pix, params.H);
            h_idx = randperm(size(A, 2), params.H);
            for h=1:params.H
                scaled = imresize(reshape(A(:, h_idx(h)), orban_size), params.size, 'bicubic');
                params.G(:, h) = scaled(:);
            end
        otherwise
            error('Method must be one of ''pca,'' ''orban,'' or ''rand''');
    end
end

% Initialize pixel noise based on residuals from projective fields, times a factor of 10 because it
% helps convergence to over-estimate at the beginning
if ~any(strcmpi('sigma', params.fixed))
    bestfit = pinv(params.G) * data;
    errors = data - params.G * bestfit;
    sigma2 = .1 + mean(sum(errors.^2, 1));
    params.sigma = 10 * sqrt(sigma2);
end

end