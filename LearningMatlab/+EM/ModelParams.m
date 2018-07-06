function [params, data] = ModelParams(varargin)
%EM.MODELPARAMS construct parameterss structure
%
% params = EM.MODELPARAMS('key1', val1, 'key2', val2, ...) create params with defaults and override
% any specified 'key' with the given value.

%% Set defaults
params = struct(...
    ... % --- Model parameters ---
    'H', 16, ... % number of hidden units/neurons
    'sigma', .1, ... % pixelwise noise
    'prior',  .001, ... % prior probability of spiking for neurons
    'size', [8 8], ... % [height width] of images
    'fixed', {{}}, ... % Which, if any, parameters are 'fixed' and should not change
    ... % --- Data parameters ---
    'dataset', 'bars', ... % One of 'bars', 'rand', or a path to a '.h5' file generated from extract_vanhateren_to_bsc.py
    'N', 1000, ... % number of samples to generate (this is overwritten if an h5 file is given)
    ... % --- EM parameters ---
    'init_method', 'pca', ... % 'method' argument to initialization
    'anneal_init', 100, ... % Initial temperature for annealing. A value of 1 does no annealing
    'anneal_decay', .5, ... % Every epoch, decay temperature towards 1 at this rate
    'debug', false, ... % flag whether to print / plot diagnostic information as EM is running
    'truncate', 10, ... % expectation truncation with up to 'truncate' ones in any state
    'tvi_samples', 512, ... % number of new samples to propose in each step of TVI (not used in EM)
    'tvi_propose', 100, ... % in TVI, the number of new samples to propose each iteration
    'tol',  1e-5, ... % tolerance of change in parameters defining convergence
    'max_iter',  300); % maximum number of iterations allowed for convergence

%% Parse any extra (..., 'key', value, ...) pairs passed in through varargin.
for val_idx = 2:2:length(varargin)
    key = varargin{val_idx-1};
    if ~ischar(key)
        warning('invalid input to newSamplingParams. All arguments should be (..., ''key'', value, ...)');
    elseif ~isfield(params, key)
        warning('unrecognized sampling parameter: ''%s''', key);
    else
        params.(key) = varargin{val_idx};
    end
end

%% Set some derived parameters
params.pix = prod(params.size);

%% Load or generate data

switch params.dataset
    case {'bars'}
        if params.H > sum(params.size)
            warning('''bars'' data with more latents than total rows + columns will result in multiple copies of the same bar');
        end
        
        params.G = zeros(params.pix, params.H);
        for i=1:params.H
            patch = zeros(params.size);
            idx = 1 + mod(i-1, sum(params.size));
            if idx <= params.size(1)
                % row bar
                patch(idx, :) = 1;
            else
                % column bar
                col = idx - params.size(1);
                patch(:, col) = 1;
            end
            
            params.G(:, i) = patch(:);
        end
        
        data = EM.generate(params);
    case {'rand'}
        params.G = randn(params.pix, params.H);
        data = EM.generate(params);
    otherwise
        if ~exist(params.dataset, 'file')
            error('expected ''params.dataset'' to be ''bars'', ''rand'', or a path to an ''.h5'' file containing image patches');
        end
        
        % Load data from h5 file and reverse order of dimensions to flip memory layout convention
        data = permute(h5read(params.dataset, '/patches'), [3 2 1]);
        [n, h, w] = size(data);
        data = reshape(data, n, h*w)';
        
        % Preprocess data: normalize luminance range.
        data = data / std(data(:));
        
        % Override params
        params.N = n;
        params.size = [h w];
        params.pix = h * w;
        % Initialize projective fields
        params = EM.initialize(params, data);
end

%% Sanity checks

if params.tvi_samples > 2^params.H
    warning('Limiting params.tvi_samples to 2^params.H');
    params.tvi_samples = 2^params.H;
end

if params.truncate > params.H
    warning('Limiting params.truncate to params.H');
    params.tvi_samples = params.H;
end

end