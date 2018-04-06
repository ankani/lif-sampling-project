function params = ModelParams(filename,varargin)
%LearningParams.ModelParams construct params struct for the sampling model.
%
% params = LearningParams.MODELPARAMS('key1', val1, 'key2', val2, ...) create params
% with defaults and override any specified 'key' with the given value. For
% example, 'params = LearningParams.ModelParams('Neurons_hidden', 10)' creates a struct
% with all default params (e.g. params.K_samples is 10), but with
% params.Neurons_hidden set to 10.
%
params = struct(...
    'save_dir', fullfile('+LearningParams', 'saved results'), ...
    'Neurons_hidden' , 8,... % number of hidden units/neurons
    'K_samples' , 20,... % number of samples for the Loss function of IWAE
	'sigma_stim' ,0.5 ,... % initialize pixelwise noise(sigma)
	'prior' , .2,... % initialize the prior probability of spiking for neurons
	'batch' , 1000,... % size of batch of data to be trained on
    'pix' , 8,... % each RF and natural image patch is pix x pix size
    'tol' , 0.1,... % convergence of learning tolerence
    'max_iter' , 50,... % maximum number of iterations allowed for convergence
    'eta' , 0.5); % learning rate, used while updating parameters
    %params.G = randn(params.pix^2,params.Neurons_hidden); % initialize RFs, here size (pix*pix) x Neurons_hidden
    % opening data file containg patches from natural images
    data = h5read(filename,'/patches');
    data = reshape(data,params.pix^2,params.batch); % data containing patches on which learning will be done
    params.data = transpose(data);
for i=1:params.Neurons_hidden
    tmp=zeros(8,8);
    tmp(:,i)=1;
    params.G(:,i)=tmp(:);
end
params.G = params.G ./ sqrt(sum(params.G.^2, 1));
% Parse any extra (..., 'key', value, ...) pairs passed in through
% varargin.
for val_idx=3:2:length(varargin)
    key = varargin{val_idx-1};
    if ~ischar(key)
        warning('invalid input to newSamplingParams. All arguments should be (..., ''key'', value, ...)');
    elseif ~isfield(params, key)
        warning('unrecognized sampling parameter: ''%s''', key);
    else
        params.(key) = varargin{val_idx};
    end
end
end