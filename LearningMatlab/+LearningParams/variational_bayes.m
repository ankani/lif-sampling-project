% function which returns mean field probability of firing of each neuron given a data (image patch)
function mu = variational_bayes(params,stim,epsilon,itrs)
if nargin < 3, epsilon = 1e-9; end
if nargin < 4, itrs = 1000; end

stim = stim(:);
feedforward = stim' * params.G;
R = -(params.G' * params.G);
mu = params.prior * ones(1,params.Neurons_hidden);
log_prior = log(params.prior) - log(1.0-params.prior);

diff = inf;
i = 1;
while i < itrs && diff > epsilon
    mu_prev = mu;
    for j = 1:(params.Neurons_hidden)
        mu1 = mu;
        mu1(j) = 0.0;
        x = (feedforward(j) + R(j,j)/2 + (mu1*R(:,j)))/(params.sigma_stim^2) + log_prior;
        mu(j) = LearningParams.sigmoid_x(x);
    end
    diff = norm(mu - mu_prev);
    i = i + 1;
end

if i >= itrs
    warning('Max VB iterations reached before convergence');
end
end