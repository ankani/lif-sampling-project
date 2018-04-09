function mu = variational_bayes(params, stim, epsilon, max_itrs)
%variational_bayes get mean-field VB estimate of marginal probabilities for each latent

if nargin < 3, epsilon = 1e-9; end
if nargin < 4, max_itrs = 1000; end

stim = stim(:);
feedforward = stim' * params.G;
R = -(params.G' * params.G);
mu = params.prior * ones(1,params.H);
log_prior = log(params.prior) - log(1.0-params.prior);

diff = 1e10;
i = 1;
while i < max_itrs && diff > epsilon
    mu_prev = mu;
    for j = 1:(params.H)
        mu1 = mu;
        mu1(j) = 0.0;
        x = (feedforward(j) + R(j,j)/2 + mu1*R(:,j))/(params.sigma^2) + log_prior;
        mu(j) = LearningParams.sigmoid_x(x);
    end
    diff = norm(mu - mu_prev);
    i = i + 1;
end

if i >= max_itrs
    warning('Max VB iterations reached before convergence');
end
end