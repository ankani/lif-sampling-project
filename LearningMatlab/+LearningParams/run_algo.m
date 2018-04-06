% Code that returns the final values of the paramteres as params,
% where params.G will be RFs of neurons, params.sigma_stim will be pixelwise noise
% and params.prior will be prior probability of neurons firing
% This code is Parallelized for each data point in the batch
function params  = run_algo(params,varargin)

sigma_stim_old = params.sigma_stim;
prior_old = params.prior;
G_grad_data = zeros(params.pix^2,params.Neurons_hidden);
sig_grad_data = 0.0;
pr_grad_data = 0.0;
iterations = 1;
if ~isempty(varargin)
    data = varargin{1};
else
    data = params.data;
end
% params.G = repmat(mean(data)',1,params.Neurons_hidden);
G_old = params.G;
batch = params.batch;
stim_fix = data(10,:);
elbo_track = [];
while iterations<=params.max_iter
    disp(iterations);
    params.eta = 0.001/(sqrt(iterations));
    for d_pt = 1:5
        stim = data(d_pt,:);
        mu = LearningParams.variational_bayes(params,stim);
        mu(mu<0.001) = 0.001;
        if sum(isnan(mu))>0
            keyboard
        end
        [gr_G,gr_sig,gr_pr,G_vb_avg,s_vb_avg,pr_vb_avg,norm_wt,~] = LearningParams.Each_Data(params,stim,mu);
        if isnan(gr_G)==0 & isinf(gr_G)==0
            G_grad_data = G_grad_data + gr_G / norm_wt + log(norm_wt / params.K_samples) * G_vb_avg;
        end
        if isnan(gr_sig)==0 && isinf(gr_sig)==0
            sig_grad_data = sig_grad_data + gr_sig / norm_wt + log(norm_wt / params.K_samples) * s_vb_avg;
        end
        if isnan(gr_pr)==0 && isinf(gr_pr)==0
            pr_grad_data = pr_grad_data + gr_pr / norm_wt + log(norm_wt / params.K_samples) * pr_vb_avg;
        end
    end
    
    G_new = G_old + params.eta * G_grad_data;
    sigma_stim_new = sigma_stim_old + params.eta * sig_grad_data;
    prior_new = (prior_old + params.eta * pr_grad_data);
    params.G = G_new;
    params.sigma_stim = sigma_stim_new;
    params.prior = prior_new;
    if ((norm(G_old-G_new)<=params.tol) && (abs(sigma_stim_new-sigma_stim_old)<=params.tol)) && ((abs(prior_new-prior_new)<=params.tol))
        disp('Tolerence Satisfied');
        break;
    end
    G_old = G_new;
    sigma_stim_old = sigma_stim_new;
    prior_old = prior_new;
    iterations = iterations + 1;
    if iterations>params.max_iter
        disp('Maximum Iteration Reached');
    end
    
    
    s = 0;
    mu_test = LearningParams.variational_bayes(params,stim_fix);
    for t=1:params.K_samples
        z = binornd(1,mu_test', params.Neurons_hidden,1);
        s = s + log(LearningParams.prob(params,stim_fix,z)/LearningParams.q_compute(params,z,mu_test));
    end
    s = s / params.K_samples;
    elbo_track(end+1) = s;
    
end
plot((1:length(elbo_track)),elbo_track,'o-');
disp(elbo_track)

end

