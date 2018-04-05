% Code that returns the final values of the paramteres as params,
% where params.G will be RFs of neurons, params.sigma_stim will be pixelwise noise
% and params.prior will be prior probability of neurons firing
% This code is Parallelized for each data point in the batch
function params  = run_algo_test(params,I)
G_old = params.G;
sigma_stim_old = params.sigma_stim;
prior_old = params.prior;
G_grad_data = zeros(params.pix^2,params.Neurons_hidden);
sig_grad_data = 0.0;
pr_grad_data = 0.0;
iterations = 0;
data = params.data;
batch = params.batch;
params.data = I;
while iterations<=params.max_iter
    disp(iterations);
    params.eta = 0.01/(sqrt(iterations+1));
    for d_pt = 1:batch
        stim = data(d_pt,:);
        mu = LearningParams.variational_bayes(params,stim);
        [gr_G,gr_sig,gr_pr,norm_wt] = LearningParams.Each_Data(params,stim,mu);
        if isnan(gr_G)==0 & isinf(gr_G)==0
            G_grad_data = G_grad_data + gr_G / norm_wt;
        end
        if isnan(gr_sig)==0 && isinf(gr_sig)==0
            sig_grad_data = sig_grad_data + gr_sig / norm_wt;
        end
        if isnan(gr_pr)==0 && isinf(gr_pr)==0
            pr_grad_data = pr_grad_data + gr_pr / norm_wt;
        end
    end
    
    G_new = G_old - params.eta * G_grad_data;
    sigma_stim_new = sigma_stim_old - params.eta * sig_grad_data
    prior_new = prior_old - params.eta * pr_grad_data
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
    
end
end

