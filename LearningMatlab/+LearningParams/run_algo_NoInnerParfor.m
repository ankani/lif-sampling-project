function params  = run_algo_NoInnerParfor(params)
G_old = params.G;
sigma_stim_old = params.sigma_stim;
prior_old = params.prior;
G_grad_data = zeros(params.pix^2,params.Neurons_hidden);
sig_grad_data = 0;
pr_grad_data = 0;
iterations = 0;
data = params.data;
N = params.Neurons_hidden;
while iterations<=params.max_iter
    disp(iterations);
    params.eta = 0.01/(sqrt(iterations+1));
    parfor d_pt = 1:params.batch
        stim = data(d_pt,:);
        mu = LearningParams.variational_bayes(params,stim);
        w_hidden = zeros(params.K_samples,1);
        gradient_G = zeros(params.pix^2,params.Neurons_hidden);
        gradient_sig = 0;
        gradient_pr = 0;
        for k = 1:params.K_samples
            z_hidden = binornd(1,transpose(mu),params.Neurons_hidden,1);
            w_hidden(k) = LearningParams.prob(params,stim,z_hidden)/LearningParams.q_compute(params,z_hidden,mu);
            G,s,pr = LearningParams.compute_gradients_elbo(params,stim,z_hidden);
            gradient_G = gradient_G + G * w_hidden(k);
            gradient_sig = gradient_sig + s * w_hidden(k);
            gradient_pr = gradient_pr + pr * w_hidden(k);
        end
        norm_wt = sum(w_hidden);
%         [gr_G,gr_sig,gr_pr,norm_wt] = LearningParams.Each_Data(params,mu)
        G_grad_data = G_grad_data + gradient_G / norm_wt;
        sig_grad_data = sig_grad_data + gradient_sig / norm_wt;
        pr_grad_data = pr_grad_data + gradient_pr / norm_wt;
    end
    G_new = G_old - params.eta * G_grad_data;
    sigma_stim_new = LearningParams.Proj_sig(sigma_stim_old - params.eta * sig_grad_data);
    prior_new = LearningParams.Proj_pr(prior_old - params.eta * pr_grad_data);
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

