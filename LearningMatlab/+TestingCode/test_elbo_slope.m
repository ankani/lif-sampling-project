% CODE TO TEST THE WORKING OF THE VB GRADIENTS. WE TRY TO SEE IF
% LOG Loss_Function APPROXIMATELY EQUAL TO (LOG Loss_Function(PARAMS + DELTA) - LOG Loss_Function) / DELTA
clear;
params = LearningParams.ModelParams('patches_8x8_100.h5');
trials = 100;
[nimages, ~] = size(params.data);
index = randi(nimages,1,trials);

G_slope = zeros(trials,params.pix^2,params.Neurons_hidden);
sig_slope = zeros(trials,1);
pr_slope = zeros(trials,1);
G_grad_comp = zeros(trials,params.pix^2,params.Neurons_hidden);
sig_grad_comp = zeros(trials,1);
pr_grad_comp = zeros(trials,1);
for i=1:trials
    disp(i)
    stim = params.data(index(i),:);
    epsilon = 0.001;
    mu_vb = LearningParams.variational_bayes(params,stim);
    z_hidden = rand(1,params.Neurons_hidden) < mu_vb;
    [G_grad,sig_grad,pr_grad] = LearningParams.compute_gradients_elbo(params,mu_vb,stim,z_hidden);
    
    G_grad_comp(i,:,:) = G_grad;
    sig_grad_comp(i) = sig_grad;
    pr_grad_comp(i) = pr_grad;
    
    for k=1:params.Neurons_hidden
        for j=1:params.pix^2
            params_G = params;
            params_G.G(j,k) = params.G(j,k) + epsilon;
            log_prob = LearningParams.compute_log_p(params_G,stim,z_hidden);
            mu_vb_G_plus = LearningParams.variational_bayes(params_G,stim);
            tot_log_prob_G_plus = log_prob - sum(z_hidden(:) .* log(mu_vb_G_plus(:)) + (1 - z_hidden(:)) .* log(1-mu_vb_G_plus(:)));
            
            params_G = params;
            params_G.G(j,k) = params.G(j,k) - epsilon;
            mu_vb_G_minus = LearningParams.variational_bayes(params_G,stim);
            log_prob = LearningParams.compute_log_p(params_G,stim,z_hidden);
            tot_log_prob_G_minus = log_prob - sum(z_hidden(:) .* log(mu_vb_G_minus(:)) + (1 - z_hidden(:)) .* log(1-mu_vb_G_minus(:)));
            
            G_slope(i,j,k) = (tot_log_prob_G_plus - tot_log_prob_G_minus)/(2*epsilon);
        end
    end
    
    params_sig = params;
    params_sig.sigma_stim = params.sigma_stim + epsilon;
    log_prob = LearningParams.compute_log_p(params_sig,stim,z_hidden);
    mu_vb_sig_plus = LearningParams.variational_bayes(params_sig,stim);
    
    tot_log_prob_sig_plus = log_prob - sum(z_hidden .* log(mu_vb_sig_plus) + (1-z_hidden) .* log(1-mu_vb_sig_plus));
    params_sig = params;
    params_sig.sigma_stim = params.sigma_stim - epsilon;
    log_prob = LearningParams.compute_log_p(params_sig,stim,z_hidden);
    mu_vb_sig_minus = LearningParams.variational_bayes(params_sig,stim);
    
    tot_log_prob_sig_minus = log_prob - sum(z_hidden .* log(mu_vb_sig_minus) + (1-z_hidden) .* log(1-mu_vb_sig_minus));
    sig_slope(i) = (tot_log_prob_sig_plus - tot_log_prob_sig_minus)/(2*epsilon);
    
    
    params_prior = params;
    params_prior.prior = params.prior + epsilon;
    log_prob = LearningParams.compute_log_p(params_prior,stim,z_hidden);
    mu_vb_prior_plus = LearningParams.variational_bayes(params_prior,stim);
    tot_log_prob_prior_plus = log_prob - sum(z_hidden(:) .* log(mu_vb_prior_plus(:)) + (1 - z_hidden(:)) .* log(1-mu_vb_prior_plus(:)));
    params_prior = params;
    params_prior.prior = params.prior - epsilon;
    log_prob = LearningParams.compute_log_p(params_prior,stim,z_hidden);
    mu_vb_prior_minus = LearningParams.variational_bayes(params_prior,stim);
    tot_log_prob_prior_minus = log_prob - sum(z_hidden(:) .* log(mu_vb_prior_minus(:)) + (1 - z_hidden(:)) .* log(1-mu_vb_prior_minus(:)));
    pr_slope(i) = (tot_log_prob_prior_plus - tot_log_prob_prior_minus)/(2*epsilon);
    
    
end

subplot(1,3,1)
X = reshape(G_slope,1,trials*params.pix^2*params.Neurons_hidden);
Y = reshape(G_grad_comp,1,trials*params.pix^2*params.Neurons_hidden);
scatter(X,Y);
axis tight
hold on;
mn = min(min(X),min(Y));
mx = max(max(X),max(Y));
r = linspace(mn,mx,100);
plot(r,r);
ylabel('VB Code grad');
xlabel('Computed grad');
title('G');
subplot(1,3,2)
X1 = reshape(sig_slope,1, trials);
Y1 = reshape(sig_grad_comp, 1, trials);
scatter(X1,Y1);
axis tight
hold on;
mn1 = min(min(X1),min(Y1));
mx1 = max(max(X1),max(Y1));
r1 = linspace(mn1,mx1,100);
plot(r1,r1);
ylabel('VB Code grad');
xlabel('Computed grad');
title('Sigma');
subplot(1,3,3)
X2 = reshape(pr_slope, 1, trials);
Y2 = reshape(pr_grad_comp, 1, trials);
scatter(X2,Y2);
axis tight
hold on;
mn2 = min(min(X2),min(Y2));
mx2 = max(max(X2),max(Y2));
r2 = linspace(mn2,mx2,100);
plot(r2,r2);
ylabel('VB Code grad');
xlabel('Computed grad');
title('Prior');



