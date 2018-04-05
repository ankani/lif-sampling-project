% CODE TO TEST THE WORKING OF THE VB GRADIENTS. WE TRY TO SEE IF
% LOG Q APPROXIMATELY EQUAL TO (LOG Q(PARAMS + DELTA) - LOG Q) / DELTA
clear;
params = LearningParams.ModelParams('patches_8x8_100.h5');
trials = 100;
[nimages, ~] = size(params.data);
index = randi(nimages,1,trials);

G_slope = zeros(trials,params.Neurons_hidden,params.pix^2*params.Neurons_hidden);
sig_slope = zeros(trials,params.Neurons_hidden);
pr_slope = zeros(trials,params.Neurons_hidden);
G_grad_comp = zeros(trials,params.Neurons_hidden,params.pix^2*params.Neurons_hidden);
sig_grad_comp = zeros(trials,params.Neurons_hidden);
pr_grad_comp = zeros(trials,params.Neurons_hidden);
for t=1:trials
    disp(t)
    stim = params.data(index(t),:);
    epsilon = 0.001;
    mu_vb = LearningParams.variational_bayes(params,stim);
    [~,~,~,G_grad,sig_grad,pr_grad] = LearningParams.compute_gradients_vb(params,mu_vb,stim,zeros(1,params.Neurons_hidden));
    
    G_grad_comp(t,:,:) = G_grad;
    sig_grad_comp(t,:) = sig_grad;
    pr_grad_comp(t,:) = pr_grad;
    idx = 1;
    for i=1:params.Neurons_hidden
        for p=1:params.pix^2
            params_G = params;
            params_G.G(p,i) = params.G(p,i) + epsilon;
            mu_vb_plus1 = LearningParams.variational_bayes(params_G,stim);
            params_G = params;
            params_G.G(p,i) = params.G(p,i) - epsilon;
            mu_vb_minus1 = LearningParams.variational_bayes(params_G,stim);
            G_slope(t,:,idx) = (mu_vb_plus1 - mu_vb_minus1) / (2*epsilon);
            idx = idx + 1;
        end
    end
    
    params_sig = params;
    params_sig.sigma_stim = params.sigma_stim + epsilon;
    mu_vb_plus2 = LearningParams.variational_bayes(params_sig,stim);
    params_sig = params;
    params_sig.sigma_stim = params.sigma_stim - epsilon;
    mu_vb_minus2 = LearningParams.variational_bayes(params_sig,stim);
    sig_slope(t,:) = (mu_vb_plus2 - mu_vb_minus2) / (2*epsilon);
    
    params_pr = params;
    params_pr.prior = params.prior + epsilon;
    mu_vb_plus3 = LearningParams.variational_bayes(params_pr,stim);
    params_pr = params;
    params_pr.prior = params.prior - epsilon;
    mu_vb_minus3 = LearningParams.variational_bayes(params_pr,stim);
    pr_slope(t,:) = (mu_vb_plus3 - mu_vb_minus3) / (2*epsilon);
end
subplot(1,3,1);
cla;
X = reshape(G_slope,1,trials*params.pix^2*params.Neurons_hidden^2);
Y = reshape(G_grad_comp,1,trials*params.pix^2*params.Neurons_hidden^2);
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
subplot(1,3,2);
cla;
X1 = reshape(sig_slope,1, trials*params.Neurons_hidden);
Y1 = reshape(sig_grad_comp, 1, trials*params.Neurons_hidden);
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
subplot(1,3,3);
cla;
X2 = reshape(pr_slope, 1, trials*params.Neurons_hidden);
Y2 = reshape(pr_grad_comp, 1, trials*params.Neurons_hidden);
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