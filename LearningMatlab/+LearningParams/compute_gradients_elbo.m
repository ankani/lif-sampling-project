% Computes the gradients of log (p/q) w.r.t RFs (G), pixelwise noise (sigma_stim) and Prior probability of spiking (prior)
function [G_grad,sig_grad,pr_grad,G_grad_vb, sig_grad_vb, pr_grad_vb,det_zero] = compute_gradients_elbo(params,mu_vb,stim,z_hidden)
stim = stim(:);
z = z_hidden(:);
diff = stim - params.G * z;

% Gradients of log p
dlogP_dG = diff * z' / params.sigma_stim^2;
dlogP_dsig = diff' * diff / params.sigma_stim^3 - params.pix^2 / params.sigma_stim;
dlogP_dpr = sum(z) / params.prior - sum(1 - z) / (1 - params.prior);

% Gradients of log q
[G_grad_vb, sig_grad_vb, pr_grad_vb, det_zero] = LearningParams.compute_gradients_vb(params,mu_vb,stim,z_hidden);

% ELBO gradients are difference of logP and logQ gradients
G_grad = dlogP_dG - G_grad_vb;
sig_grad = dlogP_dsig - sig_grad_vb;
pr_grad = dlogP_dpr - pr_grad_vb;
end