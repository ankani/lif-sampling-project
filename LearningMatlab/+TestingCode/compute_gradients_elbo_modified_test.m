% Computes the gradients of log (p/q) w.r.t RFs (G), pixelwise noise (sigma_stim) and Prior probability of spiking (prior)
function [G_grad,sig_grad,pr_grad] = compute_gradients_elbo_modified_test(params,mu_vb,stim,z_hidden)
temp1 = (transpose(stim) - (params.G*z_hidden(:)));
temp2 = z_hidden;
[G_grad_vb, sig_grad_vb, pr_grad_vb] = LearningParams.compute_gradients_vb(params,mu_vb,stim,z_hidden);
[G_grad_conj,sig_grad_conj,pr_grad_conj] = LearningParams.compute_gradients_conjugate(params,z_hidden,mu_vb,stim);
G_grad = (1.0/params.sigma_stim^2) * (temp1*temp2) - G_grad_vb + G_grad_conj;
sig_grad = (1.0/params.sigma_stim^3) * norm(transpose(stim) - ((params.G)*z_hidden(:)))^2 - params.pix^2/(params.sigma_stim) - sig_grad_vb + sig_grad_conj;
pr_grad = sum(z_hidden)*(1.0/params.prior)-(sum(1- z_hidden))*(1.0/(1.0-params.prior))- pr_grad_vb + pr_grad_conj;
end