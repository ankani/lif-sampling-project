function [G_grad_conj,sig_grad_conj,pr_grad_conj] = compute_gradients_conjugate(params,z_hidden,mu,stim)
log_q = TestingCode.compute_log_q(params,z_hidden,mu);
log_p = TestingCode.compute_log_p(params,stim,z_hidden);
[G_grad_vb, sig_grad_vb, pr_grad_vb] = TestingCode.compute_gradients_vb_test(params,mu,stim,z_hidden);
G_grad_conj = (log_p - log_q)*G_grad_vb;
sig_grad_conj = (log_p - log_q)*sig_grad_vb;
pr_grad_conj = (log_p - log_q)*pr_grad_vb;
end