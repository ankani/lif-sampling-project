function log_prob_G = compute_log_likelihood_G(prms,stim,z_hidden)
% log_prob_sig = log(1.0/(2.0^(prms.pix*prms.pix/2.0)*pi^(prms.pix*prms.pix/2.0)*prms.sigma_stim^(prms.pix*prms.pix))) - ((1.0/(2.0*prms.sigma_stim^2)) * norm(stim(:) - prms.G*z_hidden(:))^2) + log(prms.prior)*sum(z_hidden) + log(1-prms.prior)*sum((1-z_hidden)); 
log_prob_G =  - ((1.0/(2.0*prms.sigma_stim^2)) * norm(stim(:) - prms.G*z_hidden(:))^2); 

end