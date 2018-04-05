function log_prob_pr = compute_log_likelihood_pr(prms,stim,z_hidden)
log_prob_pr = log(prms.prior)*sum(z_hidden) + log(1-prms.prior)*sum((1-z_hidden)); 
end