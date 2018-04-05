function log_q = compute_log_q(params,z_hidden,pr)
log_q = log(1.0);
N = params.Neurons_hidden;
for i = 1:N
   log_q = log_q + (z_hidden(i)) * log(pr(i)) + (1 - z_hidden(i)) * log(1.0 - pr(i)) ;
end

end