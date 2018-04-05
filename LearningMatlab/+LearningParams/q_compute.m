% computes the proposal distribution value in this case mu_i^z_hidden(i) * (1 - mu_i)^(1 - z_hidden(i)) 
% where mu_i has been obtained from mean field solution VB, variational_bayes() function
function p = q_compute(params,z_hidden,pr)
p = 1.0;
N = params.Neurons_hidden;
for i = 1:N
    p = p * (pr(i))^(z_hidden(i)) * (1.0 - pr(i))^(1 - z_hidden(i)) ;
end
end