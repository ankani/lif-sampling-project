% computes the joint probability of z_hidden and data under the current parameter values
function y = prob(params,stim,z_hidden)
p = LearningParams.q_compute(params,z_hidden,params.prior*ones(params.Neurons_hidden)); 
temp = (params.G * (z_hidden));
y = exp(-1.0/(2.0*(params.sigma_stim^2)) * (norm(stim - temp))^2)*p;
end        