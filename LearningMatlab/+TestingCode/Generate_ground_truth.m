function I = Generate_ground_truth(params)
% prior = 0.1;
% sigma = 1.0;
% Neurons = 10;
data_size = 500;
% pix = 5;
% G = rand(params.pix^2,Neurons); 
pr_array = ones(1,params.Neurons_hidden) * params.prior;
I = zeros(data_size,params.pix^2);
for i=1:data_size
    z_hidden = rand(1,params.Neurons_hidden)<pr_array;
    mu = params.G * z_hidden(:);
    sig = eye(params.pix^2) * params.sigma_stim^2;
    I(i,:) = mvnrnd(mu,sig);
    I(i,:) = (I(i,:) - mean(I(i,:)) )/ std(I(i,:));
end
end