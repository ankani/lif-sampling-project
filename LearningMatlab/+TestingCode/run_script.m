clear
params = LearningParams.ModelParams('patches_8x8_1000.h5');
I = TestingCode.Generate_ground_truth(params);
params2=params;
params2.sigma_stim = 1;
params2.prior = 0.8;
params2.G = randn(params.pix^2,params.Neurons_hidden);
params2.G = params2.G ./ sqrt(sum(params2.G.^2, 1));
params1 = LearningParams.run_algo(params2,I);