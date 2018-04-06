function data = generate(params)
%EM.GENERATE sample data from the prior. Result is size [pix x N].

z = rand(params.H, params.N) < params.prior;
data = params.G * z + randn(params.pix, params.N) * params.sigma;

end