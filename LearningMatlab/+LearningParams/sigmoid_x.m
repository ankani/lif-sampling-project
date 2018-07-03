% function to compute the sigmoid of any value
function result = sigmoid_x(x)
if x > 36
    result = 1-eps;
elseif x < -36
    result = eps;
else
    result = (1.0)/(1.0 + exp(-x));
end
end