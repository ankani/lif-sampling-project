function z = truncate(params, stim)
%EM.TRUNCATE enumerate only the most likely values of z.

trunc = min(params.truncate, params.H);

marginals = EM.variational_bayes(params, stim);
[~, ordering] = sort(marginals, 'descend');
topK = ordering(1:trunc);
bottomK = ordering(trunc+1:end);

% Compute what output size will be and pre-allocate. The top-k values will be fully enumerated and
% have 2^trunc states. Other states only included as one-hot so there are exactly as many 'other'
% states as the number of bottom-k latents.
num_top_K = 2^trunc;
num_other = length(bottomK);
total = num_top_K + num_other;
z = zeros(total, params.H);

% Use all combinations of the top-K marginal probability latents. This includes the state with
% all-zeros
z(1:num_top_K, topK) = EM.enumerate(trunc);

% Also include all states where any single z is on
z(num_top_K+1:end, bottomK) = eye(params.H - trunc);

end