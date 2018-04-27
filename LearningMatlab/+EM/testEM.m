%% Construct ground truth params and data for 'bars' test

[ground_truth, data] = EM.ModelParams('dataset', 'bars', 'H', 10, 'debug', true);

%% Initialize parameters to be learned

params = ground_truth;
params.sigma = 3;
params.prior = .5;
params.fixed = {'G'};
% params.G = randn(size(ground_truth.G)) * std(ground_truth.G(:));

%% Run EM

[fit, Q, updates] = EM.run(params, data);
iters = length(Q);

%% Plot

figure;

% Q should have increased monotonically
subplot(1, 3, 1);
plot(Q);
xlabel('iteration');
ylabel('Q');

% Visualize trajectory of sigma and prior over time
subplot(1, 3, 2);
hold on;
plot([1 iters], [ground_truth.sigma ground_truth.sigma], '--k');
plot([updates.sigma]);
xlabel('iteration');
ylabel('sigma');

subplot(1, 3, 3);
hold on;
plot([1 iters], [ground_truth.prior ground_truth.prior], '--k');
plot([updates.prior]);
xlabel('iteration');
ylabel('prior');

% Visualize learned projective fields
figure;
m = round(sqrt(params.H));
n = ceil(params.H / m);
for i=1:params.H
    subplot(m, n, i);
    imagesc(reshape(fit.G(:, i), fit.size));
    axis image;
    set(gca, 'XTick', [], 'YTick', []);
end