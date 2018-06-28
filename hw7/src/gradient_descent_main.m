%
C = COST
learning_rate = LR
epsilon = EPS
xi = XI
training_method = METHOD
max_iterations = 100000000
% Load data
fprintf('======================= Loading Data ======================\n')
start_time = clock;
fprintf('Start: %d:%d:%d\n', start_time(4), start_time(5), start_time(6));
[train_Y, train_X] = libsvmread('../data/kddb');
end_time = clock;
fprintf('End: %d:%d:%d\n', end_time(4), end_time(5), end_time(6));
fprintf('=================== Loading Data Done =====================\n')
% Modify label to -1 and 1
train_Y = modify_label(train_Y);
% Fit
model = LogisticRegressionModel(C, learning_rate, epsilon, xi, training_method, max_iterations);
fprintf('======================== Training =========================\n')
model.fit(train_X, train_Y);




