%% Initialization
clear ; close all; clc

X = [0; 1 ; 2]
y = [2; 3; 4]
m = length(y) % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

X = [ones(m, 1), X(:,1)] % Add a column of ones to x

%% =================== Part 3: Cost and Gradient descent ===================

theta = zeros(2, 1) % initialize fitting parameters

% Some gradient descent settings
iterations = 500;
alpha = 0.5;

% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);