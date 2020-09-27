function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    H = X * theta;
    theta_new = zeros(length(theta), 1);
    for theta_index = 1:length(theta)
        theta_new(theta_index) = theta(theta_index) - alpha/m * sum((H - y) .* X(:, theta_index));
    end
    
    %theta1_new = theta(1) - alpha/m * sum(H - y);
    %theta2_new = theta(2) - alpha/m * sum((H .* X(:, 2)) - (y .* X(:, 2)));

    % ============================================================
    
    % update theta
    theta = theta_new; 
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end;

end;
