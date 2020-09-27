function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% add column of ones as the first column to X to represent x0 = 1
X = [ones(size(X, 1), 1) X];
Z2 = X * Theta1';
A2 = sigmoid(Z2);

% add column of ones as the first column to A2 to represent a0 = 1
A2 = [ones(size(A2, 1), 1) A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);
H = A3;

% transcode y from single digit to vector of 0 and 1
y = eye(num_labels)(y,:);

% calculate regularization (thetas should have their first column removed, since we are not regularizing 
regularization = lambda/(2*m) * (sum(sum((Theta1(:,2:end)).^2)) + sum(sum((Theta2(:,2:end)).^2)));

% calculate cost function
J = 1/m * sum(sum(-y .* log(H) - (1 - y) .* log(1 - H))) + regularization;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% for each training sample execute the below steps
sum_delta2 = zeros(size(Theta2, 1), size(Theta2, 2));
sum_delta1 = zeros(size(Theta1, 1), size(Theta1, 2));
for t=1 : m
  % STEP 1 - forward propagation
  % X already has added bias inputs (x0)
  a1 = X(t, :)';
  
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);

  % add column of ones as the first column to a2 to represent a0 = 1
  a2 = [1; a2];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  % STEP 2 - calculate error of the output layer
  % y is a vector of 0s and 1s, representing transcoded ouput values 
  delta3 = a3 - y(t, :)';
  
  % STEP 3 - calculate error of the hidden layer
  delta2 = Theta2(:,2:end)' * delta3 .* sigmoidGradient(z2);
  
  % STEP 4 - sum the errors for each layer
  sum_delta2 = sum_delta2 + delta3 * a2';
  sum_delta1 = sum_delta1 + delta2 * a1';
endfor

Theta1_grad = 1/m * sum_delta1;
Theta2_grad = 1/m * sum_delta2;
% regularize thetas
Theta1_grad += lambda/m * Theta1;
Theta2_grad += lambda/m * Theta2;
Theta1_grad(:,1) -= ((lambda/m) * (Theta1(:,1)));
Theta2_grad(:,1) -= ((lambda/m) * (Theta2(:,1)));


grad = [Theta1_grad(:) ; Theta2_grad(:)];

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
