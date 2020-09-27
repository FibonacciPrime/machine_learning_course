function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add column of 1 to the input X to represent x0
X = [ones(size(X, 1), 1) X];
A1 = X * Theta1';
A1 = sigmoid(A1);

% add column of 1 to A1 to represent a0
A1 = [ones(size(A1, 1), 1) A1]; 
A2 = A1 * Theta2';
A2 = sigmoid(A2);

% for each row in A2, get the index of the highest value, and set prediction to this value
[max_values, p] = max(A2, [], 2);
p = mod(p, 10);






% =========================================================================


end
