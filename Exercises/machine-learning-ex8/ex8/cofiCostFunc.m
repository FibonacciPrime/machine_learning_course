function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% =============== COST FUNCTION =====================
JSum = 0;

% this is unvectorized implementation of cost function
%for (i = 1:num_movies)
%  for (j = 1:num_users)
%    if (R(i,j) == 0) 
%      continue
%    endif
%    JSum += (Theta(j, :) * X(i, :)' - Y(i,j))^2;
%  endfor
%endfor

% vectorized implementation of cost function
Diff = X * Theta' - Y;
JSum = sum(sum((Diff .* R).^2));
J = 1/2 * JSum;

% regularize cost function
ThetaReg = lambda/2 * sum(sum(Theta .^2));
XReg = lambda/2 * sum(sum(X .^2));

J = J + ThetaReg + XReg;

% =============== GRADIENT =====================

% gradients for movies (X)
%for (i = 1:num_movies)
%  for (k = 1:num_features)
%    XKgrad = 0;
%    for (j = 1:num_users)
%      if (R(i,j) == 0) 
%        continue
%      endif
%      XKgrad += (Theta(j, :) * X(i, :)' - Y(i,j)) * Theta(j, k);
%    endfor
%    X_grad(i,k) = XKgrad + (lambda * X(i,k));
%  endfor
%endfor
%
%% gradients for users (Theta)
%for (j = 1:num_users)
%  for (k = 1:num_features)
%    ThetaKgrad = 0;
%    for (i = 1:num_movies)
%      if (R(i,j) == 0) 
%        continue
%      endif
%      ThetaKgrad += (Theta(j, :) * X(i, :)' - Y(i,j)) * X(i, k);
%    endfor
%    Theta_grad(j,k) = ThetaKgrad + (lambda * Theta(j,k));
%  endfor
%endfor

% vectorized implementation of gradient

for (i=1:num_movies)
  idx = find(R(i, :)==1);   % indexes of users which have rated the movie i
  Ytemp = Y(i, idx);        % all ratings movie i has gotten
  Thetatemp = Theta(idx, :);    % all features of users which rated the movie i
  X_grad(i,:) = (X(i,:) * Thetatemp' - Ytemp) * Thetatemp + (lambda * X(i, :));
endfor

for (j=1:num_users)
  idx = find(R(:, j)==1);   % indexes of all movies user j has ranked
  Ytemp = Y(idx, j);        % all ratings user j has given
  Xtemp = X(idx, :);        % features of all movies user j has raned
  Theta_grad(j,:) = (Xtemp * Theta(j,:)' - Ytemp)' * Xtemp  + (lambda * Theta(j, :));
endfor

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
