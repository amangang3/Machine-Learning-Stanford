function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%cost function
h = sigmoid(X*theta);
%regularization
regularization_term = lambda/(2*m) * sum ((theta(2:end)).^2);
addition_term = sum(-y.*log(h) - (1-y) .* log(1-h));
J = (1/m * addition_term) + regularization_term;

%gradient
addition_term = sum((h-y).*X);
regularization_term = lambda/m .* theta(2:end);
grad = 1/m * addition_term;
transpose_reg = transpose(regularization_term);
grad(2:end) = grad(2:end) + transpose_reg;

% =============================================================
end
