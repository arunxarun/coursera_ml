function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0.0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%for i = 1:m;
%    J = J+ ( theta(1)*X(i,1) + theta(2)*X(i,2)-y(i))^2;
%endfor;


%J = 1/(2*m)*J;


part = ((1/(2*m))*(X*theta-y).^2); 
sumJ = sum(part);

theta1ToN = theta;
theta1ToN(1) = 0;

reg = (lambda/(2*m))*(sum(theta1ToN.^2));

J = sumJ+reg;
grad = (1/m)*(X*theta - y)'*X + (lambda/m)*theta1ToN'; 



% =========================================================================

grad = grad(:);

end
