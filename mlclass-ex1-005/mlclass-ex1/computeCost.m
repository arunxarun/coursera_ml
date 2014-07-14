function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta is the thing you are minimizing.
% X array contains values for constant(1) + X value
% y vector contains y value

for i = 1:m;
    J += ( theta(1)*X(i,1) + theta(2)*X(i,2)-y(i))^2;
endfor;


J = 1/(2*m)*J;

return;
% =========================================================================

end
