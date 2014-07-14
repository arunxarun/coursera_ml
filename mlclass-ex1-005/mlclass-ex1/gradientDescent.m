function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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


    theta0 = theta(1);
    theta1 = theta(2);
    
    %calculate d/dtheta0
    dxCost0 = 0;
    for i = 1:m;
        dxCost0 += (theta0*X(i,1) + theta1*X(i,2)-y(i));
    endfor;
    
    newTheta0 = theta0 - alpha*(1/m)*dxCost0; 
    
    %calculate d/dtheta1
    dxCost1 = 0;

    for i = 1:m;
        dxCost1 +=  (theta0*X(i,1) + theta1*X(i,2)-y(i))*X(i,2); % this is a pure scalar op because dxCost1 is a scalar. X(i) here is the X value in the array
    endfor;
    
    newTheta1 = theta1 - alpha*(1/m)*dxCost1; 
    % reset theta in one shot.
    theta(1) = newTheta0;
    theta(2) = newTheta1;

    %fprintf("theta = %f, theta2 = %f\n",theta(1),theta(2));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
