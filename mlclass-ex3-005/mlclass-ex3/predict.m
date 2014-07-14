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

% X = 5000x400 -- training examples. 
% hidden layer 1 = 400x25
% Theta1 = 25x401
% add the bias value to X

X = [ones(m, 1) X]; 
A = sigmoid(X*Theta1'); % this is layer 1: [5000x401]*[401*25] = [5000x25]

% Theta2 = 10x26
% add the bias value to A
A = [ones(m, 1) A];

H = sigmoid(A*Theta2'); % A = 5000x26, Theta2' = 26x10: [5000x26]*[26*10] = [5000x10]  


[vals,p] = max(H,[],2); # this generates a max over the columns.




% =========================================================================


end
