function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
dims = length(X_norm(1,:));
%fprintf("dims = %d\n",dims);
mu = zeros(1, size(X, dims));
sigma = zeros(1, size(X, dims));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


for dim = 1:dims;
    mu = mean(X(:,dim));
    
    
    sigma = std(X(:,dim));
    
    if sigma == 0
        sigma = 1;
    endif;
    for i = 1:size(X,1);
        X_norm(i,dim) = (X(i,dim) - mu)/sigma;
        %fprintf('dimension %f, iteration %f, oldVal %f, newVal %f\n',dim,i,X(i,dim),X_norm(i,dim));
    endfor;
endfor;






% ============================================================

end
