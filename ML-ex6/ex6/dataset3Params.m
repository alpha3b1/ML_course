function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

p1 = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
p2 = [0.03; 0.1; 0.3; 1; 3; 9; 27; 81] ;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

[p,q] = ndgrid(p1,p2);
params = [p(:),q(:)];

m = size(params,1);
min_error = Inf;
best_params = [0,0];
for l=1:m
    model= svmTrain(X, y, params(l,1), @(x1, x2) gaussianKernel(x1, x2, params(l,2)));
    predictions= svmPredict(model, Xval);
    model_error = mean(double(predictions ~= yval));
    if(model_error < min_error)
        min_error = model_error;
        best_params = params(l,:);
    end
end

C = best_params(1);
sigma = best_params(2);      

% =========================================================================

end
