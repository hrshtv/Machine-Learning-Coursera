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
models =zeros(64, 3);
i = 1;
steps = [0.01 0.03 0.1 0.3 1, 3, 10 30];
for C_i = steps
    for sigma_i = steps
        model = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        models(i, :) = [error, C_i, sigma_i];
        i = i + 1;
    end
end

sorted = sortrows(models, 1);
best_model = sorted(1, :);
C = best_model(2); 
sigma = best_model(3); 

% =========================================================================

end
