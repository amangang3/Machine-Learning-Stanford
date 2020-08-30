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

% need to loop 0.3 - 3
minimum_error = Inf; 
steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for c_check = steps
    for sigma_check = steps
        model = svmTrain(X, y, c_check, @(x1, x2) gaussianKernel(x1, x2, sigma_check));
        preds = svmPredict(model, Xval);
        error_in_preds = mean(double(preds ~= yval));
        if error_in_preds < minimum_error
            C = c_check; 
            sigma = sigma_check;
            minimum_error = error_in_preds;
        end
    end
end
% =========================================================================

end
