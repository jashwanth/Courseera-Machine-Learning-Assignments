function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

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

values = [0.01, 0.03, 0.09, 0.1, 0.3, 0.9, 1, 3, 10, 30];

listOutcomes = [];

for i=1:size(values,2)
  for j=1:size(values,2)
     testC = values(1,i);
     testSigma = values(1,j);
     model= svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSigma));
     predictions = svmPredict(model, Xval);
     errorVal = mean(double(predictions ~= yval));
     listOutcomes = [listOutcomes; testC, testSigma, errorVal];
     fprintf("\n errorval=%f testC=%f testSgima=%f ", errorVal, testC, testSigma);
  end
end

[minError, minIndex] = min(listOutcomes(:, 3));

C = listOutcomes(minIndex, 1);
sigma = listOutcomes(minIndex, 2);
fprintf("\n\n Final Evaluation errorval=%f testC=%f testSgima=%f ", listOutcomes(minIndex,3), C, sigma);




% =========================================================================

end
