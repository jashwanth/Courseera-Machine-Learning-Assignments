function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m,1) X];
% size of X now is 5000*401
 

% Theta1 has size 25 x 401
% Theta2 has size 10*26

activation2 = sigmoid(X*Theta1');
% size is 5000*25
activation2 = [ones(size(activation2,1),1) activation2];
% size is 5000*26
hypothesis = sigmoid(activation2*Theta2');
% hypothesis is 5000*10

%fprintf('Program paused. Press enter to continue.\n');
%pause;

% now the variable y contains the labels 1..10. We need to convert y into
% vectors with values 0 and 1 only.

convertYtoBinary = zeros(m, num_labels);
% size is 5000*10
for i = 1:m
convertYtoBinary(i, y(i))=1;
endfor



J = (1/m) * sum(sum ( ( (-convertYtoBinary .* log(hypothesis) ) - ( (1-convertYtoBinary) .* log ( 1-hypothesis) ) ) ) )...
% now add regularisation ...
% dont regular bias unit which is column1 of theta matrix 
     +  ( lambda / (2*m)) * (sum( sum(Theta1(:,2:size(Theta1,2)).^2))  + sum( (sum(Theta2(:, 2:size(Theta2,2) ).^2)) ) );


% end of part1 solution 

% -------------------------------------------------------------


% part 2 solution 

% we need to run back propogation for each of 5000 examples

for t = 1 : m 
% compute activation 1 for backpropagation
% size is 401*1 ( sinxe X has already 1 bias column added to it. (no need to add again)
activationB1 = [X(t, :)'];
% size of z2 is 25*1
z2 = Theta1*activationB1;

activationB2 = sigmoid(z2);
%size now is 26*1
activationB2 = [1 ; activationB2];
%theta2 is 10*26 so size of z3 is 10*1
z3 = Theta2* activationB2;
activationB3 = sigmoid(z3);

%yexpected = zeros(num_labels,1);
%yexpected(y(t,1), 1) = 1;

delta3 = activationB3 - convertYtoBinary(t, :)';

% size is 26*10 mult 10*1 = 26*1 - 
% add 1 to z2 to make size as 26*1 

delta2 = (Theta2' * delta3) .* sigmoidGradient(([1 ; z2]));

% to compute gradient formula given is tildaLayeratY = tildaLayeratY +...
% (deltaY+1) * (activationB-Y)';
% size =  10*1 mult with 1*26 to get 10*26 theta2 matrix 
Theta2_grad = Theta2_grad + ( (delta3) * (activationB2)' ) ;


% remove the bias unit from delta2
delta2 = delta2(2:size(delta2,1));

% size of delta2 now is 25*1 
% size is Theta1_grad is 25*401  which is 26*1 mult 1*401 = 26*401 
Theta1_grad = Theta1_grad + ( (delta2) * (activationB1)' );

% size is  
 
endfor


% now apply regularisation using lamda
Theta2_grad(:,1) = Theta2_grad(:,1) * (1/m);
Theta2_grad(:,2 : size(Theta2_grad,2)) = (Theta2_grad(:,2 : size(Theta2_grad,2)) + (lambda  * (Theta2(:, 2:size(Theta2,2)))) ) * (1/m);

Theta1_grad(:,1) =  Theta1_grad(:,1) * (1/m);
Theta1_grad(:,2 : size(Theta1_grad,2)) = (Theta1_grad(:,2 : size(Theta1_grad,2)) +  (lambda * ( Theta1(:, 2: size(Theta1,2)))) ) * (1/m);


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
