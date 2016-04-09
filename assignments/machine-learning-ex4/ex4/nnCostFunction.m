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

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

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

%%% Get J.

X = [ones(m, 1) X];

z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];

% m * num_labels
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%% TODO: see if it is possible to replace loop with vectorization.
for i=1:m
  % Get t. 
  label = y(i);
  t = zeros(num_labels, 1);
  t(label) = 1;
  
  % Get a.
  a = a3(i,:)';

  % Get energy between t and a. 
  J += sum(-t .* log(a) - (1.0 - t) .* log(1.0 - a));
end

J /= m;

Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);

Theta1_reg .^= 2;
Theta2_reg .^= 2;

sum_theta1_reg = sum(sum(Theta1_reg));
sum_theta2_reg = sum(sum(Theta2_reg));
reg = (sum_theta1_reg + sum_theta2_reg) * lambda / (m * 2.0);

J += reg; 

%%% Get grad.
for i=1:m
  % Get t.
  label = y(i);
  t = zeros(num_labels, 1);
  t(label) = 1;
  
  % Get a_1, a_2.
  a_1 = X(i,:)';
  a_2 = a2(i,:)';

  % Get delta_3(10*1), delta_2(25*1). 
  delta_3 = a3(i,:)' - t;
  %fprintf('size delta_3: %f %f\n', size(delta_3));
  delta_2 = (Theta2' * delta_3)(2:end) .* sigmoidGradient(z2(i,:)'); 

  % Get Delta.
  Delta1 += delta_2 * a_1'; 
  Delta2 += delta_3 * a_2';
end

Theta1_grad = Delta1 / m + [zeros(size(Delta1, 1), 1) (lambda * Theta1 / m)(:,2:end)];
Theta2_grad = Delta2 / m + [zeros(size(Delta2, 1), 1) (lambda * Theta2 / m)(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
