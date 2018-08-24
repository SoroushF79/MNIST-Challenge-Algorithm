% Admittedly, a near copy of the cost function file provided by Andrew Ng however
% altered to reflect the change in architecture. This file returns the cost and 
% the gradients which are needed for the fmincg algorithm to run.





function [J grad] = nnCost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

[m, n] = size(X);


J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
y_matrix = eye(num_labels)(y, :);

% Forward Propagation

a1 = [ones(m, 1) X]; % Account for the bias neuron in the input layer
z2 = a1*Theta1';
a2 = sigmoid(z2); % I would like to eventually test with tanh as the activation function since it's more used these days
[m2, n2] = size(a2);
a2 = [ones(m2, 1) a2]; % Account for the bias neuron in the 2nd layer
z3 = a2*Theta2';
a3 = sigmoid(z3);

% Calculating cost

[m3, n3] = size(a3);
J = sum(sum((-y_matrix).*log(a3) - (1-y_matrix).*log(1-a3), 2))/m;

regularization = sum(sum(Theta1(:, 2:785).^2, 2))+sum(sum(Theta2(:, 2:101).^2, 2));

J = J + ((lambda)/(2*m))*regularization

% Backpropagation

d3 = a3 - y_matrix;
d2 = (d3*Theta2(:, 2:101)).*sigmoidGradient(z2);

Delta1 = d2'*a1;
Delta2 = d3'*a2;

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;

a = Theta1;
b = Theta2;

a(:, 1) = 0;
b(:, 1) = 0;

Theta1_grad = Theta1_grad + (lambda/m)*a; % We don't want to regularize the first column
Theta2_grad = Theta2_grad + (lambda/m)*b;


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
