% Written by Soroush Famili (Github: @SoroushF79) the week of Auguest 20, 2018.
% Submission for the MNIST Challenge to train an algorithm (a neural network in this case)
% to predict the value of handwritten numbers. Currently(8/24/18) stands at # 2129 out of 2316 on Kaggle with 
% ~89% success rate with 20 iterations. fmincg optimization algorithm was written by Carl Edward Rasmussen (c) 1999-2002.
% The architecture of the neural network is 784 inputs (for 28 x 28 pixel images),
% 2 100 neuron hidden layers, and 10 output classes for the 10 single digit numbers.
% The number of hidden layers and neurons in each label were, admittedly, arbitrarily chosen.
% They were good numbers that didn't take an extraordinary amount of time to train.
% Some credit is due to Andrew Ng and his Coursera course for the structure of the code.




input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 100;   % 100 hidden units
num_labels = 10;          % 10 labels, from 1 to 10 with 10 going to be mapped with 0. 

X = load('C:\Users\sorou\Desktop\Handwritting\train.txt'); % Text file of the csv training data provided

reshape(X, 42000, 785);
y = X(:, 1);
y(y ==0) = 10; % Mapping 10 with the digit 0. This is necessary because octave is 1-indexed not 0-indexed.
X = X(:, 2:785);

Theta1 = .1 .* randn(100, 785); % Applying Xavier Initialization: N(0, sqrt(2 layers/(100 + 100))). Thus, use N(0, .1)to initialize.
Theta2 = .1 .* randn(10, 101);


nn_params = [Theta1(:) ; Theta2(:)];


lambda = 10; % Arbitrarily chosen 

J = nnCost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);



options = optimset('GradObj', 'on', 'MaxIter', 20); % 20 iterations with my computer takes ~3.5 hours so better specifications (or a better algorithm)
% would decrease the time. 10 iterations takes ~2 hours and results in a ~81% success rate.


costFunction = @(p) nnCost(p,input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

X_test = load('C:\Users\sorou\Desktop\Handwritting\test.txt'); % Text file of the csv testing data provided
reshape(X_test, 28000, 784);

% Testing

pred = prediction(Theta1, Theta2, X_test);

pred(pred == 10) = 0; % Replace the 10's that were used as placeholders with 0's

r = xlswrite('C:\Users\sorou\Documents\sample_submission.xlsx',pred,'sample_submission','B2:B28001'); % Put results in the submission xlsx file

printf("Completely Done.\n");








