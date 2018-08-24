% A rather self-explanatory function. We now have trained Thetas 1 and 2
% so now I apply them to the test data. The result, p, is a 28000 x 1 column vector.



function p = prediction(Theta1, Theta2, X)

m = size(X, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

end