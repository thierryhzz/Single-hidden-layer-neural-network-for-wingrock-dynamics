function p = SHLNN_predict(Theta1, Theta2, X)
%PREDICT Predict PDOT of an input given a trained neural network
%   p = SHLNN_PREDICT(Theta1, Theta2, X) outputs the predicted value of PDOT based on X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
p = ([ones(m, 1) h1] * Theta2');

% =========================================================================


end
