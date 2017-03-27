function [J Theta1_grad Theta2_grad] = nnCostFunction(Theta1, Theta2, X, y, lambda)

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones([m,1]) X]; % Add column one to the left of matrix which is the bias unit.
a2 = sigmoid(X*Theta1'); % Output of the hidden layer
a2 = [ones([size(a2,1),1]) a2]; % The bias unit added
a3 = (a2*Theta2'); % Output of the NN.
y_p=a3;

for n=1:m % Calculating cost function
    J = J + 0.5*(y(n,:)-y_p(n,:))*(y(n,:)-y_p(n,:))';
end
J = J/m; % Cost function without regularization

for n=2:size(Theta1,2)
    J=J+lambda/(2*m)*Theta1(:,n)'*Theta1(:,n);
end
for n=2:size(Theta2,2)
    J=J+lambda/(2*m)*Theta2(:,n)'*Theta2(:,n);
end % Regularized cost function

%Back Propagation
tri1 = zeros(size(Theta1)); %Initialization
tri2 = zeros(size(Theta2));


for n=1:m
   % delta3 = (y_p(n,:)-y(n,:)).*sigmoidGradient((a2(n,:)*Theta2')'); % Calculating the error of NN output 
   delta3=y_p(n,:)-y(n,:); 
   delta2 = Theta2(:,2:end)'*delta3; % Calculating the error of the hidden layer output 
   % delta2 = delta2(2:end);
    delta2 = delta2.*sigmoidGradient((X(n,:)*Theta1')');
    tri1=tri1+delta2*X(n,:);
    tri2=tri2+delta3*a2(n,:);
end
Theta1_grad = tri1/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda/m*Theta1(:,2:end); % Regularization added.
Theta2_grad = tri2/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda/m*Theta2(:,2:end); % Regularization added.

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
