function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta); % The hypothesis function.
ttheta = theta;
ttheta(1)=0; % Temporary theta having first element equal to 0 as we don't regularize the first term.
J = (1/m)*(-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*sum(ttheta.^2, 'all'); % The cost function.
grad = (1/m)*X'*(h-y) + (lambda/m)*(ttheta); % The gradients.
tgrad = (1/m)*X'*(h-y);
grad(1) = tgrad(1); % For j=1, it remains unchanged.

% =============================================================

end
