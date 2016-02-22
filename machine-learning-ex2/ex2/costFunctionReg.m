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
feature_num = length(theta);
hx = sigmoid(X*theta);
J = sum(-y.*log(hx)-(1-y).*log(1-hx))/m + lambda/2/m*sum(theta(2:feature_num,1).^2);
grad(1) = X(:,1)'*(hx-y)/m;
grad(2:feature_num) = X(:,2:feature_num)'*(hx-y)/m+lambda/m*theta(2:feature_num);

%J = sum(-y.*log(hx)-(1-y).*log(1-hx))/m + lambda/2/m*sum(theta.^2);
%grad = X'*(hx-y)/m+lambda/m*theta;

% =============================================================

end
