function [J,grad] = lr_cost(X, y, theta, lambda)
    theta_2 = theta;
    theta_2(1) = 0;
    m = size(y,1);
    
    J = 1/m * sum(-y.*log(sigmoid(X*theta)) - (1-y).*log(1-sigmoid(X*theta))) + lambda/(2*m)*(theta_2'*theta_2);

    %J = 1/m*sum(-(1-y).*log(1-sigmoid(X*theta))-y.*log(sigmoid(X*theta))) + 0.5*m*lambda*theta_2.^2 ;
    
    grad = 1/m * ((X'*(sigmoid(X*theta)-y))+(lambda*theta_2));
end