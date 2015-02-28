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
    n = length(grad);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta

    J1 = 0;
    for i = 1:m
        hTheta = sigmoid(dot(theta, X(i, :)));

        if y(i) == 1
            J1 = J1 - log(hTheta);
        else
            J1 = J1 - log(1 - hTheta);
        end
    end

    J1 = J1 / m;

    J2 = 0;
    for j = 2:n
        J2 = J2 + theta(j)^2;
    end

    J2 = J2*lambda/(2*m);

    J = J1 + J2;

    for j = 1:n
        ss = 0;
        for i = 1:m
            hTheta = sigmoid(dot(theta, X(i, :)));
            ss = ss + (hTheta - y(i))*X(i, j);
        end
        if j > 1
            grad(j) = ss/m + lambda/m*theta(j);
        else
            grad(j) = ss/m;
        end
    end

    % =============================================================

end
