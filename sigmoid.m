function [answer] = sigmoid(z)
    answer = 1./(1+exp(-z));
end