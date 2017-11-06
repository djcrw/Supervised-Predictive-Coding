function [f,f_p] = f_b(x,type)
%function m = f(x)
%
%This function applies an activation function to a layer of neurons
%
% x -
% This is a column vector of neurons from a perticular layer.

switch type
    case 'lin'
        f = x;
        f_p = ones(size(x));
    case 'tanh'
        f = tanh(x);
        f_p = ones(size(x)) - f.^2;
    case 'logsig'
        f = 1./ (ones(size(x)) + exp(-x));
        f_p = f .* (ones(size(x)) - f) ;
    case 'reclin'
        f = max(x,0);
        f_p = sign(f);
    case 'exp'
        f = exp(x);
        f_p = exp(x);
    otherwise
        f = x;
        f_p = ones(size(x));
end
end
