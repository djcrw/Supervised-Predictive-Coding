function m = f(x, type)
%function m = f(x)
%
%This function applies an activation function to a layer of neurons
%
% x -
% This is a column vector of neurons from a perticular layer.

switch type
    case 'lin'
        m = x;
    case 'tanh'
        m = tanh(x);
    case 'logsig'
        m = 1./ (ones(size(x)) + exp(-x));
        %m = logsig(x);
    case 'reclin'
        m = max(x,0);
    case 'exp'
        m = exp(x);
    otherwise
        m = x;
end
end
