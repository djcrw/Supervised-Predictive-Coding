function m = f_inv(x, type)
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
        m = 0.5 * log ( (ones(size(x)) + x) ./ (ones(size(x)) - x));
    case 'logsig'
        m = log ( x./(ones(size(x)) - x) );
    case 'exp'
        m = log(x);
    otherwise
        m = x;
end
end
