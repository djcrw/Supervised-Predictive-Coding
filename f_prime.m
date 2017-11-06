function m = f_prime(x,type)
%function m = f(x)
%
%This function applies the derivaitve of an activation function to a layer of neurons
%
% x -
% This is a column vector of neurons from a perticular layer.

switch type
    case 'lin'
        m = ones(size(x));
    case 'tanh'
        m = ones(size(x)) - tanh(x).^2;
    case 'logsig'
        a = 1./ (ones(size(x)) + exp(-x)) ;
        m = a.* (ones(size(x)) - a) ;
        %m = logsig(x) .* (ones(size(x)) - logsig(x)) ;
    case 'reclin'
        m = sign(max(x,0));
    otherwise
        m = ones(size(x));
end
end