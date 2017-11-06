function [w,b,momentum] = w_init_momentum_v1(params,neurons)

type = params.type;
n_layers = params.n_layers;
w = cell(n_layers,1);
b = cell(n_layers,1);

for i = 1:n_layers-1
    norm_b = 0;
    switch type
        case 'lin'
            norm_w = sqrt(1/(neurons(i+1) + neurons(i))) ;
        case 'tanh'
            norm_w = sqrt(6/(neurons(i+1) + neurons(i))) ;
        case 'logsig'
            norm_w = 4 * sqrt(6/(neurons(i+1) + neurons(i))) ;
        case 'exp'
            norm_w = sqrt(6/(neurons(i+1) + neurons(i))) ;
        case 'reclin'
            norm_w = sqrt(1/(neurons(i+1) + neurons(i))) ;
            norm_b = 0.1;
    end
    w{i} = unifrnd(-1,1,neurons(i+1),neurons(i)) * norm_w ;
    b{i} = zeros(neurons(i+1),1) + norm_b * ones(neurons(i+1),1) ;
    
    momentum.c_b{i} = 0 * b{i};
    momentum.c_w{i} = 0 * w{i};
    momentum.v_b{i} = 0 * b{i};
    momentum.v_w{i} = 0 * w{i};
end