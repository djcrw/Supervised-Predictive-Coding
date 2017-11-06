function [w,b,momentum] = ann_learn_v1(input,output,w,b,params,momentum)
l_rate = params.l_rate_ann;
d_rate = params.d_rate;
type = params.type;
net_type = params.net_type;
n_layers = params.n_layers;
batch_size = params.batch;
a = n_layers-1;

decay_r = params.decay_r;
eps = params.eps;
beta1 = params.beta1;
beta2 = params.beta2;
c_b = momentum.c_b;
c_w = momentum.c_w;
v_b = momentum.v_b;
v_w = momentum.v_w;

[in,out,n_batch] = get_batches(input, output, batch_size);
batches = length(n_batch);
for batch = 1:batches
    x = cell(n_layers,1);
    u = cell(n_layers,1);
    d = cell(n_layers,1);
    f_p = cell(n_layers,1);
    f_n = cell(n_layers,1);
    grad_b = cell(size(c_b));
    grad_w = cell(size(c_w));
    
    x{1} = in{batch};
    x_out = out{batch};
    n_b = n_batch{batch};
    
    if strcmp(net_type,'bp') == 1
        %% BP
        for ii = 2:n_layers
            u{ii} = w{ii-1} * x{ii-1} + repmat(b{ii-1},1,n_b) ;
            [x{ii}, f_p{ii}] = f_b( u{ii}, type) ;
        end
        d{n_layers} = x_out - x{n_layers} ;
        for ii=n_layers-1:-1:2
            d{ii} = w{ii}' * (d{ii+1} .* f_p{ii+1}) ;
        end
        
        for ii = 1:a
            grad_b{ii} = (1/n_b)*sum(d{ii+1} .* f_p{ii+1} ,2) ;
            grad_w{ii} = (1/n_b)*(d{ii+1} .* f_p{ii+1}) * x{ii}' - d_rate*w{ii} ;
        end
        
    elseif strcmp(net_type,'bio') == 1
        %% BIO
        for ii = 2:n_layers
            [f_n{ii-1}, f_p{ii-1}] = f_b( x{ii-1}, type) ;
            x{ii} = w{ii-1} * ( f_n{ii-1} ) +  repmat(b{ii-1},1,n_b) ;
        end
        d{n_layers} = x_out - x{n_layers} ;
        for ii=n_layers-1:-1:2
            d{ii} = (w{ii}' * d{ii+1}) .* f_p{ii} ;
        end
        
        for ii = 1:a
            grad_b{ii} = (1/n_b)*sum(d{ii+1} , 2);
            grad_w{ii} = (1/n_b)*d{ii+1} * f_n{ii}' - d_rate*w{ii} ;
        end
    end
    % UPDATE
    for ii = 1:a
        switch params.momentum
            case 'RMSPROP'
                c_b{ii} = decay_r * c_b{ii} + ( 1 - decay_r ) * (grad_b{ii}.^2) ;
                c_w{ii} = decay_r * c_w{ii} + ( 1 - decay_r ) * (grad_w{ii}.^2) ;
                
                w{ii} = w{ii} + l_rate * grad_w{ii} ./ (sqrt(c_w{ii}) + eps)  ;
                b{ii} = b{ii} + l_rate * grad_b{ii} ./ (sqrt(c_b{ii}) + eps)  ;
            case 'SM'
                c_b{ii} = decay_r * c_b{ii} + l_rate * grad_b{ii} ;
                c_w{ii} = decay_r * c_w{ii} + l_rate * grad_w{ii} ;
                
                w{ii} = w{ii} + c_w{ii}  ;
                b{ii} = b{ii} + c_b{ii}  ;
            case 'NAM'
                c_b_prev = c_b;
                c_w_prev = c_w;
                
                c_b{ii} = decay_r * c_b{ii} + l_rate * grad_b{ii} ;
                c_w{ii} = decay_r * c_w{ii} + l_rate * grad_w{ii} ;
                
                w{ii} = w{ii} - decay_r * c_w_prev{ii} + (1 + decay_r ) * c_w{ii}  ;
                b{ii} = b{ii} - decay_r * c_b_prev{ii} + (1 + decay_r ) * c_b{ii}  ;                
            case 'none'
                w{ii} = w{ii} + l_rate * grad_w{ii}   ;
                b{ii} = b{ii} + l_rate * grad_b{ii}   ;
            case 'Adam'
                c_b{ii} = beta1*c_b{ii} + (1 - beta1)*grad_b{ii};
                c_w{ii} = beta1*c_w{ii} + (1 - beta1)*grad_w{ii};
                
                v_b{ii} = beta2*v_b{ii} + (1 - beta2)*(grad_b{ii}.^2);
                v_w{ii} = beta2*v_w{ii} + (1 - beta2)*(grad_w{ii}.^2);
                
                t = (params.epoch_num - 1) * batches + batch;
                
                w{ii} = w{ii} + l_rate * (sqrt(1 - beta2^t) / (1 - beta1^t)) * c_w{ii} ./ (sqrt(v_w{ii}) + eps)  ;
                b{ii} = b{ii} + l_rate * (sqrt(1 - beta2^t) / (1 - beta1^t)) * c_b{ii} ./ (sqrt(v_b{ii}) + eps)  ;                
        end
    end    
end
momentum.c_b = c_b;
momentum.c_w = c_w;
momentum.v_b = v_b;
momentum.v_w = v_w;
end