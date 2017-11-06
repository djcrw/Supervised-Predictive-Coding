function [x,e,u,its] = infer_bp_v1(x,u,w,b,params)

it_max = params.it_max;
n_layers = params.n_layers;
type = params.type;
beta = params.beta;
% beta_min = params.beta_min;
div = params.div;
e = cell(n_layers,1);
f_n = cell(n_layers,1);
f_p = cell(n_layers,1);
n_batch = params.n_batch;
condition = params.condition;
var = params.var;
F_t = 0;
for ii=2:n_layers
    b{ii-1} = repmat(b{ii-1},1,n_batch);
    [f_n{ii},f_p{ii}] = f_b( u{ii}, type) ;
    e{ii} = (x{ii} - f_n{ii})/var(ii);
    F_t = F_t - var(ii)*sum(e{ii}.*e{ii}, 1);
end

for i = 1:it_max
    condition1 = condition*beta/var(n_layers);
    for ii=2:n_layers-1
        g = w{ii}' * ( e{ii+1} .* f_p{ii+1} );
        x{ii} = x{ii} + beta * ( - e{ii} + g );
    end
    
    F = 0;
    for ii=2:n_layers
        u{ii} = w{ii-1} * x{ii-1}  + b{ii-1} ;
        [f_n{ii},f_p{ii}] = f_b( u{ii}, type) ;
        e{ii} = (x{ii} - f_n{ii})/var(ii);
        F = F - var(ii)*sum(e{ii}.*e{ii}, 1);
    end
    diff = F -F_t;
    if any(diff<0)
        beta = beta/div;
    elseif mean(diff)<condition1
        break
    end
    F_t = F;
    
end
its=i;
end