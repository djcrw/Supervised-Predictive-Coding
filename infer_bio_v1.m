function [x,e,its] = infer_bio_v1(x,w,b,params)

it_max = params.it_max;
n_layers = params.n_layers;
type = params.type;
beta = params.beta;
e = cell(n_layers,1);
f_n = cell(n_layers,1);
f_p = cell(n_layers,1);
n_batch = params.n_batch;
condition = params.condition;
div = params.div;
F_t = 0;
var = params.var;
for ii=2:n_layers
    b{ii-1} = repmat(b{ii-1},1,n_batch);
    [f_n{ii-1},f_p{ii-1}] = f_b( x{ii-1}, type) ;
    e{ii} = (x{ii} - w{ii-1} * ( f_n{ii-1} ) - b{ii-1})/var(ii) ;
    F_t = F_t - var(ii)*sum(e{ii}.*e{ii}, 1);
end

for i = 1:it_max
    condition1 = condition*beta/var(n_layers);
    for ii=2:n_layers-1
        g = ( w{ii}' *  e{ii+1} ) .* f_p{ii} ;
        x{ii} = x{ii} + beta * ( - e{ii} + g );
    end
    
    F = 0;
    for ii=2:n_layers
        [f_n{ii-1},f_p{ii-1}] = f_b( x{ii-1}, type) ;
        e{ii} = (x{ii} - w{ii-1} * ( f_n{ii-1} ) - b{ii-1})/var(ii) ;
        F = F - var(ii)*sum(e{ii}.*e{ii}, 1);
    end
    diff = F -F_t;
    if any(diff<0)
        beta = beta/div;
%         keyboard
    elseif mean(diff)<condition1
        break
    end
    F_t = F;
    
end
its=i;
end