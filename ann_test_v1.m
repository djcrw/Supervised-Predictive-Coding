function [rmse, error_percent, nrmse] = ann_test_v1(input,output,w,b,params)

iterations=size(input,2);
type = params.type;
n_layers = params.n_layers;
net_type = params.net_type;

x_output = zeros(size(output));
x = cell(n_layers,1);

batch_size=1000;
[in,~,n_batch] = get_batches(input, output, batch_size);
batches = length(n_batch);

for batch = 1:batches
    x{1} = in{batch};
    n_b = n_batch{batch};
    if strcmp(net_type,'bp') == 1
        for ii = 2:n_layers
            x{ii} = f_batch( w{ii-1} * x{ii-1} + repmat(b{ii-1},1,n_b), type);
        end
    elseif strcmp(net_type,'bio') == 1
        for ii = 2:n_layers
            x{ii} = w{ii-1} * (f_batch(x{ii-1}, type)) + repmat(b{ii-1},1,n_b) ;
        end
    end
    
    x_output(:,(batch-1)*batch_size+1:(batch-1)*batch_size + n_batch{batch}) = gather(x{n_layers});
end

[~, rmse, nrmse] = rsquare(output, x_output);

[~,lab_act] = max(output,[],1);
[~,lab_pred] = max(x_output,[],1);

errors = numel(find(lab_act~=lab_pred));
% errors = sum(xor(lab_act,lab_pred));
error_percent = 100*errors/iterations;
end

