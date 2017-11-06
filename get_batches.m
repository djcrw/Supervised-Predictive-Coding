function [batches_in,batches_out,n_batch] = get_batches(input, output, batch_size)
% return a cell of batches
n_data = size(input,2);

if batch_size == 0 || batch_size == 1
    batches_in = cell(n_data,1);
    batches_out = cell(n_data,1);
    n_batch = cell(n_data,1);
    for batch = 1:n_data
        batches_in{batch} =  input(:,batch);
        batches_out{batch} =  output(:,batch);
        n_batch{batch} = 1 ;
    end
    
else
    batches = ceil(n_data/batch_size);
    last_batch = n_data - batch_size*(batches-1);
    batches_in = cell(batches,1);
        batches_out = cell(batches,1);
        n_batch = cell(batches,1);
    for batch = 1:batches        
        if batch == batches
            batches_in{batch} = input(:,(batches-1)*batch_size+1:end);
            batches_out{batch} = output(:,(batches-1)*batch_size+1:end);
            n_batch{batch} = last_batch ;
        else
            batches_in{batch} = input(:,(batch-1)*batch_size+1:batch*batch_size);
            batches_out{batch} = output(:,(batch-1)*batch_size+1:batch*batch_size);
            n_batch{batch} = batch_size ;
        end
    end
end



