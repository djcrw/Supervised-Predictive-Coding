function [rmse_tr,e_p_tr,rmse_val,e_p_val] = im2class_v1(params,w_pc,b_pc,momentum_pc)
disp(datestr(now) )
[im_tr, lab_tr, im_val, lab_val] = MNIST_load_v2(params);
n_data_train = size(lab_tr,2);
n_input = size(im_tr,1);
%perform inverse of input - only if net_type is 'bio' (as that performs
%activation function immediately to input)
if strcmp(params.net_type,'bio') == 1
    if params.bio_inv ==1
        im_tr = f_inv(im_tr,params.type);
        im_val = f_inv(im_val,params.type);
    end
end
w_ann = w_pc;
b_ann = b_pc;
momentum_ann = momentum_pc;

% rejig condition to suit number of neurons
params.condition = params.condition/(sum(params.neurons) - n_input);

%% Test before any training
disp('Testing');   % Accuracy measures for validation data
if params.ANN == 1 ; tic % ANN TEST + TRAIN
    [rmse_val(1,1), e_p_val(1,1), nrmse_val(1,1)] = ann_test_v1(im_val,lab_val,w_ann,b_ann,params);
    [rmse_tr(1,1), e_p_tr(1,1), nrmse_tr(1,1)] = ann_test_v1(im_tr,lab_tr,w_ann,b_ann,params);
    toc
end
if params.PC ==1 ; tic % PC TEST + TRAIN
    [rmse_val(1,2), e_p_val(1,2), nrmse_val(1,2)] = ann_test_v1(im_val,lab_val,w_pc,b_pc,params);
    [rmse_tr(1,2), e_p_tr(1,2), nrmse_tr(1,2)] = ann_test_v1(im_tr,lab_tr,w_pc,b_pc,params);
    toc
end

g=sprintf('%d ', e_p_tr(1,:));
fprintf('training error = %s \n',g);
g=sprintf('%d ', e_p_val(1,:));
fprintf('test error = %s \n',g);
g=sprintf('%d ', rmse_tr(1,:).^2);
fprintf('training mse = %s \n',g);

%% Supervised Training
for epoch = 1:params.epochs
    fprintf('epoch = %d of %d \n',epoch,params.epochs);
    params.epoch_num = epoch;
    if params.ANN == 1; disp('ANN'); tic
        [w_ann,b_ann,momentum_ann] = ann_learn_v1(im_tr,lab_tr,w_ann,b_ann,params,momentum_ann);
        toc
    end;
    if params.PC ==1; disp('PC'); tic
        [w_pc,b_pc,momentum_pc] = pc_learn_v1(im_tr,lab_tr,w_pc,b_pc,params,momentum_pc);
        toc
    end;
    %% Test during training
    skip = params.skip;
    if mod(epoch,skip)==0
        disp('Testing');
        if params.ANN == 1; tic
            [rmse_val(epoch/skip+1,1), e_p_val(epoch/skip+1,1), nrmse_val(epoch/skip+1,1)] = ann_test_v1(im_val,lab_val,w_ann,b_ann,params);
            [rmse_tr(epoch/skip+1,1), e_p_tr(epoch/skip+1,1), nrmse_tr(epoch/skip+1,1)] = ann_test_v1(im_tr,lab_tr,w_ann,b_ann,params);
            toc
        end;
        if params.PC ==1; tic
            [rmse_val(epoch/skip+1,2), e_p_val(epoch/skip+1,2), nrmse_val(epoch/skip+1,2)] = ann_test_v1(im_val,lab_val,w_pc,b_pc,params);
            [rmse_tr(epoch/skip+1,2), e_p_tr(epoch/skip+1,2), nrmse_tr(epoch/skip+1,2)] = ann_test_v1(im_tr,lab_tr,w_pc,b_pc,params);
            toc
        end;
        g=sprintf('%d ', e_p_tr(epoch/skip+1,:));
        fprintf('training error = %s \n',g);
        g=sprintf('%d ', e_p_val(epoch/skip+1,:));
        fprintf('test error = %s \n',g);
        g=sprintf('%d ', rmse_tr(epoch/skip+1,:).^2);
        fprintf('training mse = %s \n',g);
    end
    perm = randperm(n_data_train);
    im_tr = im_tr(:,perm);
    lab_tr = lab_tr(:,perm);
end