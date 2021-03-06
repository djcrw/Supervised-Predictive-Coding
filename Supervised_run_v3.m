clear all
close all
%%  PARAMETER SETUP
params.data_type = 'MNIST';
% need to download the following files from
% 'http://yann.lecun.com/exdb/mnist/'
%train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
%train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
%t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
%t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

% Determines whether activation function is applied before or after weight matrix multiplication.
% 'bp' variant from our 2015 bioRxiv paper, 'bio' from 2017 Neural Computation paper.
params.net_type = 'bio';
params.type = 'logsig'; % activation function type
params.nodes = [500 500]; % number of nodes in hidden layers
params.l_rate_ann = 1e-3; %learning rate for ANN
params.l_rate_pc = 1e-3; %learning rate for PC
params.beta = 0.1; %euler integration step PC inference
params.it_max = 50; %max iteration nu,ber for PC inference
params.batch = 20; %batch size
params.epochs = 100; %numner of epochs
params.shorten = 1000; %shorten training data. 0 means no shortening
params.skip = 1;
params.bio_inv = 1; %whether to perform inverse activation function to input data (useful for 'bio' variant)

%momentum params
params.momentum = 'Adam';
params.eps = 1e-8;
params.decay_r = 0.9;
params.beta1 = 0.9;
params.beta2 = 0.999;

params.d_rate = 0*0.001; %weight decay
params.min_its = 10; %minimum iterations of PC inference
params.condition = 1e-6; %condition to satisfy for inference to stop
params.div = 2; %lower beta if inference not converginh
params.beta_min = 0; %minium value for beta

% which models to train
params.ANN = 1; %include ANN
params.PC = 1; %include PC
var_out = 1; % variance on output layer

%data preprocessing
params.val_proportion=0.2;
params.scale_im = 1;
params.scale_lab = 0.94;

params.neurons = [784,params.nodes,10];
params.n_layers = length(params.neurons);
var = ones(1, params.n_layers);
var(end) =  var_out ;
params.var = var;

%% RUN MODELS
[w,b,momentum] = w_init_momentum_v1(params,params.neurons);
[rmse_tr, error_percent_tr, rmse_val, error_percent_val] = im2class_v1(params,w,b,momentum);

save (['simplified_results[' datestr(clock,30) ']'])
