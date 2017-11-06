function [images_train, labels_train, images_test, labels_test] = MNIST_load_v2(params)
%% Load training data
images_train = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte')';
%convert labels to vectors
n_data_train = length(labels);
labels_train = zeros(10, n_data_train);
for i=1:n_data_train
    index = labels(i) + 1;
    labels_train(index,i) = 1;
end

%% Load test data
images_test = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte')';
%convert labels to vectors
n_data_test = length(labels);
labels_test = zeros(10, n_data_test);
for i=1:n_data_test
    index = labels(i) + 1;
    labels_test(index,i) = 1;
end

%% Scale
images_train = images_train*params.scale_im + 0.5*(1-params.scale_im)*ones(size(images_train));
images_test = images_test*params.scale_im + 0.5*(1-params.scale_im)*ones(size(images_test));

labels_train = labels_train*params.scale_lab + 0.5*(1-params.scale_lab)*ones(size(labels_train));
labels_test = labels_test*params.scale_lab + 0.5*(1-params.scale_lab)*ones(size(labels_test));

%% Shorten for checking purposes
if params.shorten > 0
    n_data_train = params.shorten;
    n_data_test = params.shorten*params.val_proportion;
    images_train = images_train(:,1:n_data_train);
    labels_train = labels_train(:,1:n_data_train);
    images_test  = images_test (:,1:n_data_test);
    labels_test  = labels_test (:,1:n_data_test);
end