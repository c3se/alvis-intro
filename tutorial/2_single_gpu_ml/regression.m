%% ML REGRESSION ON A SINGLE GPU
% This will introduce the very basics of using MATLAB on Alvis
% N.B. there are some local functions defined at the end of the script

settings; % we're calling this to not get warning on exit

gpuDevice

%% Creating the data
% First we will create some example dataset

n_data = 1000000
[x, y] = get_data(n_data);

%% Take a look at the data
if n_data < 10000 && ~batchStartupOptionUsed
    figure(1); hold off
    x_plot = linspace(-1, 1, 20)';
    plot(x, y, '.'); hold on
    plot(x_plot, f_true(x_plot));
    xlabel('X');
    ylabel('Y');
    legend('Samples', 'Underlying relation');
end

%% Machine learning model
% MATLAB has a lot of different functions for machine learning models
% see https://mathworks.com/discovery/machine-learning-models.html
% We will use some premade functions from the Deep Learning toolbox
% this is overkill for the simple example but it might serve your future
% tasks the best.

% The layers in our model see
% https://www.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html
% for a list of all possible layers
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(1)
    regressionLayer
];

%% Training the model
% See https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
% for all the possible training options and what they mean
batchSize = min(n_data, 2^13);
[valData{1:2}] = get_data(100);
options = trainingOptions("sgdm", ...
    "Plots","none", ...
    "MaxEpochs", 5, ...
    "MiniBatchSize", batchSize, ...
    "Shuffle","every-epoch", ...
    "InitialLearnRate", 0.01, ...
    "ValidationData", valData, ...
    "ValidationFrequency", floor(n_data / batchSize), ...
    "CheckpointPath", "", ...
    "Verbose", true, ...
    "VerboseFrequency", 20 ...
);

net = trainNetwork(x, y, layers, options);

%% Visualize predictions
if n_data < 10000 && ~batchStartupOptionUsed
    figure(2); hold off
    plot(x, y, "."); hold on
    plot(x_plot, f_true(x_plot));
    plot(x_plot, predict(net, x_plot));
    xlabel("X");
    ylabel("Y");
    legend("Samples", "Underlying relation", "Predicted relation");
end

%% Local functions
function y = f_true(x)
% F_TRUE the underlying noiseless relation f: x -> y
    y = 0.5 * x + 0.3;
end
function [x, y] = get_data(n_points)
% GET_DATA function to produce an example dataset for regression
    % See
    % https://se.mathworks.com/help/deeplearning/ug/datastores-for-deep-learning.html
    % for alternatives for more advanced MATLAB dataloading
    x = 2 * rand(n_points, 1, 'single') - 1;
    y = 0.5 * x + 0.3 + 0.1 * randn(n_points, 1, 'single');
    % EXCERCISE modify this function to return gpuArrays instead
    % https://se.mathworks.com/help/parallel-computing/gpuarray.html
    % Optional: can you generate the random numbers on the GPU directly?
end

