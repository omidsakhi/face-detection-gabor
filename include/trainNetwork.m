% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi

function [net,cost] = trainNetwork(net,X,T,num_iterations,lambda)

options = optimset('MaxIter', num_iterations,'Display','iter');

% define cost function
costFunction = @(p) netcost(p, ...
                            net.input_layer_size, ...
                            net.hidden_layer_size, ...
                            net.num_labels, X, T, lambda);

% Unroll parameters
initial_nn_params = [net.Theta1(:) ; net.Theta2(:)];

[nn_params, cost] = fminsearch(costFunction, initial_nn_params, options);
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Enroll parameters
net.Theta1 = ...
    reshape(...
    nn_params(1:net.hidden_layer_size * (net.input_layer_size + 1)), ...
                 net.hidden_layer_size, (net.input_layer_size + 1));
             
             
net.Theta2 = ...
    reshape(...
    nn_params((1 + (net.hidden_layer_size * (net.input_layer_size + 1))):end), ...
                 net.num_labels, (net.hidden_layer_size + 1));
