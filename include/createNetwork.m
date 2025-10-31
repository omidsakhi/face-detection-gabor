% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi

function net = createNetwork(input_layer_size,hidden_layer_size,num_labels)

net.input_layer_size = input_layer_size;
net.hidden_layer_size = hidden_layer_size;
net.num_labels = num_labels;

net.Theta1 = initWeights(input_layer_size, hidden_layer_size);
net.Theta2 = initWeights(hidden_layer_size, num_labels);




