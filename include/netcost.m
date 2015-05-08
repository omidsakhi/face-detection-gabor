% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi
% http://www.facedetectioncode.com

function [J,grad] = netcost(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);

a1 = [ones(m,1) X]; % [ m x(num_inputs+1)]
z2 = a1*Theta1'; % [m x (num_inputs+1)] * [(num_hidden_neurons+1) x (num_inputs+1)]' = [m x (num_hidden_neurons+1)]\
a2 = sigmoid(z2); % [ m x(num_hidden_neurons+1)]
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2'; % [ m x (num_hidden_neurons+1)] * [(num_neurons_output) x (num_hidden_neurons+1)]' = [ m x (num_output_neurons)]
a3 = sigmoid(z3); %[m x (num_neurons_output)]
ey = eye(num_labels);
y = ey(y,:);
J = sum(sum((-y.*log(a3))-(1-y).*(log(1-a3))))/m; % [1x1]
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2))); % [1x1]
d3 = a3-y;
d2 = (d3*Theta2(:,2:end)).*sigmoidGradient(z2);
Theta1_grad = (1/m)*(d2'*a1 + lambda*Theta1);
Theta2_grad = (1/m)*(d3'*a2 + lambda*Theta2);
Theta1_grad(:,1) = Theta1_grad(:,1) - ((lambda/m) * (Theta1(:,1)));
Theta2_grad(:,1) = Theta2_grad(:,1) - ((lambda/m) * (Theta2(:,1)));
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
