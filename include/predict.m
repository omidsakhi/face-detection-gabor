% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi
% http://www.facedetectioncode.com

function p = predict(net, X)
m = size(X, 1); %num-samples
h1 = sigmoid([ones(m, 1) X] * net.Theta1');
h2 = sigmoid([ones(m, 1) h1] * net.Theta2');
[dummy, p] = max(h2, [], 2); %p : num-samples x 1
end
