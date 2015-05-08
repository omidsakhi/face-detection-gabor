% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi
% http://www.facedetectioncode.com

function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end
