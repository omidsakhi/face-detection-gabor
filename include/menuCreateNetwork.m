% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi
% http://www.facedetectioncode.com

function menuCreateNetwork

fprintf ('Creating neural network ...\n');

net = createNetwork(2160,20,2);

save ('../data/net.mat','net');