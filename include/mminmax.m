% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi
% http://www.facedetectioncode.com

function output = mminmax(input)

max_ = max(max(input));
min_ = min(min(input));

output = ((input-min_)/(max_-min_) - 0.5) * 2;