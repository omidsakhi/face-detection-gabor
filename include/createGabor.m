% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi

function createGabor

fprintf ('Creating Gabor Filters ...\n');

G = cell(5,8);

for s = 1:5
    for j = 1:8
        G{s,j}=zeros(32,32);
    end
end

for s = 1:5
    for j = 1:8
        G{s,9-j} = gabor([32 32],(s-1),j-1,pi,sqrt(2),pi);
    end
end

figure;
for s = 1:5
    for j = 1:8        
        subplot(5,8,(s-1)*8+j);        
        imshow(real(G{s,j}),[]);
    end
end

for s = 1:5
    for j = 1:8        
        G{s,j}=fft2(G{s,j});
    end
end
drawnow;
save ('../data/gabor.mat','G');