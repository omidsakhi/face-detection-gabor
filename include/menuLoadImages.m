% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi

function menuLoadImages

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
face_folder = '../face/';  %LOCATION OF FACE IMAGES
non_face_folder = '../non-face/'; %LOCATION OF NON-FACE IMAGES
file_ext = '.png';
out_max = 1;  % DESIRED OUTPUT FOR DETECTING A FACE          
out_min = 2;  % DESIRED OUTPUT FOR NOT DETECTING A FACE      
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if exist('../data/gabor.mat','file')
    load ('../data/gabor.mat');
else
    fprintf('Could not load Gabor filters from data folder\n');
    return;
end

IMGDB = cell (3,[]);

fprintf ('Loading Faces ');

folder_content = dir ([face_folder,'*',file_ext]);
nface = size (folder_content,1);
for k=1:nface
    string = [face_folder,folder_content(k,1).name];
    image = imread(string);    
    [m,n] = size(image);
    if (m~=27 || n~=18)
        continue;
    end
    f=0;
    for i=1:length(IMGDB)
        if strcmp(IMGDB{1,i},string)
            f=1;
        end
    end
    if f==1
        continue;
    end
    fprintf ('.');    
    IM {1} = im2vec (image,G);    % ORIGINAL FACE IMAGE
    IM {2} = im2vec (fliplr(image),G);    % MIRROR OF THE FACE 
    IM {3} = im2vec (circshift(image,1),G); 
    IM {4} = im2vec (circshift(image,-1),G);
    IM {5} = im2vec (circshift(image,[0 1]),G);
    IM {6} = im2vec (circshift(image,[0 -1]),G);
    IM {7} = im2vec (circshift(fliplr(image),1),G);
    IM {8} = im2vec (circshift(fliplr(image),-1),G);
    IM {9} = im2vec (circshift(fliplr(image),[0 1]),G);
    IM {10} = im2vec (circshift(fliplr(image),[0 -1]),G);
    for i=1:10
        IMGDB {1,end+1}= string;
        IMGDB {2,end} = out_max;
        IMGDB (3,end) = {IM{i}};
    end    
end

fprintf ('\nLoading non-faces ');

folder_content = dir ([non_face_folder,'*',file_ext]);
nnface = size (folder_content,1);
for k=1:nnface
    string = [non_face_folder,folder_content(k,1).name];
    image = imread(string);
    [m n] = size(image);
    if (m~=27 || n~=18)
        continue;
    end
    f=0;
    for i=1:length(IMGDB)
        if strcmp(IMGDB{1,i},string)
            f=1;
        end
    end
    if f==1
        continue;
    end    
    fprintf ('.');
    IM {1} = im2vec (image,G);
    IM {2} = im2vec (fliplr(image),G);
    IM {3} = im2vec (flipud(image),G);
    IM {4} = im2vec (flipud(fliplr(image)),G);    
    for i=1:4
        IMGDB {1,end+1}= string;
        IMGDB {2,end} = out_min;
        IMGDB (3,end) = {IM{i}};
    end    
end
fprintf('\n');
save ('../data/imgdb.mat','IMGDB');