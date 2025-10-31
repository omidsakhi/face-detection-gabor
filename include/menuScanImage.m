% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi

function im_out = menuScanImage

if ~exist('../data/net.mat','file')
    fprintf('Neural network does not exist!\n');
    return;
else
    load ('../data/net.mat');
end

if exist('../data/gabor.mat','file')
    load ('../data/gabor.mat');
else
    fprintf('Could not load Gabor filters from data folder\n');
    return;
end

pause(0.001);
[file_name,file_path] = uigetfile ('../*.jpg');
if file_path ~= 0
    im = imread ([file_path,file_name]);
    try 
        im = rgb2gray(im);
    end  
else
    return;
end

close all

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% PARAMETERS
TEMPLATE1 = 'template1.png';      
TEMPLATE2 = 'template2.png';      

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fprintf ('Scanning image ...\n');

warning off;

[m,n]=size(im);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% First Section
C1 = mminmax(double(im));
C2 = mminmax(double(imread (TEMPLATE1)));
C3 = mminmax(double(imread (TEMPLATE2)));
Corr_1 = double(conv2 (C1,C2,'same'));
Corr_2 = double(conv2 (C1,C3,'same'));
Cell.state = int8(imregionalmax(Corr_1) | imregionalmax(Corr_2));
Cell.state(1:13,:)=-1;
Cell.state(end-13:end,:)=-1;
Cell.state(:,1:9)=-1;
Cell.state(:,end-9:end)=-1;
Cell.net = ones(m,n)*-1;
[LUTm,LUTn]= find(Cell.state == 1);
imshow(im);
hold on
plot(LUTn,LUTm,'.y');drawnow;

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Second Section
while (1==1)
    [i,j] = find(Cell.state==1);
    if isempty(i)
        break;
    end
    toPredict = zeros(size(i,1),2160);
    for row=1:size(i,1)
        toPredict(row,:) = im2vec(im(i(row)-13:i(row)+13,j(row)-9:j(row)+8),G);
    end
    result = predict(net,toPredict);
    for row=1:size(i,1)
        Cell.state(i(row),j(row))=-1;
        Cell.net(i(row),j(row)) = result(row);        
    end
    for row=1:size(i,1)
        if (Cell.net(i(row),j(row)) == 1)
            plot(j(row),i(row),'.b');
            for i_=-1:1
                for j_=-1:1
                    if (i_~=0 && j_~=0)
                        continue;
                    end
                    m_=i(row)+i_;                    
                    n_=j(row)+j_;            
                    if (Cell.state(m_,n_) ~= -1)
                        Cell.state(m_,n_) = 1;
                        plot(n_,m_,'.y');
                    end
                end
            end
        else
            plot(j(row),i(row),'.r')
        end
    end
    ;drawnow;
end
    
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Third Section
hold off
figure;imshow (Cell.net,[]);
faceMap = Cell.net == 1; %Select all face pixels
faceMap = imregionalmax(faceMap);
faceMap = imdilate (faceMap,strel('disk',2,4));
[LabelMatrix,nLabel] = bwlabeln(faceMap,4);
CentroidMatrix = regionprops(LabelMatrix,'centroid');
faceMap = zeros(m,n);
for i = 1:nLabel
    faceMap(fix(CentroidMatrix(i).Centroid(2)),...
            fix(CentroidMatrix(i).Centroid(1))) = 1;
end
faceMap = drawRect(faceMap,[27 18]);
im_out (:,:,1) = im;
im_out (:,:,2) = im;
im_out (:,:,3) = im;
for i = 1:m
    for j=1:n
        if faceMap(i,j)==1
            im_out (i,j,1)=0;
            im_out (i,j,2)=255;
            im_out (i,j,3)=0;            
        end
    end
end

figure;
imshow(im_out,'InitialMagnification','fit');
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~