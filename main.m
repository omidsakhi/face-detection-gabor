% Version : 7.0
% Date : 2015-05-03
% Tested on MATLAB 2013a
% Author  : Omid Sakhi
% http://www.facedetectioncode.com

clear all;
close all;
clc;

if ~exist('./data')
    mkdir('./data');
end

if ~exist('main.m')
    fprintf ('You must navigate to the folder which contains main.m\n');
    fprintf ('Use dir command to make sure you are in the correct folder.\n');
    fprintf ('Use  cd command to navigate to the folder.\n');
    return;
end

if ~exist('./data/gabor.mat','file')    
    run ('include/createGabor.m');
end
if ~exist('./data/net.mat','file')
    run ('include/menuCreateNetwork.m');
end
if ~exist('./data/imgdb.mat','file')
    run ('include/menuLoadImages.m');
end

while (1==1)
    choice=menu('Face Detection',...
                'Create Database',...
                'Create Netwoek',...
                'Train Network',...
                'Test on Photos',...
                'Exit');
    if (choice ==1)
        run ('include/menuLoadImages.m');
    end
    if (choice == 2)
        run ('include/menuCreateNetwork.m');
    end    
    if (choice == 3)
        run ('include/menuTrainNetwork.m');
    end    
    if (choice == 4)
        run ('include/menuScanImage.m');
    end
    if (choice == 5)
        clear all;
        clc;
        close all;
        return;
    end    
end
