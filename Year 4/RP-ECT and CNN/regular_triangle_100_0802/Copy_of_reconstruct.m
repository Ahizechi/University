clear
close all

%   load S
load ref1
load bb
  load JJ2

p = 20;
% A = S;
N = [p p ];

% Recover the image
mu      = 3;
lambda  = 100;
gamma   = 100;
nInner  = 1;
nBreg   = 3;
% 
% R=ones(66,1);
% data2 = ref1;
% Form the CS data
% data    = R.*data2;
% 
% u       = reshape(A'*data2,N);

folder = uigetdir(pwd, 'Select folder');
if folder == 0
	return;
end

dataFiles = [dir(fullfile(folder,'tri100_0_*.mat'))];
numberOfFiles = length(dataFiles);

% return
for k = 1 : numberOfFiles
	fullFileName = fullfile(folder, dataFiles(k).name);
    load(fullFileName)
    fileName = erase(fullFileName,folder)
    fileName = erase(fileName,'\')
    rotDeg = erase(fileName,'tri100_0_')
    rotDeg = rotDeg(1:end-9)
    fileName = fileName(1:end-4)

    [n_row,n_column] = size(framedata);

     for frame = 1:n_row
        plane2 = framedata(frame,:)';
        b=-((plane2)-(bb'))./1;
        if frame==2
            bb=plane2';
        end
        b(13)=0;
        bcombined(k,:)=b;
        
     end
end

bnew=bcombined(:);
bnew(bnew<-200)=0;

x=(JJ2'*JJ2+1e-3*eye(400))\JJ2'*bnew;
%   x(x<0)=0;
   Kent_draw_new_s(x,20);colormap jet;colorbar
