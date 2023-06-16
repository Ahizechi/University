
close all
clear all

 p=20;
 
folder = uigetdir(pwd, 'Select folder');
if folder == 0
	return;
end

dataFiles = [dir(fullfile(folder,'tri100_0_*.mat'))];
numberOfFiles = length(dataFiles);

for k = 1 : numberOfFiles
	fullFileName = fullfile(folder, dataFiles(k).name);
    load(fullFileName)
    fileName = erase(fullFileName,folder)
    fileName = erase(fileName,'\')
    rotDeg = erase(fileName,'tri100_0_')
    rotDeg = rotDeg(1:end-9)
    fileName = fileName(1:end-4)

    framedata=framedata';
% %  
    b = (framedata( :,5)-framedata(:,1))./framedata(:,1); 
%     b(13)=0;
%  load ref1
%  load ref2
    load S
    J=S;
% load JJ50
% J=reshape(JJ50,[50 50 50 66]);
% Jz= J(:,:,39:48,:) ;
% 
% J=reshape(Jz,[25000,66]);
% J=J';
% for i=1:10
%     
% ff(:,i)=framedata(i,:)'-ref1;
% end
 
    N       = [p p p]; %128; % The image will be NxN
    sparsity = .85; % use only 30% on the K-Space data for CS 

% build an image of a square
    image   = zeros(N);
    image(N(1)/4:3*N(1)/4,N(2)/4:3*N(2)/4,N(3)/4:3*N(3)/4)=255;

    rand('state',0);
    R       = rand(N);
    R       = R<sparsity;
    R(1,1)  = 1;
    R       = R(:); 
    R=ones(66,1);
    mu      =  1;
    lambda  =1;
    gamma   = .01;
    nInner  = 1;
    nBreg   =3;
    arr_idx = ones(N);


    % target  = image;
    for i=1:1
    [u,errAll] = mrics3D_FDOT_SB_NB_P0_GNK(J,b,R,N,mu, lambda, gamma, nInner, nBreg);
    % uu(:,:,:,i)=u;
    end

     x=mean(u(:,:,:),1); xx=reshape(x,[p p]);
       xx=imrotate(xx,- str2double(rotDeg) ,'bilinear','crop'); % rotate negative angle
       xx(xx<0.3*max(xx(:)))=0;
     Kent_draw_new_s(xx,p);colormap jet

    F = getframe();
    IMG_to_write = F.cdata;
    imwrite(IMG_to_write,[fileName,'_polish','.png'],'png')
end