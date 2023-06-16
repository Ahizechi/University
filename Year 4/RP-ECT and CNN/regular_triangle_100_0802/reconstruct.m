clear
close all

load S
load ref1
load bb

p = 20;
A = S;
N = [p p p];

% Recover the image
mu      = 3;
lambda  = 100;
gamma   = 100;
nInner  = 1;
nBreg   = 3;

R=ones(66,1);
data2 = ref1;
% Form the CS data
data    = R.*data2;

u       = reshape(A'*data2,N);

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

    [n_row,n_column] = size(framedata);

     for frame = 1:n_row
        plane2 = framedata(frame,:)';
        b=-((plane2)-(bb'))./1;
        if frame==2
            bb=plane2';
        end
        b(13)=0;
        [u,errAll] = mrics3D_FDOT_SB_NB_P0_GNK(A,b,R,N,mu, lambda, gamma, nInner, nBreg);
    %       u(u<0)=0;
        uu(:,:,:,frame)=u;
        subplot(1,3,1)
       y1=u(:,:,10);
    yy1=reshape(y1,[20 20]);
    imagesc(yy1');shading interp;colormap jet;axis equal; axis off;title(frame)
    subplot(1,3,2)
    y2=u(:,10,:);
    yy2=reshape(y2,[20 20]);
    imagesc(yy2');shading interp;colormap jet;axis equal;axis off; title(frame)
    subplot(1,3,3)
    y3=u(5,:,:);
    yy3=reshape(y3,[20 20]);
    image = imagesc(yy3');shading interp;colormap jet;axis equal; axis off;colorbar; title(frame)

       title(frame)

            drawnow

        if frame == 5    
            F = getframe();
            IMG_to_write = F.cdata;
        %         imwrite(IMG_to_write,['frame_',num2str(frame),'.jpg'],'jpg')
            imwrite(IMG_to_write,[fileName,'.jpg'],'jpg')
        end
     end
end