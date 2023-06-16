clear all
close all
load S
J=S;

J1=reshape(J,[66,20,20,20]);
J2D=mean(J1(:,2:19,:,:),2);

x=J2D(66,:);
JJ2=[];
for k=1:16
    
    angle=22.5*(k-1);
for i=1:66
    
    x=J2D(i,:);
    
    x=reshape(x,[20 20]);
    y=imrotate(x,-double(angle),'bilinear','crop');
    JJ1(i,:)=y(:);
    
end

JJ2=[JJ2;JJ1];
end

save JJ2 JJ2
% Kent_draw_new_s(x,20);colormap jet
% return
% 
% folder = uigetdir(pwd, 'Select folder');
% if folder == 0
% 	return;
% end
% 
% dataFiles = [dir(fullfile(folder,'tri100_0_*.mat'))];
% numberOfFiles = length(dataFiles);
% 
% for i=1:numberOfFiles
%     for j=1:66
%         z
%         