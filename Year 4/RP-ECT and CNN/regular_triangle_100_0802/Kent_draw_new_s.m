function Kent_draw_new_s(k,l);

for index=0:(l^2)-1;
    i = fix(index/l)+1;
    j = mod(index,l)+1;
    image(i,j) = k(index+1);
end 


% Declaration of x and y axes
%x = [-l/2+1:1:l/2];
%y = [-l/2+1:1:l/2];
x = [-80:160/(l-1):80];
y = [-80:160/(l-1):80];

        
% Create x and y axes with 200 points
xlin = linspace(min(x),max(x),200);
ylin = linspace(min(y),max(y),200);

% Form a 2D matrix of plotting points
[X,Y] = ndgrid(xlin,ylin);

% Create a 3D matrix using interpolation of impedance matrix.
Z = griddata(x,y,image,X,Y,'cubic');

% Produce a circular map 
ydia = length(Y);
xdia = length(X);
yrad = length(Y)/2;
xrad = length(X)/2;

% for f=1:ydia
%     for g=1:xdia
%         if ((f-yrad)^2)+((g-xrad)^2)>(yrad*xrad)
%            Z(f,g)=NaN;
%         end
%     end
% end

surf(X,Y,Z)

axis tight
view(0,90);
axis square
shading interp
axis off
%xlabel(['X-Direction']);
%ylabel(['Y-Direction']);
%zlabel(['Impedance']);

