
function [u,errAll] = mrics3D_FDOT_SB_NB_P0_GNK(J,f, R,N,mu, lambda, gamma, nInner, nBreg,varargin)
%
% A         = measurementsxpixels (mxn) Jacobian
% f         = data=mx1
% R         = logival indices for data entries=N x numProy x time
% N         = size in pixels=NxNxN
% mu        = 1 (decrease if data is noisy to 0.5 or 0.1)
% lambda    = 1
% gamma     = 1
% nInner    = inner iterations, 1 is fine
% nBreg     = Bregman iterations, 20, 50 200, choose for data type?
% varargin  = {uTarget} = solution 
%
% J F P J Abascal 2014

switch nargin
    case 10
        uTarget     = varargin{1};
end % nargin

rows        = N(1);
cols        = N(2);
height      = N(3);

errAll      = zeros(nBreg,1);

% Normalize data
normFactor  = getNormalizationFactor(f,f);
f           = normFactor*f;

f           = f.*R;

% Nornalize Jacobian such that its Hessian diagonal is equal to 1
normFactorJ = 1/sqrt(max(diag(J'*J)));
J           = J*normFactorJ;

% Scale the forward and adjoint operations so doent depend on the size
scale       = 1/max(abs(J'*f));

tolKrylov   = 8e-2;% 1e-2                    % Krylov convergence criterion

% Reserve memory for the auxillary variables
f0          = f;
u           = zeros(N);
x           = zeros(N);
y           = zeros(N);
z           = zeros(N);
v           = zeros(N);

bx          = zeros(N);
by          = zeros(N);
bz          = zeros(N);
bv          = zeros(N);

murf        = reshape(mu*(J'*f),N)*scale;

%  Do the reconstruction
for outer = 1:nBreg;
    
    for inner = 1:nInner;
        
        % update u
        rhs     = murf+lambda*Dxt(x-bx)+lambda*Dyt(y-by)+lambda*Dzt(z-bz)+gamma*(v-bv);
        
        u           = reshape(krylov(rhs(:)),N);
        
        %u(u<0)=0;
        
        v           = u;
        for iz = 1:N(3)
            v(:,:,iz) = medfilt2(v(:,:,iz));
        end
        v(1,:,:) = 0; v(:,:,end) = 0; v(:,1,:) = 0; v(:,end,:) = 0; v(:,:,1) = 0; v(:,:,end) = 0;
        
        dx          = Dx(u);
        dy          = Dy(u);
        dz          = Dz(u);
        
        % update x and y
        [x,y]       = shrink2(dx+bx,dy+by,1/lambda);
        z           = shrink1(dz+bz,1/lambda);
        
        % update bregman parameters
        bx          = bx+dx-x;
        by          = by+dy-y;
        bz          = bz+dz-z;
        
        bv          = bv+u-v;
    end   % inner loop
    
    fForw               = ((J*u(:)).*R)/scale;
    f                   = f + f0-fForw;
    murf                = reshape(mu*scale*(J'*f),N);
    
    if nargin >= 10
        
        % Solution error norm
        errAll(outer)       = norm(uTarget(:)-abs(u(:)*normFactorJ/(normFactor*scale)))/norm(uTarget(:));
        
        if any([outer ==1, outer == 5, outer == 10, outer == 50, rem(outer, 100) == 0])
             close all;
            c = max(abs(u(:)));
           % h=figure;
%             subplot(4,3,1);
%             imagesc((u(:,:,1))); title(['u 1']); colorbar; caxis([-c c]);
%             subplot(4,3,2);
%             imagesc(u(:,:,5)); title(['u 5']); colorbar;  caxis([-c c]);
%             subplot(4,3,3);
%             imagesc(u(:,:,8)); title(['u 8']); colorbar;caxis([-c c]);
%             subplot(4,3,4);
%             imagesc(u(:,:,11)); title(['u 11']); colorbar;caxis([-c c]);
%             subplot(4,3,5);
%             imagesc((u(:,:,14))); title(['u 14']); colorbar; caxis([-c c]);
%             subplot(4,3,6);
%             imagesc(u(:,:,15)); title(['u 15']); colorbar;  caxis([-c c]);
%             subplot(4,3,7);
%             imagesc(x(:,:,1)); title(['x 1']); colorbar;
%             subplot(4,3,8);
%             imagesc(x(:,:,5)); title(['x 5']); colorbar;
%             subplot(4,3,9);
            % imagesc(u(:,:,8)); %title(['x 8']); colorbar;colormap jet;caxis([-c c]);
%             subplot(4,3,10);
%             imagesc(x(:,:,11)); title(['x 1']); colorbar;
%             subplot(4,3,11);
%             imagesc(x(:,:,14)); title(['x 14']); colorbar;
%             subplot(4,3,12);
%             imagesc(x(:,:,15)); title(['x 15']); colorbar;
            %drawnow;
%             h=figure;
%             subplot(4,3,1);
%             imagesc((v(:,:,1))); title(['u 1']); colorbar; caxis([-c c]);
%             subplot(4,3,2);
%             imagesc(v(:,:,5)); title(['u 5']); colorbar;  caxis([-c c]);
%             subplot(4,3,3);
%             imagesc(v(:,:,8)); title(['u 8']); colorbar;caxis([-c c]);
%             subplot(4,3,4);
%             imagesc(v(:,:,11)); title(['u 11']); colorbar;caxis([-c c]);
%             subplot(4,3,5);
%             imagesc((v(:,:,14))); title(['u 14']); colorbar; caxis([-c c]);
%             subplot(4,3,6);
%             imagesc(v(:,:,20)); title(['u 20']); colorbar;  caxis([-c c]);
        end % rem
    end % nargin
    
end % outer

% undo the normalization so that results are scaled properly
u = u*normFactorJ/(normFactor*scale);

    function normFactor = getNormalizationFactor(R,f)
        
        normFactor = 1/norm(f(:)/size(R==1,1));
        
    end

    function d = Dx(u)
        [rows,cols,height] = size(u);
        d = zeros(rows,cols,height);
        d(:,2:cols,:) = u(:,2:cols,:)-u(:,1:cols-1,:);
%         d(1,:,:) = 0; d(:,:,end) = 0; d(:,1,:) = 0; d(:,end,:) = 0; d(:,:,1) = 0; d(:,:,end) = 0;
        %d(:,1,:) = u(:,1,:)-u(:,cols,:);
        %d(:,1,:) = u(:,1,:)-u(:,2,:);
    end

    function d = Dxt(u)
        [rows,cols,height] = size(u);
        d = zeros(rows,cols,height);
        d(:,1:cols-1,:) = u(:,1:cols-1,:)-u(:,2:cols,:);
%         d(1,:,:) = 0; d(:,:,end) = 0; d(:,1,:) = 0; d(:,end,:) = 0; d(:,:,1) = 0; d(:,:,end) = 0;
        %d(:,cols,:) = u(:,cols,:)-u(:,1,:);
        %d(:,cols,:) = u(:,cols,:)-u(:,cols-1,:);
    end

    function d = Dy(u)
        [rows,cols,height] = size(u);
        d = zeros(rows,cols,height);
        d(2:rows,:,:) = u(2:rows,:,:)-u(1:rows-1,:,:);
%         d(1,:,:) = 0; d(:,:,end) = 0; d(:,1,:) = 0; d(:,end,:) = 0; d(:,:,1) = 0; d(:,:,end) = 0;
        %d(1,:,:) = u(1,:,:)-u(rows,:,:);
        %d(1,:,:) = u(1,:,:)-u(2,:,:);
    end

    function d = Dyt(u)
        [rows,cols,height] = size(u);
        d = zeros(rows,cols,height);
        d(1:rows-1,:,:) = u(1:rows-1,:,:)-u(2:rows,:,:);
%         d(1,:,:) = 0; d(:,:,end) = 0; d(:,1,:) = 0; d(:,end,:) = 0; d(:,:,1) = 0; d(:,:,end) = 0;
        %d(rows,:,:) = u(rows,:,:)-u(1,:,:);
        %d(rows,:,:) = u(rows,:,:)-u(rows-1,:,:);
    end

    function d = Dz(u) % Time derivative for 3D matrix
        [rows,cols,height] = size(u); 
        d = zeros(rows,cols,height);
        d(:,:,2:height) = u(:,:,2:height)-u(:,:,1:height-1);
%         d(1,:,:) = 0; d(:,:,end) = 0; d(:,1,:) = 0; d(:,end,:) = 0; d(:,:,1) = 0; d(:,:,end) = 0;
        %d(:,:,1) = u(:,:,1)-u(:,:,height);
        %d(:,:,1) = u(:,:,1)-u(:,:,2);

    end

    function d = Dzt(u) % Time derivative for 3D matrix, transpose
        [rows,cols,height] = size(u); 
        d = zeros(rows,cols,height);
        d(:,:,1:height-1) = u(:,:,1:height-1)-u(:,:,2:height);
%         d(1,:,:) = 0; d(:,:,end) = 0; d(:,1,:) = 0; d(:,end,:) = 0; d(:,:,1) = 0; d(:,:,end) = 0;
        %d(:,:,height) = u(:,:,height)-u(:,:,1);
        %d(:,:,height) = u(:,:,height)-u(:,:,height-1);
    end

    function [xs,ys] = shrink2(x,y,lambda)
        
        s = sqrt(x.*conj(x)+y.*conj(y));
        ss = s-lambda;
        ss = ss.*(ss>0);
        
        s = s+(s<lambda);
        ss = ss./s;
        
        xs = ss.*x;
        ys = ss.*y;
        
    end

    function xs = shrink1(x,lambda)
        s = abs(x);
        xs = sign(x).*max(s-lambda,0);
    end

% =====================================================================
% Krylov solver subroutine
% X = GMRES(A,B,RESTART,TOL,MAXIT,M)
% bicgstab(A,b,tol,maxit)
    function dx = krylov(r)
        %dx = gmres (@jtjx, r, 30, tolKrylov, 100);
        [dx, tmp] = bicgstab(@jtjx, r, tolKrylov, 100);
    end


% =====================================================================
% Callback function for matrix-vector product (called by krylov)

    function b = jtjx(sol)
        solMat  = reshape(sol,N);
        
        % Laplacian part
        bTV     = lambda*(Dxt(Dx(solMat))+Dyt(Dy(solMat))+Dzt(Dz(solMat)));
        
        % Jacobian u part
        bJac    = J*sol;
        bJac    = mu*(J'*bJac);
        
        % Stability term
        bG      = gamma*sol;
        
        b       = bTV(:) + bJac(:) + bG(:);
        
    end

% =====================================================================
end

%