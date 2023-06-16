
function [u,errAll, dx,dy,dt] = TV_SB_2D_Time(J,f, N,mu, lambda, gamma, alpha, nInner, nBreg,varargin)
% [u,errAll] = TV_SB_2D_Time(J,f, N,mu, lambda, gamma, alpha, nInner,nBreg, uTarget) for a general Jacobian J.  
% 
% The following repository contains a faster implementation where the
% linear system is solved in the Fourier domain, see
% https://github.com/HGGM-LIM/Split-Bregman-ST-Total-Variation-MRI)
%
% TV_SB_3D solves the 3D(2D + Time) constrained total variation image reconstruction
% problem using the Split Bregman formulation. 
%
% Inputs: 
% J         = Linear operator or Jacobian that maps image pixels into
% measurements, matrix of size mxn, where m and n are the number of
% measurements and voxels, respectively    
% f         = data, a column vector mx1
% N         = image size vector [size_x size_y size_z] 
% mu        = 1 (decrease if data is noisy and increase for faste
% convergence) 
% lambda    = 1, regularization parameter
% gamma     = 1, regularization parameter
% alpha     = [1 1], sparsity weighting parameter for spatial TV for x-y 
% and t dimensions, repectively
% nInner    = 1 
% nBreg     = the number of Bregman iterations. Choose a number of
% iterations such that the method converges (for that see the solution
% error norm errAll)   
% uTarget   = target image solution of size N used to compute the error
% R         = logical indices of same size as data
%
% Outputs: 
% u         = image of size N
%
%
% It solves the constrained total variation problem 
% min_u alpha_xy||grad_x,y u||_1 + alpha_t||grad_t u||_1 st. ||Au-f||^2 < delta, 
% where A is a linear operator (a matrix) the projects the image u to the data f. 
% The code works for general linear inverse problems. It currently expects A to be a
% matrix; it can be easily modified to use A as a MATLAB function by
% changing A and A' for functions that compute forward and adjoint
% operators (eg. projection and retroprojection operations). 
%
% If you use this code, please, cite the following papers: 
%
% Abascal JF, Chamorro-Servent J, Aguirre J, Arridge S, Correia T, Ripoll
% J, Vaquero JJ, Desco M. Fluorescence diffuse optical tomography using the
% split Bregman method. Med Phys. 38(11):6275-84, 2011.   
% DOI: http://dx.doi.org/10.1118/1.3656063
%
% % JFPJ Abascal, L Cussó, J J Vaquero, M Desco. Application of the
% compressed sensing technique to self-gated cardiac cine sequences in
% small animals. Magn Reson Med., 72(2): 369–380, 2013. DOI:
% http://dx.doi.org/10.1002/mrm.24936    
%
% Code downloaded from repository: 
% https://github.com/HGGM-LIM/Split-Bregman-Total-Variation-Jacobian-Matrix
% 
% Juan FPJ Abascal
% Departamento de Bioingeniería e Ingeniería Aeroespacial
% Universidad Carlos III de Madrid, Madrid, Spain
% juanabascal78@gmail.com, juchamser@gmail.com, desco@hggm.es

% Normalize data
normFactor  = getNormalizationFactor(f,f);
f           = normFactor*f;

switch nargin
    case 10
        uTarget     = varargin{1};
end % nargin

errAll      = zeros(nBreg,1);

% Normalize Jacobian such that its Hessian diagonal is equal to 1
normFactorJ = 1/sqrt(max(diag(J'*J)));
J           = J*normFactorJ;

% Scale the forward and adjoint operations so doent depend on the size
scale       = 1/max(abs(J'*f(:,1)));

% Define forward and adjoint operators on each volume
A       = @(x)(((J*x(:)))/scale);          
AT          = @(x)(reshape((J'*x)*scale,N(1:2)));

% Krylov convergence criterion: decrease to improve precision for solving
% the linear system, increase to go faster
tolKrylov   = 10e-6; % 1e-4   

% Reserve memory for the auxillary variables
rows        = N(1);
cols        = N(2);
time        = N(3);
f0          = f;
u           = zeros(N);
x           = zeros(N);
y           = zeros(N);
t           = zeros(N);
bx          = zeros(N);
by          = zeros(N);
bt          = zeros(N);

for it = 1:time
  murf(:,:,it) = mu*AT(f(:,it));
end

%  Do the reconstruction
for outer = 1:nBreg;    
    for inner = 1:nInner;        
        % update u
        rhs     = murf+lambda*Dxt(x-bx)+lambda*Dyt(y-by)+ ...
          + lambda*Dtt(t-bt);
        
        u       = reshape(krylov(rhs(:)),N);
        
        dx      = Dx(u);
        dy      = Dy(u);
        dt      = Dt(u);
        
        % update x and y and z
        [x,y]   = shrink2(dx+bx,dy+by,alpha(1)/lambda);
        t       = shrink1(dt+bt,alpha(2)/lambda);
        
        % update bregman parameters
        bx          = bx+dx-x;
        by          = by+dy-y;
        bt          = bt+dt-t;
    end   % inner loop
    
    for it = 1:time
      fForw           = A(u(:,:,it));
      f(:,it)         = f(:,it) + f0(:,it) - fForw;
      murf(:,:,it)  = mu*AT(f(:,it));
    end
    
    if 1==2
    if nargin >= 10        
        % Solution error norm
        errAll(outer)       = norm(uTarget(:)-abs(u(:)*normFactorJ/(normFactor*scale)))/norm(uTarget(:));
        
        if any([outer ==1, outer == 10, rem(outer, 50) == 0])
            close;
            h=figure;
            subplot(2,2,1);
            imagesc(abs(u(:,:,3)*normFactorJ/(normFactor*scale))); title(['u, iter. ' num2str(outer)]); colorbar;
            subplot(2,2,2);
            imagesc(abs(x(:,:,3))); title(['x']); colorbar;
            subplot(2,2,3);
            imagesc(abs(t(:,:,3))); title(['t']); colorbar;
            subplot(2,2,4);
            plot(errAll(1:outer)); axis tight; title(['Sol. error' ]);   
            colormap gray;
            drawnow;
        end % rem
    end % nargin    
end % outer

end
% undo the normalization so that results are scaled properly
u = u*normFactorJ/(normFactor*scale);

    function normFactor = getNormalizationFactor(R,f)
        
        normFactor = 1/norm(f(:)/size(R==1,1));
        
    end

    function d = Dx(u)
        [rows,cols,time] = size(u);
        d = zeros(rows,cols,time);
        d(:,2:cols,:) = u(:,2:cols,:)-u(:,1:cols-1,:);
        d(:,1,:) = u(:,1,:)-u(:,cols,:);
    end

    function d = Dxt(u)
        [rows,cols,time] = size(u);
        d = zeros(rows,cols,time);
        d(:,1:cols-1,:) = u(:,1:cols-1,:)-u(:,2:cols,:);
        d(:,cols,:) = u(:,cols,:)-u(:,1,:);
    end

    function d = Dy(u)
        [rows,cols,time] = size(u);
        d = zeros(rows,cols,time);
        d(2:rows,:,:) = u(2:rows,:,:)-u(1:rows-1,:,:);
        d(1,:,:) = u(1,:,:)-u(rows,:,:);
    end

    function d = Dyt(u)
        [rows,cols,time] = size(u);
        d = zeros(rows,cols,time);
        d(1:rows-1,:,:) = u(1:rows-1,:,:)-u(2:rows,:,:);
        d(rows,:,:) = u(rows,:,:)-u(1,:,:);
    end
      
    function d = Dt(u) %
        [rows,cols,time] = size(u); 
        d = zeros(rows,cols,time);
        d(:,:,2:time) = u(:,:,2:time)-u(:,:,1:time-1);
        d(:,:,1) = u(:,:,1)-u(:,:,time);
    end
    
    function d = Dtt(u) 
        [rows,cols,time] = size(u); 
        d = zeros(rows,cols,time);
        d(:,:,1:time-1) = u(:,:,1:time-1)-u(:,:,2:time);
        d(:,:,time) = u(:,:,time)-u(:,:,1);
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
        [dx,flag,relres,iter] = bicgstab(@jtjx, r, tolKrylov, 100);
    end

% =====================================================================
% Callback function for matrix-vector product (called by krylov)
    function b = jtjx(sol)
        solMat  = reshape(sol,N);
        
        % Laplacian part
        bTV     = lambda*(Dxt(Dx(solMat))+Dyt(Dy(solMat)) +...
          Dtt(Dt(solMat)));
        
        % Jacobian part
        for it = 1:time
            tmp             = solMat(:,:,it);
            bJac(:,:,it)  = mu*AT(A(tmp(:)));
        end
        
        % Stability term
        bG      = gamma*sol;
        
        b       = bTV(:) + bJac(:) + bG(:);        
    end
% =====================================================================
end

%