%% MATLAB-TOAST sample script:
% Time Domain Image reconstruction with Gauss-Newton solver
% version adds noise, and uses lower resolution reconstruction mesh
%
% This version uses time window binning
% 
% Author : Simon Arridge 09-Sept-2023
% ======================================================================
% User-defined parameters
% ======================================================================

function reconTD_GN(depth, separation, resol, change, nopt)

%% 0. Define parameters

% remove parameters
nopt       = 8;   % number of sources (and detectors)
depth      = 10;  % [mm]
separation = 20;  % [mm]
change     = 2;   % percent change in optical properties

% mesh parameters
rad   = 40; % mesh radius [mm]
nsect = 6;  % number of sectors
nring = 32; % number of rings
nbnd  = 2;  % number of boundary rings

% optical parameters @ 850nm
refind = 1.4;                % refractive index
c0     = 0.3;                % speed of light in vacuum [mm/ps]
cm     = c0/refind;          % speed of light in the medium [mm/ps]
mua0   = 0.02;               % background absorption [1/mm]
mus0   = 0.67;               % background scattering [1/mm];
kap0   = 1./(3*(mua0+mus0)); % diffusion coefficient

% temporal parameters
dt    = 20;  % time step in picoseconds
nstep = 256; % number of time steps
ww    = 32;  % -> 7 time bins of width 32 (in units of time-step)
twin  = zeros(nstep/ww-1,2); twin(1:nstep/ww-1,1) = [1:ww:(nstep-ww)]';
twin(:,2) = twin(:,1)+ww-1;  
nwin  = size(twin,1);

% solution basis: grid dimension
bx = 64; by = 64;                    
blen = bx*by;

% optimization parameters
tau    = 1e-2;    % regularisation parameter
beta   = 0.01; % TV regularisation parameter
GNtol  = 5e-4; % Gauss-Newton convergence criterion
itrmax = 20;   % Gauss-Newton max. iterations
tol    = 1e-4;
maxit  = 100;

%% 1. Define the target mesh
[vtx,idx,eltp] = mkcircle(rad,nsect,nring,nbnd);
hMesh = toastMesh (vtx,idx,eltp);
n = hMesh.NodeCount;
dmask = hMesh.DataLinkList ();
bb = hMesh.BoundingBox();
bmin = bb(1,:); bmax = bb(2,:);
vscale = (bmax(1)-bmin(1))/bx * (bmax(2)-bmin(2))/by * cm;
hMesh.Display

%% 2. Define the source and sensor geometry
distrib = 'quarter';
for i=1:nopt
  switch distrib
      case 'full'
          phiq = (i-1)/nopt*2*pi; 
          phim = (i-0.5)/nopt*2*pi;
      case 'quarter'
          phiq = (i-1)/nopt*pi/2+pi/4;
          phim = (i-0.5)/nopt*pi/2+pi/4;
  end
  source(i,:) = rad*[cos(phiq), sin(phiq)];
  sensor(i,:) = rad*[cos(phim), sin(phim)];
end

hMesh.SetQM(source,sensor);
qvec = real(hMesh.Qvec('Neumann','Gaussian',2)); nQ = size(qvec,2);
mvec = real(hMesh.Mvec('Gaussian',2,refind)); nM = size(mvec,2);

hold on
plot(source(:,1),source(:,2),'ro','MarkerFaceColor','r');
plot(sensor(:,1),sensor(:,2),'bsquare','MarkerFaceColor','b');
hold off
legend('','Source', 'Sensor');

%% 3. Assign optical properties

% Set up the mapper between FEM and inverse solution bases
hBasis = toastBasis (hMesh, [bx by]);
dx = 2*rad/by; dy = 2*rad/by;
% solmask = hBasis.Map('S->B',ones(hBasis.slen,1));
% figure; imagesc(reshape(solmask,bx,by)); title('solution mask');

% Set up homogeneous initial parameter estimates
mua = ones(n,1) * mua0; mus = ones(n,1) * mus0; ref = ones(n,1) * refind;
kap = 1./(3*(mua+mus));

% Setup up heterogenous parameter estimates
swidth    = int32(8/dx);
s1x_start = int32(1+bx/2 - (separation/2)/dx - swidth/2);
s2x_start = int32(bx/2 + (separation/2)/dx - swidth/2);
sy_start  = int32(by-depth/dy-swidth);

muaim = zeros(bx,by); musim = muaim;
muaim(s1x_start:s1x_start+swidth, sy_start:sy_start+swidth) = change*mua0;
muaim(s2x_start:s2x_start+swidth, sy_start:sy_start+swidth) = change*mua0;
musim(s1x_start:s1x_start+swidth, sy_start:sy_start+swidth) = change*mus0;
musim(s2x_start:s2x_start+swidth, sy_start:sy_start+swidth) = change*mus0;

mua1 = mua + hBasis.Map('B->M',muaim);
mus1 = mus + hBasis.Map('B->M',musim);

figure(2); clf;
subplot(2,2,1); hMesh.Display(mua1); axis off; title('\mu_a target');
subplot(2,2,2); hMesh.Display(mus1); axis off; title('\mu_s target');

% figure(4);clf;
% subplot(1,2,1); imagesc(reshape(hBasis.Map('M->B',mua0*ones(n,1)),bx,by) + muaim);title('\mu_a target');
% subplot(1,2,2); imagesc(reshape(hBasis.Map('M->B',mus0*ones(n,1)),bx,by) + musim);title('\mu_s target');

%% 4. Generate target data

[data,t] = toastProjectTPSF(hMesh, mua1, mus1, ref, qvec, mvec, dt, nstep);
% data = data.*(1 + 0.02*randn(size(gamma1))); % add noise
data = reshape(WindowTPSF(data,twin)',[],1); % single vector of data, ordered by time gate.
m = length(data);

%% 5. Setup inverse solver

[proj,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep);
proj = reshape(WindowTPSF(proj,twin)',[],1); % single vector of data, ordered by time gate.
sd = proj; % Assume standard deviation equal to data

figure(3); clf; sgtitle('(Target - Initial) Data')
for w = 1:min(nwin,8)
    dl = reshape(data((w-1)*nM*nQ+1:w*nM*nQ)-proj((w-1)*nM*nQ+1:(w)*nM*nQ),nM,nQ);
    subplot(2,4,w); imagesc(dl);
end

% map initial estimate of the solution vector
% bmus = hBasis.Map('M->B', mus);
bmua = hBasis.Map ('M->B', mua); bcmua = bmua*cm; scmua = hBasis.Map ('B->S', bcmua);
bkap = hBasis.Map ('M->B', kap); bckap = bkap*cm; sckap = hBasis.Map ('B->S', bckap);
% scmua = bcmua(find(solmask==1));
% sckap = bckap(find(solmask==1));

% solution vector
x = [scmua; sckap]; % logx = log(x);
p = length(x); 

% setup regularizer instance
hReg = toastRegul('TK1', hBasis, x, 1);
%hReg = toastRegul('TK1',hBasis,logx,1);
%hReg = toastRegul ('TV', hBasis, logx, tau, 'Beta', beta);
%hReg = toastRegul ('Huber', hBasis, logx, tau, 'Eps', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, hReg, x);
err = err0; errp = 1e10;
itr = 1; step = 0.02;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 6. Run the inverse loop solver

% Gauss-Newton loop
%itrmax = 1;
while (itr <= itrmax) & (err > GNtol*err0) & (errp-err > GNtol)

    % Construct the Jacobian
    fprintf (1,'Calculating Jacobian\n');
    J = toastJacobianTimedomain (hMesh, hBasis, qvec, mvec, mua, mus, ref, dt, nstep, twin, 'direct')*vscale;

    J = spdiags(1./sd,0,m,m) * J; % data normalisation
    J = J * spdiags (x,0,p,p);    % parameter normalisation
    trJ = sum(sum(J.^2));

    % nsol = size(J,2)/2;
    % figure(5);clf;
    % for w = 1:min(nwin,8)
    %     dl = reshape(hBasis.Map('S->B',J(nM/2+(w-1)*nM*nQ,1:nsol)),bx,by);
    %     subplot(2,4,w); imagesc(dl);
    % end
    % figure(6);clf;
    % for w = 1:min(nwin,8)
    %     dl = reshape(hBasis.Map('S->B',J(nM/2+(w-1)*nM*nQ,nsol+1:2*nsol)),bx,by);
    %     subplot(2,4,w); imagesc(dl);
    % end

    M = ones(1,size(J,2));               % override preconditioner
    hReg = toastRegul('TK1',hBasis,x,1); % no hyperparameter set here
    Hr = hReg.Hess(x);
    y = (data-proj)./sd;
    r = J' * y;                          % gradient of cost function
    Hr = hReg.Hess(x);                   % Hessian of regulariser
    
    % Update with implicit Krylov solver
    fprintf (1, 'Entering Krylov solver\n');
    S = speye(size(M,2));
    dx = S*pcg(@(x)JTJH(x,J*S,Hr,trJ*tau),-S*r,tol,maxit);   
    dx = dx.*x;

    % Line search
    fprintf (1, 'Entering line search\n');
    errp = err;
    [step, err] = toastLineSearch (x, dx, step, err, @objective);
    
    % Add update to solution
    x = x+step*dx;    

    % Map parameters back to mesh
    scmua = x(1:size(x)/2); smua = scmua/cm; mua = hBasis.Map ('S->M', smua);
    sckap = x(size(x)/2+1:size(x)); skap = sckap/cm;  
    smus = 1./(3*skap) - smua; mus = hBasis.Map ('S->M', smus);
    figure(2);
    subplot(2,2,3); hMesh.Display(mua); axis off; title('\mu_a recon');
    subplot(2,2,4); hMesh.Display(mus); axis off; title('\mu_s recon');

    [proj,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep);
    proj = reshape(WindowTPSF(proj,twin)',[],1); % single vector of data, ordered by time gate.
    fprintf (1, '**** GN ITERATION %d, ERROR %f\n\n', itr, err);
    itr = itr+1;
    drawnow
end

    function p = objective(x)
        
        % Map parameters back to mesh
        smua_ = x(1:size(x)/2)/cm; mua_ = hBasis.Map('S->M', smua_);
        skap_ = x(size(x)/2+1:size(x))/cm;
        smus_ = 1./(3*skap_) - smua_; mus_ = hBasis.Map ('S->M', smus_);
        % for j = 1:length(mua) % ensure positivity
        %     mua(j) = max(1e-4,mua(j));
        %     mus(j) = max(0.2,mus(j));
        % end

        [proj_,t] = toastProjectTPSF(hMesh, mua_, mus_, ref, qvec, mvec, dt, nstep);
        proj_ = reshape(WindowTPSF(proj_, twin)',[],1); % single vector of data, ordered by time gate.
        [p, p_data, p_prior] = toastObjective (proj_, data, sd, hReg, x);
        fprintf (1, '    [LH: %f, PR: %f]\n', p_data, p_prior);
    end
% save -v7.3 Jrecon J data y;
end