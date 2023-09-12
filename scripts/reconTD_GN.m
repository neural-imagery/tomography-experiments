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

function reconCW_CG(depth, separation, resol, change, nopt)
meshpath = './';
%datapath = './';
% meshname = 'circle32.msh';
% qmname   = 'circle25_32x32.qm';      % QM file
refind   = 1.4;                      % refractive index
bx = 64; by = 64;                    % solution basis: grid dimension
tau = 1;                             % regularisation parameter
beta = 0.01;                         % TV regularisation parameter
GNtol = 5e-4;                        % Gauss-Newton convergence criterion
itrmax = 20;                         % Gauss-Newton max. iterations
dt = 20;                             % time step in picoseconds
nstep = 256;                         % number of time steps
tol = 1e-6;
maxit = 100;
rad = 40;

% set the time bins :
% this example uses who time width divided into non-overalapping bins set
% by width variable

ww = 32; % this will lead to 7 bins of width 32 (in units of time-step)
twin = zeros(nstep/ww-1,2); twin(1:nstep/ww-1,1) = [1:ww:(nstep-ww)]';  twin(:,2) = twin(:,1)+ww-1;  
% ======================================================================
% End user-defined parameters
% ======================================================================
%%
% Initialisations

% Set up some variables
blen = bx*by;
c0 = 0.3;
cm = c0/refind;

% Read a TOAST mesh definition from file.
file = 'meshes/circle_two_squares/two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
hMesh = toastMesh (file,'gmsh');
lprm.hMesh = hMesh;
n = hMesh.NodeCount();
dmask = hMesh.DataLinkList ();
bb = hMesh.BoundingBox();
bmin = bb(1,:); bmax = bb(2,:);
vscale = (bmax(1)-bmin(1))/bx * (bmax(2)-bmin(2))/by * cm;
hMesh.Display

%% Define the source and sensor geometry
distrib = 'full';
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
qvec = real(hMesh.Qvec('Neumann','Gaussian',2));
nQ = size(qvec,2);
lprm.qvec = qvec;

mvec = real(hMesh.Mvec('Gaussian',2,refind));
lprm.mvec= mvec;
nM = size(mvec,2);

hold on
plot(source(:,1),source(:,2),'ro','MarkerFaceColor','r');
plot(sensor(:,1),sensor(:,2),'bsquare','MarkerFaceColor','b');
hold off
legend('','Source', 'Sensor');


% Set up homogeneous initial parameter estimates
mua0 = 0.025;
mus0 = 2;
mua = ones(n,1) * mua0;
mus = ones(n,1) * mus0;
ref = ones(n,1) * refind;
kap = 1./(3*(mua+mus));
nwin = size(twin,1);

lprm.cm = cm;
lprm.ref = ref;
lprm.freq = 0;
lprm.dmask = dmask;
lprm.dt = dt;
lprm.nstep = nstep;
lprm.twin = twin;

%% Set up the mapper between FEM and solution bases
hBasis = toastBasis (hMesh,[bx by]);
lprm.hBasis = hBasis;
solmask = hBasis.Map('S->B',ones(hBasis.slen,1));
figure; imagesc(reshape(solmask,bx,by)); title('solution mask');
solmask2 = [solmask solmask+blen];

%% data
% synthesise some data

muaim = zeros(bx,by);
musim = muaim;

dx = 2*rad/by; dy = 2*rad/by;
square_width = int32(8/dx); %[units]

square1X_startIdx = int32(1+bx/2 - (separation/2)/dx - square_width/2);
square2X_startIdx = int32(bx/2 + (separation/2)/dx - square_width/2);
squareY_startIdx = int32(by-depth/dy-square_width);

muaim(squareY_startIdx:squareY_startIdx+square_width, ...
    square1X_startIdx:square1X_startIdx+square_width) = change*mua0;
muaim(squareY_startIdx:squareY_startIdx+square_width, ...
    square2X_startIdx:square2X_startIdx+square_width) = change*mua0;

musim(squareY_startIdx:squareY_startIdx+square_width, ...
    square1X_startIdx:square1X_startIdx+square_width) = change*mus0;
musim(squareY_startIdx:squareY_startIdx+square_width, ...
    square2X_startIdx:square2X_startIdx+square_width) = change*mus0;

figure(4);clf;
subplot(1,2,1); imagesc(reshape(hBasis.Map('M->B',mua0*ones(n,1)),bx,by) + muaim);title('\mu_a target');
subplot(1,2,2); imagesc(reshape(hBasis.Map('M->B',mus0*ones(n,1)),bx,by) + musim);title('\mu_s target');

mua1 = mua + hBasis.Map('B->M',muaim);
mus1 = mus + hBasis.Map('B->M',musim);

% Call the forward projector
[gamma1,t] = toastProjectTPSF(hMesh, mua1, mus1, ref, qvec, mvec,dt,nstep);

% add noise
gamma1 = gamma1.*(1 + 0.02*randn(size(gamma1)));
data = reshape(WindowTPSF(gamma1,twin)',[],1); % single vector of data, ordered by time gate.
clear gamma1;

lprm.data = data;
m = length(data);


%% Initial data set f[x0]

[gamma,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep);
proj = reshape(WindowTPSF(gamma,twin)',[],1); % single vector of data, ordered by time gate.


%% data scaling

sd = proj; % Assume standard deviation equal to data
lprm.sd = sd;

% Visualise data

nwin = size(twin,1);
figure(3);clf;
for w = 1:min(nwin,8);
    dl = reshape(data((w-1)*nM*nQ+1:w*nM*nQ)-proj((w-1)*nM*nQ+1:(w)*nM*nQ),nM,nQ);
    subplot(2,4,w);
    imagesc(dl);
end

%% initial parameter estimates in solution basis
bmua = hBasis.Map('M->B', mua);
bmus = hBasis.Map('M->B', mus);
bmua_itr(1,:) = bmua;
bmus_itr(1,:) = bmus;
bkap = hBasis.Map('M->B', kap);
bcmua = bmua*cm;
bckap = bkap*cm;
scmua = bcmua(find(solmask==1));
sckap = bckap(find(solmask==1));

kap1 = 1./(3*(mua1+mus1));
kap0 = 1./(3*(mua+mus));

dkap = hBasis.Map('M->S',kap1 - kap0);
dmua = hBasis.Map('M->S',mua1 - mua);

x = [scmua;sckap];
logx = log(x);
p = length(x); 
step = 1.0; % initial step length for line search

%% Initialise regularisation
%hReg = toastRegul('TK1',hBasis,logx,1); % no hyperparameter set here
hReg = toastRegul('TK1',hBasis,x,1); % no hyperparameter set here
%hReg = toastRegul ('TV', hBasis, logx, tau, 'Beta', beta);
%hReg = toastRegul ('Huber', hBasis, logx, tau, 'Eps', beta);
lprm.hReg = hReg;

x0 = cm*[hBasis.Map('M->S',mua);hBasis.Map('M->S',kap0)];
x = x0;

% initial data error (=2 due to data scaling)

err0 = toastObjective (proj, data, sd, hReg, x); %initial error
err = err0;                                        % current error
errp = 1e10;                                       % previous error
itr = 1; % iteration counter
fprintf (1, '\n**** INITIAL ERROR %f\n\n', err);
% Gauss-Newton loop
%itrmax = 1;
while (itr <= itrmax) & (err > GNtol*err0) & (errp-err > GNtol)

    % Construct the Jacobian
    fprintf (1,'Calculating Jacobian\n');
    J = toastJacobianTimedomain (hMesh, hBasis, qvec, mvec, mua, mus, ref, dt,nstep,twin,'direct')*vscale;
    lprm.sd = sd;

    J = spdiags(1./sd,0,m,m) * J;  % data normalisation
    J = J * spdiags (x0,0,p,p);     % parameter normalisation
    trJ = sum(sum(J.^2));

    nsol = size(J,2)/2;
    figure(5);clf;
    for w = 1:min(nwin,8)
        dl = reshape(hBasis.Map('S->B',J(nM/2+(w-1)*nM*nQ,1:nsol)),bx,by);
        subplot(2,4,w);
        imagesc(dl);
    end
    figure(6);clf;
    for w = 1:min(nwin,8)
        dl = reshape(hBasis.Map('S->B',J(nM/2+(w-1)*nM*nQ,nsol+1:2*nsol)),bx,by);
        subplot(2,4,w);
        imagesc(dl);
    end

    M = ones(1,size(J,2)); % override preconditioner

    hReg = toastRegul('TK1',hBasis,x0,1); % no hyperparameter set here
    Hr = hReg.Hess(x0);

% Gradient of cost function
    tau = 1e-2;
    y = (data-proj)./sd;
    r = J' * y;
    
    % Hessian of regulariser
    Hr = hReg.Hess(x0);
    
    % Update with implicit Krylov solver
    fprintf (1, 'Entering Krylov solver\n');
    tol = 1e-4; maxit = 100;
    S = speye(size(M,2));
    
    dx = S*pcg(@(x)JTJH(x,J*S,Hr,trJ*tau),-S*r,tol,maxit);   
    dx = dx.*x0;

    
    % Line search
    fprintf (1, 'Entering line search\n');
    errp = err;
    step = 0.02;

    [step, err] = toastLineSearch (x0, dx, step, err, @objective);
    
    % Add update to solution
    x = x0+step*dx;    

    % Map parameters back to mesh
    scmua = x(1:size(x)/2);
    sckap = x(size(x)/2+1:size(x));
    smua = scmua/cm;
    skap = sckap/cm;
    smus = 1./(3*skap) - smua;
    mua = hBasis.Map('S->M', smua);
    mus = hBasis.Map('S->M',smus);
    bmua(find(solmask==1)) = smua;
    bmus(find(solmask==1)) = smus;
    figure(4);clf;
    subplot(2,2,1); imagesc(reshape(hBasis.Map('M->B',mua0*ones(n,1)),bx,by) + muaim);title('\mu_a target');colorbar;
    subplot(2,2,2); imagesc(reshape(hBasis.Map('M->B',mus0*ones(n,1)),bx,by) + musim);title('\mu_s target');colorbar;
    subplot(2,2,3), imagesc(reshape(bmua,bx,by),[min(mua) max(mua)]), axis equal, axis tight, colorbar('vert');
    subplot(2,2,4), imagesc(reshape(bmus,bx,by),[min(mus) max(mus)]), axis equal, axis tight, colorbar('vert');
    drawnow
    
    [gamma,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep);
    proj = reshape(WindowTPSF(gamma,twin)',[],1); % single vector of data, ordered by time gate.
    x0 = x;
    itr = itr+1;
    bmua_itr(itr,:) = bmua;
    bmus_itr(itr,:) = bmus;
    fprintf (1, '**** GN ITERATION %d, ERROR %f\n\n', itr, err);
end

    function p = objective(x)
    
        % global lprm; % a set of local parameters
        
        % Map parameters back to mesh
        smua = x(1:size(x)/2)/lprm.cm;
        skap = x(size(x)/2+1:size(x))/lprm.cm;
        smus = 1./(3*skap) - smua;
        mua = lprm.hBasis.Map('S->M', smua);
        mus = lprm.hBasis.Map('S->M', smus);
        for j = 1:length(mua) % ensure positivity
            mua(j) = max(1e-4,mua(j));
            mus(j) = max(0.2,mus(j));
        end
       
        [gamma,t] = toastProjectTPSF(lprm.hMesh, mua, mus, lprm.ref, lprm.qvec, lprm.mvec,lprm.dt,lprm.nstep);
         proj = reshape(WindowTPSF(gamma,lprm.twin)',[],1); % single vector of data, ordered by time gate.
    
        [p, p_data, p_prior] = toastObjective (proj, lprm.data, lprm.sd,lprm.hReg, x);
        fprintf (1, '    [LH: %f, PR: %f]\n', p_data, p_prior);
    end
save -v7.3 Jrecon J data y;
end