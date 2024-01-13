% Time Domain Image reconstruction with Gauss-Newton solver
%
% This version uses time window binning
% 
% Original Author : Simon Arridge
% Modified by: Thomas Ribeiro, Stephen Fay, Raffi Hotter

function reconTD_GN(depth, separation, square_width, change, nopt, ww)


%% 0. Define parameters

% mesh parameters
rad   = 70; % mesh radius [mm]
nsect = 6;  % number of sectors
nring = 32; % number of rings
nbnd  = 2;  % number of boundary rings

% optical parameters @ 850nm
refind = 1.4;                % refractive index
c0     = 0.3;                % speed of light in vacuum [mm/ps]
cm     = c0/refind;          % speed of light in the medium [mm/ps]
mua0   = 0.02;               % background absorption [1/mm]
mus0   = 0.67;               % background scattering [1/mm];

% temporal parameters
dt    = 20;  % time step in picoseconds
nstep = 256; % number of time steps
if nargin < 6
    ww = 32;  % default value for ww -> 7 time bins of width 32 (in units of time-step)
end
twin  = zeros(nstep/ww-1,2);
twin(1:nstep/ww-1,1) = [1:ww:(nstep-ww)]';
twin(:,2) = twin(:,1)+ww-1;
nwin  = size(twin,1);

bx = 70; by = 70; % solution basis: grid dimension

% optimization parameters
tau    = 1e-2; % regularisation parameter
beta   = 0.01; % TV regularisation parameter
GNtol  = 5e-6; % Gauss-Newton convergence criterion
itrmax = 20;   % Gauss-Newton max. iterations
tol    = 1e-4;
maxit  = 100; % Krulov solver max. iterations (inside each GN iteration)

% Make the save directory if it doesn't exist
dirName = sprintf('results/circle_two_squares/depth=%d_separation=%d_square_width=%d_change=%d_nopt=%d_ww=%d', depth, separation, square_width, change, nopt,ww);

if ~exist(dirName, 'dir')
  mkdir(dirName);
end

%% 1. Define the target mesh
[vtx,idx,eltp] = mkcircle(rad,nsect,nring,nbnd);
hMesh = toastMesh (vtx,idx,eltp);
n = hMesh.NodeCount;
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
saveas(gcf, fullfile(dirName, '/sensors.png'));

%% 3. Assign optical properties

% Set up the mapper between FEM and inverse solution bases
hBasis = toastBasis (hMesh, [bx by]);
dx = 2*rad/bx; dy = 2*rad/by;

% Set up homogeneous initial parameter estimates
mua = ones(n,1) * mua0; mus = ones(n,1) * mus0; ref = ones(n,1) * refind;
kap = 1./(3*(mua+mus));

swidth    = round(square_width/dx) - 1;
s1x_start = round((bx/2 + 1) - (separation/2)/dx - swidth/2);
s2x_start = round((bx/2 + 1) + (separation/2)/dx - swidth/2);
sy_start  = round(by-depth/dy-swidth/2);

muaim = zeros(bx,by); musim = muaim;
muaim(s1x_start:s1x_start+swidth, sy_start:sy_start+swidth) = change*mua0;
muaim(s2x_start:s2x_start+swidth, sy_start:sy_start+swidth) = change*mua0;
musim(s1x_start:s1x_start+swidth, sy_start:sy_start+swidth) = change*mus0;
musim(s2x_start:s2x_start+swidth, sy_start:sy_start+swidth) = change*mus0;

mua1 = mua + hBasis.Map('B->M',muaim);
mus1 = mus + hBasis.Map('B->M',musim);

figure(2); clf;

% Display target absorption
subplot(2,2,1); 
hMesh.Display(mua1); 
axis off;
title('\mu_a target');
ax1 = gca; % Get current axes for absorption
colorLimitsMua = ax1.CLim; % Store the color scale for absorption


% Display target scattering
subplot(2,2,2); 
axis off;
hMesh.Display(mus1); 
title('\mu_s target');
ax2 = gca; % Get current axes for scattering
colorLimitsMus = ax2.CLim; % Store the color scale for scattering

%% 4. Generate target data

[data,t] = toastProjectTPSF(hMesh, mua1, mus1, ref, qvec, mvec, dt, nstep);
% data = data.*(1 + 0.02*randn(size(gamma1))); % add noise
data = reshape(WindowTPSF(data,twin)',[],1); % single vector of data, ordered by time gate.
m = length(data);

%% 5. Setup inverse solver

[proj,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep);
proj = reshape(WindowTPSF(proj,twin)',[],1); % single vector of data, ordered by time gate.
sd = proj; % Assume standard deviation equal to data

figure(3); clf; sgtitle('initial error')
for w = 1:min(nwin,8)
    dl = reshape(data((w-1)*nM*nQ+1:w*nM*nQ)-proj((w-1)*nM*nQ+1:(w)*nM*nQ),nM,nQ);
    subplot(2,4,w); imagesc(dl);
end

figure(2);

% Display reconstructed absorption
subplot(2,2,3); 
hMesh.Display(mua); 
title('\mu_a recon');
ax3 = gca; % Get current axes for reconstructed absorption
% ax3.CLim = colorLimitsMua; % Apply the same color scale

% Display reconstructed scattering
subplot(2,2,4); 
hMesh.Display(mus); 
title('\mu_s recon');
ax4 = gca; % Get current axes for reconstructed scattering
% ax4.CLim = colorLimitsMus; % Apply the same color scale

% map initial estimate of the solution vector
bmua = hBasis.Map ('M->B', mua); bcmua = bmua*cm; scmua = hBasis.Map ('B->S', bcmua);
bkap = hBasis.Map ('M->B', kap); bckap = bkap*cm; sckap = hBasis.Map ('B->S', bckap);

% solution vector
x = [scmua; sckap]; % logx = log(x);
p = length(x); 

% setup regularizer instance
% hReg = toastRegul('TK1', hBasis, x, 1);
% hReg = toastRegul('TK1',hBasis,logx,1);
hReg = toastRegul ('TV', hBasis, x, tau, 'Beta', beta);
% hReg = toastRegul ('TV', hBasis, logx, tau, 'Beta', beta);
% hReg = toastRegul ('Huber', hBasis, logx, tau, 'Eps', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, hReg, x);
err = err0; errp = 1e10;
itr = 1; step = 0.02;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 6. Run the inverse loop solver
errs = [err0]; % store the errors

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
    % hReg = toastRegul('TK1',hBasis,x,1); % no hyperparameter set here
    % hReg = toastRegul ('TV', hBasis, x, tau, 'Beta', beta);
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

    % Display reconstructed absorption
    subplot(2,2,3); 
    hMesh.Display(mua); 
    axis off;
    title('\mu_a recon');
    ax3 = gca; % Get current axes for reconstructed absorption
    % ax3.CLim = colorLimitsMua; % Apply the same color scale

    % Display reconstructed scattering
    subplot(2,2,4); 
    hMesh.Display(mus); 
    axis off;
    title('\mu_s recon');
    ax4 = gca; % Get current axes for reconstructed scattering
    % ax4.CLim = colorLimitsMus; % Apply the same color scale
    % save the files
    saveas(gcf, fullfile(dirName, 'recon.png'));
    saveas(gcf, fullfile(dirName, 'recon.fig'));

    [proj,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep);
    proj = reshape(WindowTPSF(proj,twin)',[],1); % single vector of data, ordered by time gate.
    fprintf (1, '**** GN ITERATION %d, ERROR %f\n\n', itr, err);
    itr = itr+1;
    drawnow

    % Store the error
    errs = [errs err];
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

figure(4); clf; sgtitle('final error')
for w = 1:min(nwin,8)
    dl = reshape(data((w-1)*nM*nQ+1:w*nM*nQ)-proj((w-1)*nM*nQ+1:(w)*nM*nQ),nM,nQ);
    subplot(2,4,w); imagesc(dl);
end

% save the variables
save(fullfile(dirName, 'data.mat'));
% figure; plot(fvals); xlabel('Iteration No.'); ylabel('Objective function values');
% save -v7.3 Jrecon J data y;

% plot the errors and save them
figure(5); clf;
semilogy(errs);
xlabel('Iteration No.'); ylabel('Objective function values');
saveas(gcf, fullfile(dirName, 'errors.fig'));

end