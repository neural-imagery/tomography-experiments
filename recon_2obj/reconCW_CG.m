%% Diffuse Optical Tomography (DOT) reconstruction - CW mode with CG method
% www.neuralimagery.com

function reconCW_CG(depth, separation, square_width, change, nopt)

%% 0. Define some parameters

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

bx = 64; by = 64; % solution basis: grid dimension

% optimization parameters
tau    = 1e-2; % regularisation parameter
beta   = 0.01; % TV regularisation parameter
tolCG  = 1e-8; % CG convergence criterion
itrmax = 100;   % CG max. iterations

%% 1. Define the target mesh
[vtx,idx,eltp] = mkcircle(rad,nsect,nring,nbnd);
hMesh = toastMesh (vtx,idx,eltp);
n = hMesh.NodeCount;
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
qvec = real(hMesh.Qvec('Neumann','Gaussian',2));
mvec = real(hMesh.Mvec('Gaussian',2,refind));

hold on
plot(source(:,1),source(:,2),'ro','MarkerFaceColor','r');
plot(sensor(:,1),sensor(:,2),'bsquare','MarkerFaceColor','b');
hold off
legend('','Source', 'Sensor');

%% 3. Assign optical properties

% Set up the mapper between FEM and inverse solution bases
hBasis = toastBasis (hMesh, [bx by]);
dx = 2*rad/bx; dy = 2*rad/by;

% Set up homogeneous initial parameter estimates
mua = ones(n,1) * mua0; mus = ones(n,1) * mus0; ref = ones(n,1) * refind;
kap = 1./(3*(mua+mus));

swidth    = round(square_width/dx);
s1x_start = round(bx/2 - (separation/2)/dx - swidth/2);
s2x_start = round(bx/2 + (separation/2)/dx - swidth/2);
sy_start  = round(by-depth/dy-swidth/2);

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

%% 4. Generate target data
data = log(mvec' * (dotSysmat(hMesh, mua1, mus1, ref, 0)\qvec));
data = reshape(data, [], 1);

%% 5. Setup inverse solver

% reset parameters to background values
proj = log(mvec' * (dotSysmat(hMesh, mua, mus, ref, 0)\qvec));
proj = reshape(proj, [], 1);
sd = proj; % Assume standard deviation equal to data

% display sinogram of target data
figure(3); clf;
subplot(1,2,1); imagesc(reshape(data-proj,nopt,nopt)); axis equal tight; colorbar
title('initial error');

figure(2);
subplot(2,2,3); hMesh.Display(mua); axis off; title('\mu_a recon');
subplot(2,2,4); hMesh.Display(mus); axis off; title('\mu_s recon');

% map initial estimate of the solution vector
bmua = hBasis.Map ('M->B', mua); bcmua = bmua*cm; scmua = hBasis.Map ('B->S', bcmua);
bkap = hBasis.Map ('M->B', kap); bckap = bkap*cm; sckap = hBasis.Map ('B->S', bckap);

% solution vector
x = [scmua;sckap]; 
% logx = log(x);

% setup regularizer instance
% hReg = toastRegul('TK1', hBasis, x, 1);
% hReg = toastRegul('TK1',hBasis,logx,1);
hReg = toastRegul ('TV', hBasis, x, tau, 'Beta', beta);
% hReg = toastRegul ('Huber', hBasis, logx, tau, 'Eps', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, hReg, x);
err = err0; errp = 1e10;
itr = 1; step = 0.02;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 6. Run the inverse loop solver

while (itr <= itrmax) && (err > tolCG*err0) && (errp-err > tolCG)

    errp = err;
    r = -toastGradient(hMesh, hBasis, qvec, mvec, mua, mus, ref, 0, ...
        data, sd, 'method', 'cg', 'tolerance', 1e-12);
    % r = r(1:p); % drop mus gradient
    r = r .* x; r = r - hReg.Gradient(x);

    if itr > 1
        delta_old = delta_new;
        delta_mid = r' * s;
    end
    s = r;
    if itr == 1
        d = s;
        delta_new = r' * d;
    else
        delta_new = r' * s;
        beta = (delta_new - delta_mid) / delta_old;
        if mod(itr, 20) == 0 || beta <= 0
            d = s;
        else
            d = s + d*beta;
        end
    end
    step = toastLineSearch(x, d, step, err, @objective);
    x = x + d*step;
    
    smua = x(1:size(x)/2)/cm; mua = hBasis.Map ('S->M', smua);
    skap = x(size(x)/2+1:size(x))/cm; 
    smus = 1./(3*skap) - smua; mus = hBasis.Map ('S->M', smus);
    
    figure(2);
    subplot(2,2,3); hMesh.Display(mua); axis off; title('\mu_a recon');
    subplot(2,2,4); hMesh.Display(mus); axis off; title('\mu_s recon');

    proj = reshape(log(mvec' * (dotSysmat(hMesh, mua, mus, ref, 0)\qvec)), [], 1);
    err = toastObjective(proj, data, sd, hReg, x);
    fprintf('Iteration %d, objective %f\n', itr, err);
    itr = itr+1;
    drawnow
end

    function o = objective(x)
        smua_ = x(1:size(x)/2)/cm; mua_ = hBasis.Map ('S->M', smua_);
        skap_ = x(size(x)/2+1:size(x))/cm; 
        smus_ = 1./(3*skap_) - smua_; mus_ = hBasis.Map ('S->M', smus_);
        proj_ = reshape(log(mvec' * (dotSysmat(hMesh, mua_, mus_, ref, 0)\qvec)), [], 1);
        o = toastObjective(proj_, data, sd, hReg, x);
    end

figure(3); subplot(1,2,2); imagesc(reshape(data-proj,nopt,nopt)); axis equal tight; colorbar; title('final error');
path = 'results/circle_two_squares/CG/';
filename = 'recon_'+string(depth)+'_'+string(separation)+'_'+string(square_width)+'_'+string(change)+'_'+string(nopt);
save(path + filename + '.mat');
saveas(gcf, path + filename + '.fig');
% figure; plot(fvals); xlabel('Iteration No.'); ylabel('Objective function values');

end
