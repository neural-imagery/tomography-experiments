%% Diffuse Optical Tomography (DOT) reconstruction - CW mode with CG method
% www.neuralimagery.com

function reconCW_CG(depth, separation, resol, change, nopt)
% clear all
close all

%% 0. Define some parameters
refind = 1.4;   % refractive index
c0 = 0.3;       % speed of light in vacuum [mm/ps]
cm = c0/refind; % speed of light in the medium [mm/ps]
mua_bkg = 0.02; % background absorption [1/mm]
mus_bkg = 0.67; % background scattering [1/mm];
kap_bkg = 1./(3*(mua_bkg+mus_bkg)); % diffusion coefficient

rad = 40;  % mesh radius [mm]

tau = 1e-3;   % regularisation parameter
beta = 0.03;  % TV regularisation parameter
itrmax = 500; % CG iteration limit
tolCG = 1e-8; % convergence criterion

%% 1. Define the target mesh
% file = 'meshes/two_squares/two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
% file = 'meshes/circle_two_squares/two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
file = 'two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
mesh = toastMesh(file,'gmsh');
ne = mesh.ElementCount;
nv = mesh.NodeCount;
regidx = mesh.Region;
regno = unique(regidx);
abs = find((regidx == regno(2)) | (regidx == regno(3)));
mesh.Display

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

% mesh_height = 80;
% mesh_width = 140;
% shift = mesh_width/(2*nopt);
% source(:,1) = linspace(0,mesh_width-shift,nopt);
% source(:,2) = mesh_height;
% 
% sensor(:,1) = linspace(shift,mesh_width,nopt);
% sensor(:,2) = mesh_height;

mesh.SetQM(source,sensor);
qvec = real(mesh.Qvec('Neumann','Gaussian',2));
mvec = real(mesh.Mvec('Gaussian',2,refind));

hold on
plot(source(:,1),source(:,2),'ro','MarkerFaceColor','r');
plot(sensor(:,1),sensor(:,2),'bsquare','MarkerFaceColor','b');
hold off
legend('','Source', 'Sensor');

%% 3. Assign optical properties
ref = ones(ne,1)*refind;
mua = ones(ne,1)*mua_bkg; mua(abs) = mua_bkg*change;
mus = ones(ne,1)*mus_bkg; mus(abs) = mus_bkg*change;

figure;
subplot(2,3,1); mesh.Display(mua);
axis off; title('\mu_a target');
subplot(2,3,2); mesh.Display(mus);
axis off; title('\mu_s target');

%% 4. Generate target data

% compute perturbed target model
data = log(mvec' * (dotSysmat(mesh, mua, mus, ref, 'EL')\qvec));

% display sinogram of target data
subplot(2,3,3); imagesc(data); axis equal tight; colorbar
title('target data');

%% 5. Setup inverse solver

% reset parameters to background values
ref = ones(nv,1)*refind;
mua = ones(nv,1)*mua_bkg; mus = ones(nv,1)*mus_bkg; kap = ones(nv,1)*kap_bkg;
Y = log(mvec' * (dotSysmat(mesh, mua, mus, ref)\qvec));
    
proj = reshape(Y, [], 1); data = reshape(data, [], 1); sd = ones(size(proj));
subplot(2,3,4); mesh.Display(mua); axis off; title('\mu_a recon');
subplot(2,3,5); mesh.Display(mus); axis off; title('\mu_s recon');
subplot(2,3,6); imagesc(reshape(proj,nopt,nopt)); axis equal tight; colorbar; title('recon data');

% setup inverse solver basis on grid
grd = [32,32]; basis = toastBasis(mesh, grd);

% map initial estimate of the solution vector
bmua = basis.Map ('M->B', mua); bcmua = bmua*cm; scmua = basis.Map ('B->S', bcmua); p = length(scmua);
bkap = basis.Map ('M->B', kap); bckap = bkap*cm; sckap = basis.Map ('B->S', bckap);

% solution vector
x = [scmua;sckap]; logx = log(x);

% setup regularizer instance
regul = toastRegul('TV', basis, logx, tau, 'Beta', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, regul, logx);
err = err0; errp = inf;
itr = 1; step = 1.0;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 6. Run the inverse loop solver

while (itr <= itrmax) && (err > tolCG*err0) && (errp-err > tolCG)

    errp = err;
    r = -toastGradient(mesh, basis, qvec, mvec, mua, mus, ref, 0, ...
        data, sd, 'method', 'cg', 'tolerance', 1e-12);
    % r = r(1:p); % drop mus gradient
    r = r .* x; r = r - regul.Gradient(logx);

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
    step = toastLineSearch(logx, d, step, err, @objective);
    logx = logx + d*step;
    
    smua = exp(logx(1:p))/cm; mua = basis.Map ('S->M', smua);
    skap = exp(logx(p+1:end))/cm; 
    smus = 1./(3*skap) - smua; mus = basis.Map ('S->M', smus);
    subplot(2,3,4); mesh.Display(real(mua)); axis off; title('\mu_a recon');
    subplot(2,3,5); mesh.Display(real(mus)); axis off; title('\mu_s recon');

    proj = reshape(log(mvec' * (dotSysmat(mesh, mua, mus, ref)\qvec)), [], 1);
    subplot(2,3,6); imagesc(reshape(proj,nopt,nopt)); axis equal tight; colorbar; title('recon data');
    err = toastObjective(proj, data, sd, regul, logx);
    % fvals(itr) = err;
    fprintf('Iteration %d, objective %f\n', itr, err);
    itr = itr+1;
    drawnow
end

    function o = objective(logx)
        smua_ = exp(logx(1:p))/cm; mua_ = basis.Map ('S->M', smua_);
        skap_ = exp(logx(p+1:end))/cm; 
        smus_ = 1./(3*skap_) - smua_; mus_ = basis.Map ('S->M', smus_);
        proj_ = reshape(log(mvec' * (dotSysmat(mesh, mua_, mus_, ref)\qvec)), [], 1);
        o = toastObjective(proj_, data, sd, regul, logx);
    end

path = 'results/circle_two_squares/';
filename = 'recon_'+string(depth)+'_'+string(separation)+'_'+string(resol)+'_'+string(change)+'_'+string(nopt);
save(path + filename + '.mat');
saveas(gcf, path + filename + '.fig');
% figure; plot(fvals); xlabel('Iteration No.'); ylabel('Objective function values');

end
