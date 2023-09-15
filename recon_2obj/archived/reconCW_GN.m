%% Diffuse Optical Tomography (DOT) reconstruction - CW mode with GN method
% www.neuralimagery.com

function reconCW_GN
clear all
close all

%% 0. Define some parameters
refind = 1.4;   % refractive index
c0 = 0.3;       % speed of light in vacuum [mm/ps]
cm = c0/refind; % speed of light in the medium [mm/ps]
mua_bkg = 0.02; % background absorption [1/mm]
mus_bkg = 0.67; % background scattering [1/mm];
kap_bkg = 1./(3*(mua_bkg+mus_bkg)); % diffusion coefficient

rad = 40;  % mesh radius [mm]
nopt = 32; % number of sources (and sensors)

tau = 1e-3;       % regularisation parameter
beta = 0.01;      % TV regularisation parameter
itrmax = 500;     % CG iteration limit
tolGN = 1e-6;     % Gauss-Newton convergence criterion
tolKrylov = 1e-2; % Krylov convergence criterion

%% 1. Define the target mesh
mesh = toastMesh('eye_2d.msh','gmsh');
ne = mesh.ElementCount;
nv = mesh.NodeCount;
regidx = mesh.Region;
regno = unique(regidx);
abs = find((regidx == regno(2)) | (regidx == regno(3)));
% mesh.Display

%% 2. Define the source and sensor geometry
for i=1:nopt
  phiq = (i-1)/nopt*2*pi;
  source(i,:) = rad*[cos(phiq), sin(phiq)];
  phim = (i-0.5)/nopt*2*pi;
  sensor(i,:) = rad*[cos(phim), sin(phim)];
end
mesh.SetQM(source,sensor);
qvec = real(mesh.Qvec('Neumann','Gaussian',2));
mvec = real(mesh.Mvec('Gaussian',2,refind));
nq = size(qvec,2);
nm = size(mvec,2);
nqm = nq*nm;

hold on
plot(source(:,1),source(:,2),'ro','MarkerFaceColor','r');
plot(sensor(:,1),sensor(:,2),'bsquare','MarkerFaceColor','b');
hold off
legend('','Source', 'Sensor');

%% 3. Assign optical properties
ref = ones(ne,1)*refind;
mua = ones(ne,1)*mua_bkg; mua(abs) = mua_bkg*2;
mus = ones(ne,1)*mus_bkg; mus(abs) = mus_bkg*2;

nmom = 2; % highest moment to compute
grd = [32,32]; basis = toastBasis(mesh, grd);
jac = toastJacobianMoments(mesh,basis,nmom,qvec,mvec,mua,mus,ref,'DIRECT');

figure;
subplot(1,3,1);
imagesc(reshape(toastMapSolToGrid(basis,jac(8,:)),grd));axis equal tight
subplot(1,3,2);
imagesc(reshape(toastMapSolToGrid(basis,jac(8+nqm,:)),grd));axis equal tight
subplot(1,3,3);
imagesc(reshape(toastMapSolToGrid(basis,jac(8+nqm*2,:)),grd));axis equal tight

figure;
for mom=0:nmom
    mjac = jac(mom*nqm+1:(mom+1)*nqm,:);
    jmin = min(min(mjac));
    jmax = max(max(mjac));
    for i=1:nqm
        imagesc(reshape(toastMapSolToGrid(hbasis,mjac(i,:)),grd)); axis equal tight;colorbar
        title(['moment=' num2str(mom) ', meas=' num2str(i)])
        drawnow
        pause(0.1);
    end
end

% figure;
% subplot(2,3,1); mesh.Display(mua);
% axis off; title('\mu_a target');
% subplot(2,3,2); mesh.Display(mus);
% axis off; title('\mu_s target');
% 
% %% 4. Generate target data
% 
% % compute perturbed target model
% data = log(mvec' * (dotSysmat(mesh, mua, mus, ref, 'EL')\qvec));
% 
% % display sinogram of target data
% subplot(2,3,3); imagesc(data); axis equal tight; colorbar
% title('target data');
% 
% %% 5. Setup inverse solver
% 
% % reset parameters to background values
% ref = ones(nv,1)*refind;
% mua = ones(nv,1)*mua_bkg; mus = ones(nv,1)*mus_bkg; kap = ones(nv,1)*kap_bkg;
% Y = log(mvec' * (dotSysmat(mesh, mua, mus, ref)\qvec));
% 
% proj = reshape(Y, [], 1); sd = ones(size(proj)); data = reshape(data, [], 1);
% subplot(2,3,4); mesh.Display(mua); axis off; title('\mu_a recon');
% subplot(2,3,5); mesh.Display(mus); axis off; title('\mu_s recon');
% subplot(2,3,6); imagesc(reshape(proj,nopt,nopt)); axis equal tight; colorbar; title('recon data');
% 
% % setup inverse solver basis on grid
% grd = [32,32]; basis = toastBasis(mesh, grd);
% 
% % map initial estimate of the solution vector
% bmua = basis.Map ('M->B', mua); bcmua = bmua*cm; scmua = basis.Map ('B->S', bcmua); p = length(scmua);
% bkap = basis.Map ('M->B', kap); bckap = bkap*cm; sckap = basis.Map ('B->S', bckap);
% 
% % solution vector
% x = [scmua;sckap]; logx = log(x);
% 
% % setup regularizer instance
% regul = toastRegul('TV', basis, logx, tau, 'Beta', beta);
% 
% % compute initial value of the objective function
% err0 = toastObjective(proj, data, sd, regul, logx);
% err = err0; errp = inf;
% itr = 1; step = 1.0;
% fprintf('Iteration %d, objective %f\n', 0, err);
% 
% %% 6. Run the inverse loop solver
% 
% while (itr <= itrmax) && (err > tolGN*err0) && (errp-err > tolGN)
% 
%     errp = err;
%     J = toastJacobianCW(mesh, basis, qvec, mvec, mua, mus, ref, 'direct');
%     for i = 1:size(J,2)
%         M(i) = 1 ./ sqrt(sum(J(:,i) .* J(:,i)));
%     end
%     for i = 1:size(M,2)
%         J(:,i) = J(:,i) * M(1,i);
%     end
% 
%     % Gradient of cost function
%     r = J' * (data-proj);
%     dx = krylov(r);
%     [step, err] = toastLineSearch(logx, dx, step, err, @objective2);
% 
%     logx = logx + dx*step; x = exp(logx);
% 
%     smua = x(1:p)/cm; mua = basis.Map ('S->M', smua);
%     skap = x(p+1:end)/cm; 
%     smus = 1./(3*skap) - smua; mus = basis.Map ('S->M', smus);
%     subplot(2,3,4); mesh.Display(mua); axis off; title('\mu_a recon');
%     subplot(2,3,5); mesh.Display(mus); axis off; title('\mu_s recon');
% 
%     proj = reshape(log(mvec' * (dotSysmat(mesh, mua, mus, ref)\qvec)), [], 1);
%     subplot(2,3,6); imagesc(reshape(proj,nopt,nopt)); axis equal tight; colorbar; title('recon data');
%     err = objective(proj);
%     fprintf('Iteration %d, objective %f\n', itr, err);
%     itr = itr+1;
%     drawnow
% end
% 
%     % =====================================================================
%     % returns objective function, given model boundary data proj
% 
%     function o = objective(proj)
%     o = sum((data-proj).^2);
%     end
% 
% 
%     % =====================================================================
%     % returns objective function, given model parameters (used as callback
%     % function by toastLineSearch)
% 
%     function o = objective2(logx)
%         smua_ = exp(logx(1:p))/cm; mua_ = basis.Map ('S->M', smua_);
%         skap_ = exp(logx(p+1:end))/cm; 
%         smus_ = 1./(3*skap_) - smua_; mus_ = basis.Map ('S->M', smus_);
%         proj_ = reshape(log(mvec' * (dotSysmat(mesh, mua_, mus_, ref)\qvec)), [], 1);
%         o = objective(proj_);
%     end
% 
% 
%     % =====================================================================
%     % Krylov solver subroutine
% 
%     function dx = krylov(r)
%     dx = gmres (@jtjx, r, 30, tolKrylov, 100);
%     end
% 
% 
%     % =====================================================================
%     % Callback function for matrix-vector product (called by krylov)
% 
%     function b = jtjx(x)
%     b = J' * (J*x);
%     end
% 
end
