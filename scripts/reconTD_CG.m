%% Diffuse Optical Tomography (DOT) reconstruction - CW mode with CG method
% www.neuralimagery.com

function reconTD_CG(depth, separation, resol, change, nopt)
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

dt = 20;     % time step in picoseconds
nstep = 256; % number of time steps
ww = 32; % this will lead to 7 bins of width 32 (in units of time-step)
twin = zeros(nstep/ww-1,2); twin(1:nstep/ww-1,1) = [1:ww:(nstep-ww)]';  twin(:,2) = twin(:,1)+ww-1;
twin=[1,256];

%% 1. Define the target mesh
% file = 'meshes/two_squares/two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
file = 'meshes/circle_two_squares/two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
% file = 'two_squares_' + string(depth) + '_' + string(separation) + '_' + string(resol) + '.msh';
% file = 'ReconTD2023/circle32.msh';
mesh = toastMesh (file, 'gmsh');
% ne = mesh.ElementCount;
nv = mesh.NodeCount;
% regidx = mesh.Region;
% regno = unique(regidx);
% abs = find((regidx == regno(2)) | (regidx == regno(3)));
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
ref = ones(nv,1)*refind;
mua = ones(nv,1)*mua_bkg; %mua(abs) = mua_bkg*change;
mus = ones(nv,1)*mus_bkg; %mus(abs) = mus_bkg*change;
kap = ones(nv,1)*kap_bkg;   

% setup inverse solver basis on grid
bx = 64; by = 64;
grd = [bx,by]; basis = toastBasis(mesh, grd);

muaim = zeros(bx,by);musim = muaim;
muaim(18:28,14:24) = mua_bkg*change;
musim(30:40,40:50) = mus_bkg*change;

figure;
subplot(2,2,1); imagesc(reshape(basis.Map('M->B',mua_bkg*ones(nv,1)),bx,by) + muaim);title('\mu_a target'); colorbar;
subplot(2,2,2); imagesc(reshape(basis.Map('M->B',mus_bkg*ones(nv,1)),bx,by) + musim);title('\mu_s target'); colorbar;

%% 4. Generate target data

% compute perturbed target model
mua1 = mua + basis.Map('B->M',muaim);
mus1 = mus + basis.Map('B->M',musim);
[gamma1,t] = toastProjectTPSF(mesh, mua1, mus1, ref, qvec, mvec, dt, nstep);
gamma1 = gamma1.*(1 + 0.02*randn(size(gamma1)));
% data = reshape(WindowTPSF(gamma1,twin)',[],1);
data = reshape(gamma1',[],1);
clear gamma1

m = length(data);

%% 5. Setup inverse solver

% reset parameters to background values
Y = toastProjectTPSF(mesh, mua, mus, ref, qvec, mvec,dt,nstep);
% Y = WindowTPSF(Y,twin);
proj = reshape(Y',[],1); % single vector of data, ordered by time gate.
sd = ones(size(proj));

% visualise data
% nwin = size(twin,1);
% figure(3); clf;
% for w = 1:min(nwin,8);
%     dl = reshape(data((w-1)*nM*nQ+1:w*nM*nQ)-proj((w-1)*nM*nQ+1:(w)*nM*nQ),nM,nQ);
%     subplot(2,4,w);
%     imagesc(dl);
% end

% map initial estimate of the solution vector
% bmua = basis.Map ('M->B', mua); bcmua = bmua*cm; scmua = basis.Map ('B->S', bcmua); p = length(scmua);
% bkap = basis.Map ('M->B', kap); bckap = bkap*cm; sckap = basis.Map ('B->S', bckap);

kap0 = 1./(3*(mua+mus));
x0 = cm*[basis.Map('M->S',mua);basis.Map('M->S',kap0)];
x = x0;
p = length(x);

% solution vector
% x = [scmua;sckap]; %logx = log(x);

% setup regularizer instance
regul = toastRegul('TV', basis, x, tau, 'Beta', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, regul, x);
err = err0; errp = 1e10;
itr = 1; step = 1.0;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 6. Run the inverse loop solver

while (itr <= itrmax) && (err > tolCG*err0) && (errp-err > tolCG)

    errp = err;
    r = -toastGradientTimedomain(mesh, basis, qvec, mvec, mua, mus, ref, dt, nstep, data, sd, 'cg');
    % r = r(1:p); % drop mus gradient
    r = r .* x; r = r - regul.Gradient(x);

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
    
    % smua = exp(logx(1:p))/cm; mua = basis.Map ('S->M', smua);
    smua = x(1:p)/cm; mua = basis.Map ('S->M', smua);
    % skap = exp(logx(p+1:end))/cm; 
    skap = x(p+1:end)/cm;
    smus = 1./(3*skap) - smua; mus = basis.Map ('S->M', smus);

    subplot(2,2,3); imagesc(reshape(basis.Map('M->B',mua),bx,by));title('\mu_a recon'); colorbar;
    subplot(2,2,4); imagesc(reshape(basis.Map('M->B',mus),bx,by));title('\mu_s recon'); colorbar;

    Y = toastProjectTPSF(mesh, mua, mus, ref, qvec, mvec,dt,nstep);
    % Y = WindowTPSF(Y,twin);
    proj = reshape(Y',[],1);
    err = toastObjective(proj, data, sd, regul, x);
    % fvals(itr) = err;
    fprintf('Iteration %d, objective %f\n', itr, err);
    itr = itr+1;
    drawnow
end

    function o = objective(x)
        % smua_ = exp(logx(1:p))/cm; mua_ = basis.Map ('S->M', smua_);
        smua_ = x(1:p)/cm; mua_ = basis.Map ('S->M', smua_);
        % skap_ = exp(logx(p+1:end))/cm; 
        skap_ = x(p+1:end)/cm; 
        smus_ = 1./(3*skap_) - smua_; mus_ = basis.Map ('S->M', smus_);
        % for j = 1:length(mua) % ensure positivity
        %     mua_(j) = max(1e-4,mua_(j));
        %     mus_(j) = max(0.2,mus_(j));
        % end
        Y_ = toastProjectTPSF(mesh, mua_, mus_, ref, qvec, mvec,dt,nstep);
        % Y_ = WindowTPSF(Y,twin);
        proj_ = reshape(Y_',[],1);
        o = toastObjective(proj_, data, sd, regul, x);
    end

path = 'results/TD/circle_two_squares/';
filename = 'recon_'+string(depth)+'_'+string(separation)+'_'+string(resol)+'_'+string(change)+'_'+string(nopt);
save(path + filename + '.mat');
saveas(gcf, path + filename + '.fig');
% figure; plot(fvals); xlabel('Iteration No.'); ylabel('Objective function values');

end
