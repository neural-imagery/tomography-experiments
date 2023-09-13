%% Diffuse Optical Tomography (DOT) reconstruction
% www.neuralimagery.com

function reconstruction
close all

%% 0. Define some parameters
mode = 'cw';
if strcmp(mode, 'td')
    dt = 10;      % step size [ps]
    nstep = 1000; % number of time steps
    theta = 0.5;  % Crank-Nicholson
end

refind = 1.4;   % refractive index
c0 = 0.3;       % speed of light in vacuum [mm/ps]
cm = c0/refind; % speed of light in the medium [mm/ps]
mua_bkg = 0.02; % background absorption [1/mm]
mus_bkg = 0.67;    % background scattering [1/mm];
kap_bkg = 1./(3*(mua_bkg+mus_bkg));
rad = 40;       % mesh radius [mm]

tau = 1e-3;     % regularisation parameter
beta = 0.01;    % TV regularisation parameter
itrmax = 100;   % CG iteration limit
tolCG = 1e-6;   % convergence criterion
tolGN = 1e-5;
tolKrylov = 1e-2;

%% 1. Define the target mesh
mesh = toastMesh('eye_2d.msh','gmsh');
ne = mesh.ElementCount;
nv = mesh.NodeCount;
regidx = mesh.Region;
regno = unique(regidx);
abs = find((regidx == regno(2)) | (regidx == regno(3)));

mesh.Display

%% 2. Define the source and sensor geometry
nopt = 32;
for i=1:nopt
  phiq = (i-1)/nopt*2*pi;
  source(i,:) = rad*[cos(phiq), sin(phiq)];
  phim = (i-0.5)/nopt*2*pi;
  sensor(i,:) = rad*[cos(phim), sin(phim)];
end
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
mua = ones(ne,1)*mua_bkg;
mua(abs) = mua_bkg*2;
mus = ones(ne,1)*mus_bkg;
mus(abs) = mus_bkg*2;
kap = 1./(3*(mua+mus));

figure;
subplot(2,3,1); mesh.Display(mua);
axis off; title('\mu_a target');
subplot(2,3,2); mesh.Display(mus);
axis off; title('\mu_s target');

%% 4. Generate target data

% compute perturbed target model
K = dotSysmat(mesh, mua, mus, ref, 'EL');
switch mode
    case 'cw'
        data = log(mvec' * (K\qvec));
    case 'td'
        M = mesh.Massmat;                 % mass matrix
        K0 = -(K * (1-theta) - M * 1/dt); % backward difference matrix
        K1 = K * theta + M * 1/dt;        % forward difference matrix

        [L U] = lu(K1); % LU factorisation
        
        % initial condition
        q = qvec/dt;
        phi = U\(L\q);
        data = zeros(nstep,nopt,nopt);
        data(1,:,:) = mvec.' * phi;
        
        % loop over time steps
        h = waitbar(0,'loop over time steps');
        for i=2:nstep
            q = K0 * phi;              % new source vector from current field
            phi = U\(L\q);             % new field
            data(i,:,:) = mvec.' * phi; % project to boundary
            waitbar(i/nstep,h);
        end
        close(h);
        data = log(max(data,1e-20));
    otherwise
        warning('Unexpected mode.')
end

% for reference, also solve the homogeneous problem
mua = ones(ne,1)*mua_bkg;
mus = ones(ne,1)*mus_bkg;
K = dotSysmat(mesh, mua, mus, ref, 'EL');
switch mode
    case 'cw'
        data_homog = log(mvec' * (K\qvec));

        % display sinogram of target data perturbations due to inclusion
        subplot(2,3,3); imagesc(data-data_homog); axis equal tight; colorbar
        title('target data');

    case 'td'
        M = mesh.Massmat;                 % mass matrix
        K0 = -(K * (1-theta) - M * 1/dt); % backward difference matrix
        K1 = K * theta + M * 1/dt;        % forward difference matrix

        [L U] = lu(K1); % LU factorisation
        
        % initial condition
        q = qvec/dt;
        phi = U\(L\q);
        data_homog = zeros(nstep,nopt,nopt);
        data_homog(1,:,:) = mvec.' * phi;
        
        % loop over time steps
        h = waitbar(0,'loop over time steps');
        for i=2:nstep
            q = K0 * phi;              % new source vector from current field
            phi = U\(L\q);             % new field
            data_homog(i,:,:) = mvec.' * phi; % project to boundary
            waitbar(i/nstep,h);
        end
        close(h);
        data_homog = log(max(data_homog,1e-20));

        % display sinogram of target data perturbations due to inclusion
        subplot(2,3,3); imagesc(squeeze(sum(data-data_homog,1))); axis equal tight; colorbar
        title('target data');
    otherwise
        warning('Unexpected mode.')
end

%% 5. Setup inverse solver

% reset parameters to background values
mua = ones(nv,1)*mua_bkg;
mus = ones(nv,1)*mus_bkg;
kap = ones(nv,1)*kap_bkg;
ref = ones(nv,1)*refind;
K = dotSysmat(mesh, mua, mus, ref);
switch mode
    case 'cw'
        Y = log(mvec' * (K\qvec));
    case 'td'
        M = mesh.Massmat;                 % mass matrix
        K0 = -(K * (1-theta) - M * 1/dt); % backward difference matrix
        K1 = K * theta + M * 1/dt;        % forward difference matrix

        [L U] = lu(K1); % LU factorisation
        
        % initial condition
        q = qvec/dt;
        phi = U\(L\q);
        Y = zeros(nstep,nopt,nopt);
        Y(1,:,:) = mvec.' * phi;
        
        % loop over time steps
        h = waitbar(0,'loop over time steps');
        for i=2:nstep
            q = K0 * phi;              % new source vector from current field
            phi = U\(L\q);             % new field
            Y(i,:,:) = mvec.' * phi; % project to boundary
            waitbar(i/nstep,h);
        end
        close(h);
        Y = log(max(Y,1e-20));
    otherwise
        warning('Unexpected mode.')
end
proj = reshape(Y, [], 1);
sd = ones(size(proj));
data = reshape(data, [], 1);
subplot(2,3,4); mesh.Display(mua); axis off; title('\mu_a recon');
subplot(2,3,5); mesh.Display(mus); axis off; title('\mu_s recon');
subplot(2,3,6); 
switch mode
    case 'cw'
        imagesc(reshape(proj,nopt,nopt)-data_homog);
    case 'td'
        imagesc(squeeze(sum(reshape(proj,nstep,nopt,nopt)-data_homog,1)));
end
axis equal tight; colorbar
title('recon data');

% setup inverse solver basis on grid
grd = [32,32];
basis = toastBasis(mesh, grd);

% map initial estimate of the solution vector
bmua = basis.Map ('M->B', mua);    % mua mapped to full grid
bmus = basis.Map ('M->B', mus);    % mus mapped to full grid
bkap = basis.Map ('M->B', kap);    % kap mapped to full grid
bcmua = bmua*cm;                   % scale parameters with speed of light
bckap = bkap*cm;                   % scale parameters with speed of light
scmua = basis.Map ('B->S', bcmua); % map to solution basis
sckap = basis.Map ('B->S', bckap); % map to solution basis

x = [scmua;sckap];                 % linea solution vector
logx = log(x);                     % transform to log
p = length(x);                     % solution vector dimension

l = size(scmua,1);
% x = [scmua;scmus];
% x = scmua;
% logx = log(x);
% slen = length(x);

% setup regularizer instance
regul = toastRegul('TV', basis, logx, tau, 'Beta', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, regul, logx);
err = err0;
errp = inf;
itr = 1;
step = 1.0;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 6. Create the inverse loop solver

% inverse solver loop
while (itr <= itrmax) && (err > tolGN*err0) && (errp-err > tolGN)
    errp = err;
    J = toastJacobianCW(mesh,basis,qvec,mvec,mua,mus,ref,'direct',1e-12);
    % data normalisation
    for i = 1:m
        J(i,:) = J(i,:) / sd(i);
    end
    % parameter normalisation (map to log)
    for i = 1:p
        J(:,i) = J(:,i) * x(i);
    end
    % Normalisation of Hessian (map to diagonal 1)
    psiHdiag = hreg.HDiag(logx);
    M = zeros(p,1);
    for i = 1:p
        M(i) = sum(J(:,i) .* J(:,i));
        M(i) = M(i) + psiHdiag(i);
        M(i) = 1 ./ sqrt(M(i));
    end
    for i = 1:p
        J(:,i) = J(:,i) * M(i);
    end
    % Gradient of cost function
    r = J' * ((data-proj)./sd);
    r = r - regul.Gradient (logx) .* M;
    dx = toastKrylov (x, J, r, M, 0, hreg, tolKrylov);

    fprintf (1, 'Entering line search\n');
    step0 = step;
    [step, err] = toastLineSearch (logx, dx, step0, err, @objective, 'verbose', verbosity>0);
    if errp-err <= tolGN
        dx = r; % try steepest descent
        step = toastLineSearch (logx, dx, step0, err, @objective, 'verbose', verbosity>0);
    end
    % Add update to solution
    logx = logx + dx*step;
    x = exp(logx);

    % Map parameters back to mesh
    scmua = x(1:l);
    sckap = x(l+1:end);
    smua = scmua/cm;
    skap = sckap/cm;
    smus = 1./(3*skap) - smua;
    mua = basis.Map ('S->M', smua);
    mus = basis.Map ('S->M', smus);

    subplot(2,3,4); mesh.Display(mua); axis off; title('\mu_a recon');
    subplot(2,3,5); mesh.Display(mus); axis off; title('\mu_s recon');
end
% while (itr <= itrmax) && (err > tolCG*err0) && (errp-err > tolCG)
% 
%     errp = err;
%     r = -toastGradient(mesh, basis, qvec, mvec, mua, mus, ref, 0, ...
%         data, sd, 'method', 'cg', 'tolerance', 1e-12);
%     r = r(1:slen); % drop mus gradient
%     r = r .* x;    % parameter scaling
%     r = r - regul.Gradient(logx);
% 
%     if itr > 1
%         delta_old = delta_new;
%         delta_mid = r' * s;
%     end
%     s = r;
%     if itr == 1
%         d = s;
%         delta_new = r' * d;
%     else
%         delta_new = r' * s;
%         beta = (delta_new - delta_mid) / delta_old;
%         if mod(itr, 20) == 0 || beta <= 0
%             d = s;
%         else
%             d = s + d*beta;
%         end
%     end
%     step = toastLineSearch(logx, d, step, err, @objective);
%     logx = logx + d*step;
%     mua = basis.Map('S->M',exp(logx)/cm);
%     % mua = basis.Map('S->M',exp(logx(1:l))/cm);
%     % mus = basis.Map('S->M',exp(logx(l+1:end))/cm);
%     subplot(2,3,4); mesh.Display(mua); axis off; title('\mu_a recon');
%     % subplot(2,3,5); mesh.Display(mus); axis off; title('\mu_s recon');
% 
%     Ky = dotSysmat(mesh, mua, mus, ref);
%     switch mode
%         case 'cw'
%             Yy = log(mvec' * (Ky\qvec));
%         case 'td'
%             M = mesh.Massmat;                 % mass matrix
%             K0 = -(Ky * (1-theta) - M * 1/dt); % backward difference matrix
%             K1 = Ky * theta + M * 1/dt;        % forward difference matrix
% 
%             [L U] = lu(K1); % LU factorisation
% 
%             % initial condition
%             q = qvec/dt;
%             phi = U\(L\q);
%             Yy = zeros(nstep,nopt,nopt);
%             Yy(1,:,:) = mvec.' * phi;
% 
%             % loop over time steps
%             h = waitbar(0,'loop over time steps');
%             for i=2:nstep
%                 q = K0 * phi;              % new source vector from current field
%                 phi = U\(L\q);             % new field
%                 Yy(i,:,:) = mvec.' * phi; % project to boundary
%                 waitbar(i/nstep,h);
%             end
%             close(h);
%             Yy = log(max(Yy,1e-20));
%         otherwise
%             warning('Unexpected mode.')
%     end
% 
%     proj = reshape(Yy, [], 1);
%     subplot(2,3,6); 
%     switch mode
%         case 'cw'
%             imagesc(reshape(proj,nopt,nopt)-data_homog);
%         case 'td'
%             imagesc(squeeze(sum(reshape(proj,nstep,nopt,nopt)-data_homog,1)));
%     end
%     axis equal tight; colorbar
%     title('recon data');
%     err = toastObjective(proj, data, sd, regul, logx);
%     fprintf('Iteration %d, objective %f\n', itr, err);
%     itr = itr+1;
%     drawnow
% end
    function p = objective(x)

    [mua,mus] = dotXToMuaMus(basis, exp(x), ref);
    proj = toastProject(mesh, mua, mus, ref, 0, qvec, mvec);
    [p, p_data, p_prior] = toastObjective(proj, data, sd, regul, x);
    if verbosity > 0
        fprintf (1, '    [LH: %f, PR: %f]\n', p_data, p_prior);
    end
    end

    % function p = objective(logx)
    % 
    %     mua_ = basis.Map('S->M',exp(logx))/cm;
    %     % mua_ = basis.Map('S->M',exp(logx(1:l)))/cm;
    %     % mus_ = basis.Map('S->M',exp(logx(l+1:end)))/cm;
    % 
    %     Ky_ = dotSysmat(mesh, mua_, mus, ref);
    %     % Ky_ = dotSysmat(mesh, mua_, mus_, ref);
    %     switch mode
    %         case 'cw'
    %             Yy_ = log(mvec' * (Ky_\qvec));
    %         case 'td'
    %             M = mesh.Massmat;                 % mass matrix
    %             K0 = -(Ky_ * (1-theta) - M * 1/dt); % backward difference matrix
    %             K1 = Ky_ * theta + M * 1/dt;        % forward difference matrix
    % 
    %             [L U] = lu(K1); % LU factorisation
    % 
    %             % initial condition
    %             q = qvec/dt;
    %             phi = U\(L\q);
    %             Yy_ = zeros(nstep,nopt,nopt);
    %             Yy_(1,:,:) = mvec.' * phi;
    % 
    %             % loop over time steps
    %             h = waitbar(0,'loop over time steps');
    %             for i=2:nstep
    %                 q = K0 * phi;              % new source vector from current field
    %                 phi = U\(L\q);             % new field
    %                 Yy_(i,:,:) = mvec.' * phi; % project to boundary
    %                 waitbar(i/nstep,h);
    %             end
    %             close(h);
    %             Yy_ = log(max(Yy_,1e-20));
    %         otherwise
    %             warning('Unexpected mode.')
    %     end
    % 
    %     proj_ = reshape(Yy_, [], 1);
    %     p = toastObjective(proj_, data, sd, regul, logx);
    % 
    % end

end
