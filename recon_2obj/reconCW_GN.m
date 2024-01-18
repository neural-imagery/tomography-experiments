function reconCW_GN(depth, separation, square_width, change, nopt)

close all;

disp('MATLAB-TOAST sample script:')
disp('2D image reconstruction with Gauss-Newton solver')
disp('-------------------------------------------------------')

%% 1. Define parameters

% grd = [100 100];  % solution basis: grid dimension
freq = 0;           % modulation frequency [MHz]
% noiselevel = 0;   % additive data noise level
tau = 1e-2;         % regularisation parameter
beta = 0.01;        % TV regularisation parameter
tolGN = 5e-6;       % Gauss-Newton convergence criterion
tolKrylov = 1e-4;   % Krylov convergence criterion
itrmax = 100;        % Gauss-Newton max. iterations
maxit  = 100;       % Krulov solver max. iterations (inside each GN iteration)

%% 1. Define mesh, optodes and optical properties

[mua, mua1, mus, mus1, kap, ref, qvec, mvec, hMesh, hBasis, cm, grd] = twoSquaresMedium(depth, separation, square_width, change, nopt);

%% 2. Generate target data

data = log(mvec' * (dotSysmat(hMesh, mua1, mus1, ref, 0)\qvec));
data = reshape(data, [], 1);
m = length(data);

%% 3. Setup inverse solver

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

% map initial estimate of the solution vector
bmua = hBasis.Map ('M->B', mua); bcmua = bmua*cm; scmua = hBasis.Map ('B->S', bcmua);
% bkap = hBasis.Map ('M->B', kap); bckap = bkap*cm; sckap = hBasis.Map ('B->S', bckap);

% solution vector
% x = [scmua;sckap]; 
x = scmua; %logx = log(x);
p = length(x);

% Initialise regularisation
hReg = toastRegul ('TV', hBasis, x, tau, 'Beta', beta);
% hReg = toastRegul ('TV', hBasis, logx, tau, 'Beta', beta);

% compute initial value of the objective function
err0 = toastObjective(proj, data, sd, hReg, x);
err = err0; errp = 1e10;
itr = 1; step = 0.02;
fprintf('Iteration %d, objective %f\n', 0, err);

%% 4.  Run the inverse loop solver

jtype='bicgstab';
bicgstabtol=1e-12;
while (itr <= itrmax) && (err > tolGN*err0) && (errp-err > tolGN)

    errp = err;
    
    % Construct the Jacobian
    fprintf (1,'Calculating Jacobian\n');
    J = toastJacobianCW (hMesh, hBasis, qvec, mvec, mua, mus, ref, jtype, bicgstabtol);
    J = spdiags(1./sd,0,m,m) * J; % data normalisation
    J = J * spdiags (x,0,p,p);    % parameter normalisation
    
    % Normalisation of Hessian (map to diagonal 1)
    psiHdiag = hReg.HDiag(x);
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
    r = r - hReg.Gradient (x) .* M;

    % Update with implicit Krylov solver
    fprintf (1, 'Entering Krylov solver\n');
    dx = toastKrylov (x, J, r, M, 0, hReg, tolKrylov);

    clear J;
    
    % Line search
    fprintf (1, 'Entering line search\n');
    step0 = step;
    [step, err] = toastLineSearch (x, dx, step0, err, @objective);
    err = real(err);
    if errp-err <= tolGN
        dx = r; % try steepest descent
        step = toastLineSearch (x, dx, step0, err, @objective);
    end

    % Add update to solution
    x = x+step*dx;  
    % logx = logx + dx*step;
    % x = exp(logx);

    % Map parameters back to mesh
    scmua = x; smua = scmua/cm; mua = hBasis.Map ('S->M', smua);
    % sckap = x; skap = sckap/cm;  
    % smus = 1./(3*skap) - smua; mus = hBasis.Map ('S->M', smus);

    % Display reconstructed absorption
    figure(2);
    subplot(2,2,3); 
    hMesh.Display(real(mua)); 
    axis off;
    title('\mu_a recon');
    ax3 = gca; % Get current axes for reconstructed absorption
    % ax3.CLim = colorLimitsMua; % Apply the same color scale

    % Display reconstructed scattering
    subplot(2,2,4); 
    hMesh.Display(real(mus));
    axis off;
    title('\mu_s recon');
    ax4 = gca; % Get current axes for reconstructed scattering
       
    % update projection from current parameter estimate
    % proj = toastProject (hMesh, mua, mus, ref, freq, qvec, mvec);
    proj = reshape(log(mvec' * (dotSysmat(hMesh, mua, mus, ref)\qvec)), [], 1);

    % update objective function
    err = toastObjective (proj, data, sd, hReg, x);
    fprintf (1, '**** GN ITERATION %d, ERROR %f\n\n', itr, err);

    itr = itr+1;
    erri(itr) = err;
    
    % % show objective function
    % figure(4);
    % subplot(1,2,2);
    % semilogy(erri);
    % axis([1 itr 1e-2 2])
    % xlabel('iteration');
    % ylabel('objective function');
    % drawnow
end

disp('recon2: finished')

    % =====================================================================
    % Callback function for objective evaluation (called by toastLineSearch)
    function p = objective(x)

    % Map parameters back to mesh
    % x = exp(logx);
    smua_ = x/cm; mua_ = hBasis.Map('S->M', smua_);
    % skap_ = x/cm;
    % smus_ = 1./(3*skap_) - smua_; mus_ = hBasis.Map ('S->M', smus_);

    proj_ = reshape(log(mvec' * (dotSysmat(hMesh, mua_, mus, ref)\qvec)), [], 1);

    [p, p_data, p_prior] = toastObjective (proj_, data, sd, hReg, x);
    fprintf (1, '    [LH: %f, PR: %f]\n', p_data, p_prior);
    end

end