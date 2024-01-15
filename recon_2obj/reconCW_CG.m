%% Diffuse Optical Tomography (DOT) reconstruction - CW mode with CG method
% www.neuralimagery.com

function reconCW_CG(depth, separation, square_width, change, nopt)

% Make the save directory if it doesn't exist
dirName = sprintf('results/circle_two_squares_good_depth/cw_depth=%d_separation=%d_square_width=%d_change=%d_nopt=%d', depth, separation, square_width, change, nopt);

if ~exist(dirName, 'dir')
  mkdir(dirName);
end

% optimization parameters
tau    = 1e-2; % regularisation parameter
beta   = 0.01; % TV regularisation parameter
tolCG  = 1e-8; % CG convergence criterion
itrmax = 100;   % CG max. iterations

% Get the medium
[mua, mua1, mus, mus1, kap, ref, qvec, mvec, hMesh, hBasis, cm] = twoSquaresMedium(depth, separation, square_width, change, nopt);

%% ---------

%% 4. Generate target data
data = log(mvec' * (dotSysmat(hMesh, mua1, mus1, ref, 0)\qvec));
data = reshape(data, [], 1);
% m = length(data); % number of measurements

%% Setup inverse solver

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
filename = 'recon_'+string(depth)+'_'+string(separation)+'_'+string(square_width)+'_'+string(change)+'_'+string(nopt);
save(fullfile(dirName, filename + '.mat'));
saveas(gcf, fullfile(dirName + filename + '.fig'));
% figure; plot(fvals); xlabel('Iteration No.'); ylabel('Objective function values');

end
