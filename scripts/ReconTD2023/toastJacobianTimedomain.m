function J = toastJacobianTimedomain(hMesh,hBasis,qvec,mvec, mua, mus, ref,dt,nstep,twin,SolverFlag)
%
% generate complete set of TPSFs, each in interval of dt for nstep steps
%
% use a fully implicit scheme for first test
%
% 
% Author : Simon Arridge 09-Sept-2023
theta = 1;
[smat,bmat] = dotSysmat (hMesh, mua, mus, ref, 0);
mmat = Massmat(hMesh); % the bmat is only the boundary part!

K0 = -(smat * (1-theta) - mmat * 1/dt);            % backward difference matrix
K1 = smat * theta + mmat * 1/dt;                   % forward difference matrix
tphi  = InternalFieldsTimeDomain(K0,K1, qvec, dt,nstep,SolverFlag);
taphi = InternalFieldsTimeDomain(K0,K1, mvec, dt,nstep,SolverFlag);
solmask = hBasis.Map('S->B',ones(hBasis.slen,1)); %solmask = toastSolutionMask(hBasis);
nsol = hBasis.slen;
nQ = size(qvec,2);
nM = size(mvec,2);
nwin=size(twin,1);
J = zeros(nQ*nM,2*nsol);  % absorption and diffusion
rhoaf = zeros(nstep,nsol);
rhosf = rhoaf;
 
jind = 0;
for q = 1:nQ
    disp(['building Jacobian for source ',num2str(q)]);
    fpq  = fft(tphi(:,:,q),[],1); % Fourier Transform over time variable
    for m = 1:nM
        jind  = jind+1;
        fpam  = fft(taphi(:,:,m),[],1); % Fourier Transform over time variable
        for k = 1:size(fpam,1)
            rhoaf(k,:) = hBasis.Map('M->S',fpq(k,:).*fpam(k,:));
            dim = hBasis.Map('M->B',fpq(k,:));
            aim = hBasis.Map('M->B',fpam(k,:));
            gd = hBasis.ImageGradient(dim);
            ga = hBasis.ImageGradient(aim);
            gr = (gd(1,:).*ga(1,:) + gd(2,:).*ga(2,:));
            rhosf(k,:) = gr(find(solmask==1));             
        end
        rhoat = ifft(rhoaf,[],1); % time domain PMDF for absorption
        rhoaw=WindowTPSF(rhoat,twin);
        rhost = ifft(rhosf,[],1); % time domain PMDF for diffusion
        rhosw=WindowTPSF(rhost,twin);
        for w = 1:nwin
            J((w-1)*nQ*nM + jind,1:nsol)        = rhoaw(w,:);%toastMapMeshToSol(hBasis,rhoaw(w,:));
            J((w-1)*nQ*nM + jind,nsol+1:2*nsol) = rhosw(w,:);%toastMapMeshToSol(hBasis,rhosw(w,:));
        end
    end
end

