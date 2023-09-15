function G = toastGradientTimedomain(hMesh,hBasis,qvec,mvec, mua, mus, ref,dt,nstep,ddata,sd,SolverFlag)
%
% generate complete set of TPSFs, each in interval of dt for nstep steps
% use a fully implicit scheme for first test
% ddata should be difference of TPSFS on the boundary
%   ddata should be nt X (nq.nm)
%
% 
% Author : Simon Arridge 09-Sept-2023
%
nh = size(qvec,1);
nQ = size(qvec,2);
nM = size(mvec,2);
dgam = reshape(ddata,nstep,nQ,nM); % careful with the ordering
sgam = reshape(sd,nstep,nQ,nM); % careful with the ordering
rgam = dgam./(sgam.^2);  % this is the rescaled data for back projection.
theta = 1;
[smat,bmat] = dotSysmat (hMesh, mua, mus, ref, 0);
mmat = Massmat(hMesh); % the bmat is only the boundary part!

K0 = -(smat * (1-theta) - mmat * 1/dt);            % backward difference matrix
K1 = smat * theta + mmat * 1/dt;                   % forward difference matrix
tphi  = InternalFieldsTimeDomain(K0,K1, qvec, dt,nstep,SolverFlag);
asrc = zeros(nstep,nh,nQ);
for q = 1:nQ
    asrc(:,:,q) = (squeeze(rgam(:,q,:))*mvec' ) ; % may need to divide data by sd...
end
taphi = InternalFieldsAdjointTimeDomain(K0,K1, asrc, dt,nstep,SolverFlag);
solmask = hBasis.Map('S->B',ones(hBasis.slen,1)); %solmask = toastSolutionMask(hBasis);
nsol = hBasis.slen;

rhoaf = zeros(nstep,nsol);
rhosf = rhoaf;
G = zeros(2*nsol,1);
for q = 1:nQ
    disp(['building Gradient for source ',num2str(q)]);
    fpq  = fft(tphi(:,:,q),[],1); % Fourier Transform over time variable
    
    fpam  = fft(taphi(:,:,q),[],1); % Fourier Transform over time variable
    for k = 1:size(fpam,1)
        %            rhoaf(k,:) = toastIntFG(hMesh,fpq(k,:),fpam(k,:));
        %            rhosf(k,:) = toastIntGradFGradG(hMesh,fpq(k,:),fpam(k,:));
        rhoaf(k,:) = hBasis.Map('M->S',fpq(k,:).*fpam(k,:));
        dim = hBasis.Map('M->B',fpq(k,:));
        aim = hBasis.Map('M->B',fpam(k,:));
        gd = hBasis.ImageGradient(dim);
        ga = hBasis.ImageGradient(aim);
        gr = (gd(1,:).*ga(1,:) + gd(2,:).*ga(2,:));
        rhosf(k,:) = gr(find(solmask==1));
    end
    rhoat = ifft(rhoaf,[],1); % time domain PMDF for absorption
    rhost = ifft(rhosf,[],1); % time domain PMDF for diffusion
    % sum over time. Sure there is a Fourier Domain version of this!
    G(1:nsol) = G(1:nsol) + sum(rhoat,1)';
    G(nsol+1:2*nsol) = G(nsol+1:2*nsol) + sum(rhost,1)';
end

