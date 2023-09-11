function [gamma,t] = toastProjectTPSF(hMesh, mua, mus, ref, qvec, mvec,dt,nstep)
%
% generate complete set of TPSFs, each in interval of dt for nstep steps
%
% use a fully implicit scheme for first test
%
% 
% Author : Simon Arridge 09-Sept-2023
%
theta = 1;
[smat,bmat] = dotSysmat (hMesh, mua, mus, ref, 0);
mmat = Massmat(hMesh); % the bmat is only the boundary part!

K0 = -(smat * (1-theta) - mmat * 1/dt);            % backward difference matrix
K1 = smat * theta + mmat * 1/dt;                   % forward difference matrix
mvecT = mvec.';
nq = size(qvec,2);
nm = size(mvec,2);

% initial condition
phi = qvec;

% loop over time steps
[L U] = lu(K1);
t = [1:nstep] * dt;

gamma = zeros(nstep,nq*nm);
tic;
for i=1:nstep
    q = K0 * phi;
    phi = U\(L\q);
    gamma(i,:) = reshape(mvecT * phi,[],1);

end
toc