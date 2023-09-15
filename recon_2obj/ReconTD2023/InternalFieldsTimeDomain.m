
function tPhi = InternalFieldsTimeDomain(K0,K1, svec, dt,nstep,SolverFlag)
%
% This will run into trouble with large problems
% generate complete set of TPSFs, each in interval of dt for nstep steps
%
% 
% 
% Author : Simon Arridge 09-Sept-2023

nh = size(svec,1);
ns = size(svec,2);

tPhi = zeros(nstep,nh,ns);
% initial condition
tPhi(1,:,:) = svec;

% loop over time steps
[L U] = lu(K1);
t = [1:nstep] * dt;

tic;
for i=2:nstep
    q = K0 * squeeze(tPhi(i-1,:,:));
    tPhi(i,:,:) = U\(L\q);
%    phi = max(phi,0);
end
tPhi(1,:,:) = 0;
toc