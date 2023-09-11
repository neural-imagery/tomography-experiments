
function tPhi = InternalFieldsAdjointTimeDomain(K0,K1, asvec, dt,nstep,SolverFlag)
%
% This runs the adjoint problem assuming asvec is time-varying
% Have to decide if asvec is time-reversed or not!
% 
% 
% Author : Simon Arridge 09-Sept-2023

nst = size(asvec,1);
nh = size(asvec,2);
ns = size(asvec,3);
if(nst == nstep)
else
    disp(['Warning in InternalFieldsAdjointTimeDomain. Nstep=',num2str(nstep), ' not equal to adjoint steps ',num2str(nst)]);
end

%tPhi = zeros(nstep,nh,ns);
% initial condition
tPhi = asvec;

% loop over time steps
[L U] = lu(K1);
t = [1:nstep] * dt;

tic;
for i=2:nstep
    q = K0 * squeeze(tPhi(i-1,:,:));
    tPhi(i,:,:) = squeeze(tPhi(i,:,:)) + U\(L\q);
%    phi = max(phi,0);
end
tPhi(1,:,:) = 0;
toc