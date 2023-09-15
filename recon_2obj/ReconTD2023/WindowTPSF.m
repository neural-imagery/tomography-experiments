function y = WindowTPSF(TPSF,twin)
%
% take a set of TPSFs and window them
%
% 
% Author : Simon Arridge 09-Sept-2023
%
nt = size(TPSF,1);
ndat = size(TPSF,2);
nwin = size(twin,1);
y = zeros(nwin,ndat);
for w = 1:nwin
    y(w,:) = sum(TPSF(twin(w,1):twin(w,2),:));
end