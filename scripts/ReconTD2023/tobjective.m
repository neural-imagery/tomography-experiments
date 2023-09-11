% 
% Author : Simon Arridge 09-Sept-2023
% =====================================================================
   % Callback function for objective evaluation (called by toastLineSearch)

    function p = tobjective(x)

    global lprm; % a set of local parameters
    
    % Map parameters back to mesh
    smua = x(1:size(x)/2)/lprm.cm;
    skap = x(size(x)/2+1:size(x))/lprm.cm;
    smus = 1./(3*skap) - smua;
    mua = lprm.hBasis.Map('S->M', smua);
    mus = lprm.hBasis.Map('S->M', smus);
    for j = 1:length(mua) % ensure positivity
        mua(j) = max(1e-4,mua(j));
        mus(j) = max(0.2,mus(j));
    end
   
    [gamma,t] = toastProjectTPSF(lprm.hMesh, mua, mus, lprm.ref, lprm.qvec, lprm.mvec,lprm.dt,lprm.nstep);
     proj = reshape(WindowTPSF(gamma,lprm.twin)',[],1); % single vector of data, ordered by time gate.

    [p, p_data, p_prior] = toastObjective (proj, lprm.data, lprm.sd,lprm.hReg, x);
    fprintf (1, '    [LH: %f, PR: %f]\n', p_data, p_prior);
    end
