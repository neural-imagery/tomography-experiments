function [mua_bg, mua1, mus_bg, mus1, kap_bg, ref, qvec, mvec, hMesh, hBasis, cm, grd] = twoSquaresMedium(depth, separation, square_width, change, nopt)
% twoSquaresMedium
%
% This function creates a two-dimensional simulation environment for 
% optical tomography using the TOAST++ toolkit. It defines a circular mesh 
% with specified parameters and sets up a problem with two square anomalies 
% in optical properties. The function outputs various parameters and objects 
% necessary for further simulation and inverse problem solving in optical tomography.
% 
% Parameters:
%   depth           - Depth of the anomalies [mm]
%   separation      - Separation between the anomalies [mm]
%   square_width    - Width of the square anomalies [mm]
%   change          - Relative change in optical properties of the anomalies
%   nopt            - Number of sources and detectors
%
% Outputs:
%   mua_bg  - Background absorption coefficient matrix
%   mua1    - Target absorption coefficient matrix with anomalies
%   mus_bg  - Background scattering coefficient matrix
%   mus1    - Target scattering coefficient matrix with anomalies
%   kap_bg  - Diffusion coefficient matrix for the background
%   ref     - Refractive index matrix
%   qvec    - Source vectors for the quantum dots
%   mvec    - Measurement vectors
%   hMesh   - Handle to the mesh object
%   cm      - Speed of light in the medium
%
% The function sets up the mesh, defines the source and detector locations,
% assigns optical properties to the mesh, and creates the basis for the 
% finite element method (FEM) and inverse solution. It is configured for 
% a specific example and can be modified for different geometries or optical 
% properties.
%
% Example usage:
%   [mua_bg, mua1, mus_bg, mus1, kap_bg, ref, qvec, mvec, hMesh, cm] = twoSquaresMedium();  

figure;

% mesh parameters
rad   = 70; % mesh radius [mm]
nsect = 6;  % number of sectors
nring = 32; % number of rings
nbnd  = 2;  % number of boundary rings

% optical parameters @ 850nm
refind = 1.4;                % refractive index
c0     = 0.3;                % speed of light in vacuum [mm/ps]
cm     = c0/refind;          % speed of light in the medium [mm/ps]
mua0   = 0.02;               % background absorption [1/mm]
mus0   = 0.67;               % background scattering [1/mm];

bx = 64; by = 64; % solution basis: grid dimension
grd = [bx by];    % solution basis: grid size

%% 1. Define the target mesh
[vtx,idx,eltp] = mkcircle(rad,nsect,nring,nbnd);
hMesh = toastMesh (vtx,idx,eltp);
n = hMesh.NodeCount;
hMesh.Display

%% 2. Define the source and sensor geometry
distrib = 'quarter';
for i=1:nopt
  switch distrib
      case 'full'
          phiq = (i-1)/nopt*2*pi; 
          phim = (i-0.5)/nopt*2*pi;
      case 'quarter'
          phiq = (i-1)/nopt*pi/2+pi/4;
          phim = (i-0.5)/nopt*pi/2+pi/4;
  end
  source(i,:) = rad*[cos(phiq), sin(phiq)];
  sensor(i,:) = rad*[cos(phim), sin(phim)];
end

hMesh.SetQM(source,sensor);
qvec = real(hMesh.Qvec('Neumann','Gaussian',2));
mvec = real(hMesh.Mvec('Gaussian',2,refind));

hold on
plot(source(:,1),source(:,2),'ro','MarkerFaceColor','r');
plot(sensor(:,1),sensor(:,2),'bsquare','MarkerFaceColor','b');
hold off
legend('','Source', 'Sensor');

%% 3. Assign optical properties

% Set up the mapper between FEM and inverse solution bases
hBasis = toastBasis (hMesh, [bx by]);
dx = 2*rad/bx; dy = 2*rad/by;

% Set up homogeneous initial parameter estimates
mua_bg = ones(n,1) * mua0; mus_bg = ones(n,1) * mus0; ref = ones(n,1) * refind;
kap_bg = 1./(3*(mua_bg+mus_bg));

swidth    = round(square_width/dx) - 1;
s1x_start = round((bx/2 + 1) - (separation/2)/dx - swidth/2);
s2x_start = round((bx/2 + 1) + (separation/2)/dx - swidth/2);
sy_start  = round(by-depth/dy - swidth);

muaim = zeros(bx,by); musim = muaim;
muaim(s1x_start:s1x_start+swidth, sy_start:sy_start+swidth) = change*mua0;
muaim(s2x_start:s2x_start+swidth, sy_start:sy_start+swidth) = change*mua0;
musim(s1x_start:s1x_start+swidth, sy_start:sy_start+swidth) = change*mus0;
musim(s2x_start:s2x_start+swidth, sy_start:sy_start+swidth) = change*mus0;

mua1 = mua_bg + hBasis.Map('B->M',muaim);
mus1 = mus_bg + hBasis.Map('B->M',musim);

figure(2); clf;
subplot(2,2,1); hMesh.Display(mua1); axis off; title('\mu_a target');
subplot(2,2,2); hMesh.Display(mus1); axis off; title('\mu_s target');

end