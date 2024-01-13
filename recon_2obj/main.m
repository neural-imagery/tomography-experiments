close all;

%% Main script to test different recon configurations

mode         = 'TD'; % continuous wave (CW) or time domain (TD) mode
depth        = 20;   % depth of the squares from surface [mm]
separation   = 28;   % distance between squares (of same width) [mm]
square_width = separation/2;    % width of squares [mm]
change       = 2;    % change in optical properties
nopt         = 10;   % number of sources (and detectors)

switch mode
    case 'CW'
        reconCW_CG(depth, separation, square_width, change, nopt);
    case 'TD'
        reconTD_GN(depth, separation, square_width, change, nopt);
end

%% Run all combinations of parameters

% depths = [10,20,30,40]; % depth of the squares from surface [mm]
% separations = [30,20,10]; % distance between squares (of same width) [mm]
% resols = ["10.0", "5.0", "2.5", "1.25"]; % distance between squares (of same width) [mm]
% changes = [1.01,1.1,2]; % change in optical properties
% nopts = [8,16,32]; % number of sources (and detectors)

% for depth = depths
%     for dist = dists
%         for grid_spacing = grid_spacings
%             for change = changes
%                 for nopt = nopts
%                     reconCW_CG(depth, dist, grid_spacing, change, nopt)
%                 end
%             end
%         end
%     end
% end