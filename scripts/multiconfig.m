
depths = [10,20,30,40];          % depth of the squares from surface [mm]
separations = [30,20,10];   % distance between squares (of same width) [mm]
resols = ["10.0", "5.0", "2.5", "1.25"]; % distance between squares (of same width) [mm]
changes = [1.01,1.1,2];          % change in optical properties
nopts = [8,16,32];               % number of sources (and detectors)

reconTD_GN(depths(1), separations(1), resols(2), changes(end), nopts(end))

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