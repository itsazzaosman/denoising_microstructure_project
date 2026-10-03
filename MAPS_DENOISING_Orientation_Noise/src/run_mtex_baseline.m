% =========================================================================
% run_mtex_baseline.m
%
% Runs the standard MTEX denoising filters on the noisy .ang maps and
% exports each result as a plain Euler-angle .txt, in the SAME pixel order
% as the original DREAM.3D export, so the Python disorientation scorer can
% compare everything on equal terms.
%
% This is the baseline your model has to beat. Run it FIRST: if the median
% filter already fixes 5% misindexing perfectly, you have learned that in an
% afternoon and should move straight to higher corruption rates.
%
% Output, per input map and per filter:
%     <outdir>/<stem>__<filtername>.txt
%
% Set the paths below and run.
% =========================================================================

clear; close all;

% ----------------------------- SETTINGS ---------------------------------
angDir  = 'C:\Users\STUDENT\OneDrive\Documenten\MATLAB\mtex-7.1.0\ang_files_scatter9';
% outDir  = 'C:\Users\STUDENT\OneDrive\Documenten\MATLAB\mtex-7.1.0\mtex_out_scatter9';
outDir  = 'C:\Users\STUDENT\OneDrive\Documenten\MATLAB\mtex-7.1.0\mtex_out_scatter9_nograins';
nMaps   = 10;        % how many maps to benchmark; 50 is plenty
NY = 128; NX = 128;  % map size
step    = 1.0;       % micron per pixel, must match --step used when writing .ang
minPx   = 3;         % variant B: delete grains smaller than this many pixels.
                     % 3 is a common choice; try 2 and 5 to see how sensitive
                     % the baseline is to it.
% ------------------------------------------------------------------------

CS = crystalSymmetry('m-3m', [3.524 3.524 3.524], 'mineral', 'Nickel');
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','intoPlane');

if ~exist(outDir, 'dir'); mkdir(outDir); end

% the filters to benchmark. Add or remove freely.
filters = {};
F = meanFilter;                                  filters{end+1} = {'mean', F};
F = medianFilter;      F.numNeighbours = 2;      filters{end+1} = {'median', F};
F = KuwaharaFilter;    F.numNeighbours = 2;      filters{end+1} = {'kuwahara', F};
F = splineFilter;                                filters{end+1} = {'spline', F};
F = halfQuadraticFilter; F.alpha = 0.25;         filters{end+1} = {'halfquad', F};
F = infimalConvolutionFilter;                    filters{end+1} = {'infconv', F};

files = dir(fullfile(angDir, '*_noisy_noisy.ang'));
if isempty(files)
    error('No *_noisy_noisy.ang files found in %s', angDir);
end
nMaps = min(nMaps, numel(files));
fprintf('%d .ang files found, benchmarking %d\n\n', numel(files), nMaps);

for k = 1:nMaps
    fname = fullfile(files(k).folder, files(k).name);
    [~, stem, ~] = fileparts(files(k).name);
    fprintf('[%d/%d] %s\n', k, nMaps, stem);

    % IMPORTANT: 'setting', 0 switches off the coordinate correction MTEX
    % applies by default. .ang files from real machines carry a convention
    % that isn't stored in the file, so MTEX guesses "setting 2". These files
    % are synthetic and carry no such convention, so any correction would
    % rotate the data and corrupt every later measurement.
    % Verified on one map: with 'setting', 0 the round trip is exact (0.0000 deg).
    ebsd = EBSD.load(fname, CS, 'interface', 'ang', 'setting', 0);

    % grain reconstruction, needed by 'fill' so boundary pixels are respected
    [grains, ebsd.grainId] = calcGrains(ebsd('indexed'), 'angle', 10*degree);

    % ------------------------------------------------------------------
    % VARIANT A: smoothing only.
    % Filters denoise WITHIN grains. A misindexed pixel is ~40 deg from its
    % neighbours, so calcGrains makes it a single-pixel grain and the filter
    % has nothing to average it with. Expect this to fix scatter and leave
    % outliers untouched.
    % ------------------------------------------------------------------
    for f = 1:numel(filters)
        fname_f = filters{f}{1};
        F       = filters{f}{2};
        outFile = fullfile(outDir, sprintf('%s__%s.txt', stem, fname_f));
        if exist(outFile, 'file'); continue; end
        try
            % ebsdS = smooth(ebsd('indexed'), F, 'fill', grains);
            ebsdS = smooth(ebsd('indexed'), F);
            writeEulerGrid(outFile, ebsdS, NY, NX, step);
        catch ME
            fprintf('   [skip] %s: %s\n', fname_f, ME.message);
        end
    end

    % ------------------------------------------------------------------
    % VARIANT B: the standard practitioner workflow.
    %   1. delete grains smaller than minPx pixels -> removes the outliers
    %   2. re-segment what is left
    %   3. smooth with 'fill' -> interpolates the deleted pixels back
    % This is the step that actually targets misindexing, and it is the
    % baseline your model has to beat. Results are suffixed "clean+<filter>".
    % ------------------------------------------------------------------
    % for f = 1:numel(filters)
        fname_f = filters{f}{1};
        F       = filters{f}{2};
        outFile = fullfile(outDir, sprintf('%s__clean-%s.txt', stem, fname_f));
        if exist(outFile, 'file'); continue; end
        try
            ebsdC = ebsd;
            % MTEX 7 uses numPixel; older versions used grainSize
            if isprop(grains, 'numPixel')
                small = grains(grains.numPixel < minPx);
            else
                small = grains(grains.grainSize < minPx);
            end
            ebsdC(small) = [];                      % drop those pixels
            % [grainsC, ebsdC.grainId] = calcGrains(ebsdC('indexed'), 'angle', 10*degree);
            % ebsdS = smooth(ebsdC('indexed'), F, 'fill', grainsC);
            ebsdS = smooth(ebsdC('indexed'), F);
            writeEulerGrid(outFile, ebsdS, NY, NX, step);
        catch ME
            fprintf('   [skip] clean-%s: %s\n', fname_f, ME.message);
        end
    end
end

fprintf('\nDone. Results in %s\n', outDir);
fprintf('Now score them in Python against the clean maps.\n');


% =========================================================================
function writeEulerGrid(outFile, ebsd, NY, NX, step)
% Write Euler angles as NY*NX rows in DREAM.3D raster order (x fastest).
%
% MTEX does not guarantee that ebsd() keeps the original row order, and
% 'fill' can change the number of pixels, so we rebuild the grid explicitly
% from each pixel's x/y coordinate instead of trusting the order.

    E = ebsd.orientations.Euler;        % radians, Bunge ZXZ
    x = ebsd.x;  y = ebsd.y;           % MTEX 7 exposes these directly
                                        % (MTEX 5/6 used ebsd.prop.x / .prop.y)

    col = round(x(:) / step) + 1;
    row = round(y(:) / step) + 1;
    keep = col >= 1 & col <= NX & row >= 1 & row <= NY;

    grid = nan(NY * NX, 3);
    idx  = (row(keep) - 1) * NX + col(keep);
    grid(idx, :) = E(keep, :);

    nMissing = sum(any(isnan(grid), 2));
    if nMissing > 0
        fprintf('   note: %d pixels had no value, filled with 0\n', nMissing);
        grid(isnan(grid)) = 0;
    end

    fid = fopen(outFile, 'w');
    fprintf(fid, 'EulerAngles_0 EulerAngles_1 EulerAngles_2\n');
    fprintf(fid, '%.8g %.8g %.8g\n', grid');
    fclose(fid);
end