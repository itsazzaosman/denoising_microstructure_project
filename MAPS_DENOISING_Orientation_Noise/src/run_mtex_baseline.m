% =========================================================================
% run_mtex_baseline.m  (rewritten 2026-10-14)
%
% Runs the six MTEX denoising filters, each with and without pre-cleaning,
% on the test maps in the MTEX package
% folder (datasets/mtex_package, made by src/make_mtex_package.py), on a
% Windows or Linux MATLAB with MTEX 7.1.
%
% HOW TO RUN
%   1. Copy the whole mtex_package folder to the MTEX machine.
%   2. Start MATLAB and MTEX (startup_mtex) as usual.
%   3. Open this file FROM INSIDE mtex_package and press Run. No paths to
%      edit: the script finds its data next to itself.
%   4. Copy mtex_package/mtex_out back to the cluster and run
%      src/import_mtex_results.sh (see COMMANDS.md).
%
% INPUT   ang/<set>/map_XXXXX_clean_euler_noisy_noisy.ang
% OUTPUT  mtex_out/<set>/map_XXXXX_clean_euler_noisy_noisy__<filter>.txt
%         Euler angles (radians, Bunge ZXZ), one row per pixel in DREAM.3D
%         order (x fastest), the format the Python scorer reads.
%         mtex_out/<set>/mtex_log.csv: threshold, seconds and unfilled
%         pixels for every output.
%
% FILTERS (settings as in the September 2026 runs)
%   mean          default window (1 ring of neighbours)
%   median        numNeighbours = 2
%   kuwahara      numNeighbours = 2
%   spline        smoothing strength chosen automatically
%   halfquad      half-quadratic, alpha = 0.25
%   infconv       infimal convolution, default parameters
% Each filter is applied grain by grain to the raw map, and again as
% clean-<filter>: first delete the grains smaller than minPixel pixels (this
% removes isolated misindexed pixels), then smooth and fill the deleted pixels
% from their neighbours. 12 outputs per map.
%
% GRAINS are detected with the threshold angle max(10, factor * theta_max)
% degrees. Two pixels of one grain can differ by up to 2 * theta_max, and a
% smaller threshold splits grains apart. MTEX is therefore given the true
% scatter level of each set (of each map for test_random, from levels.csv),
% information the network does not get.
%
% Outputs that already exist are skipped, so a stopped run can be restarted.
% =========================================================================

clear; close all;

% ----------------------------- SETTINGS ---------------------------------
maps     = 1;          % which test maps: 1 = first map only, 1:10 = all ten in the package
base     = {'mean', 'median', 'kuwahara', 'spline', 'halfquad', 'infconv'};
filters  = [base, strcat('clean-', base)];   % or a subset, e.g. {'spline', 'clean-spline'}
factor   = 2;          % grain threshold = max(10, factor * theta_max) degrees
minPixel = 3;          % clean-*: delete grains smaller than this many pixels
numNeighbours = 2;     % median and Kuwahara window: (2*2+1) x (2*2+1) pixels
hqAlpha  = 0.25;       % half-quadratic regularisation
setsToRun = {};        % empty = every set in sets.txt; or e.g. {'scatter5', 'scatter5_mis6'}
NY = 128; NX = 128; step = 1.0;
% ------------------------------------------------------------------------

if ~exist('EBSD', 'class')
  error('MTEX is not loaded: run startup_mtex first.');
end

pkgDir = fileparts(mfilename('fullpath'));
if isempty(setsToRun)
  setsToRun = strtrim(splitlines(fileread(fullfile(pkgDir, 'sets.txt'))));
  setsToRun = setsToRun(~cellfun(@isempty, setsToRun));
end

CS = crystalSymmetry('m-3m', [3.524 3.524 3.524], 'mineral', 'Nickel');
setMTEXpref('xAxisDirection', 'east');
setMTEXpref('zAxisDirection', 'intoPlane');

tAll = tic;
for s = 1:numel(setsToRun)
  setName = setsToRun{s};
  angDir = fullfile(pkgDir, 'ang', setName);
  outDir = fullfile(pkgDir, 'mtex_out', setName);
  if ~isfolder(outDir), mkdir(outDir); end
  thetaMax = scatterLevels(angDir, setName, maps);

  logFile = fullfile(outDir, 'mtex_log.csv');
  if ~isfile(logFile)
    fid = fopen(logFile, 'w');
    fprintf(fid, 'map,filter,threshold_deg,seconds,unfilled_pixels\n');
    fclose(fid);
  end

  fprintf('[%d/%d] %s\n', s, numel(setsToRun), setName);
  for k = 1:numel(maps)
    id = maps(k);
    stem = sprintf('map_%05d_clean_euler_noisy_noisy', id);
    angFile = fullfile(angDir, [stem '.ang']);
    if ~isfile(angFile)
      warning('missing %s, skipped', angFile); continue
    end
    outFile = @(f) fullfile(outDir, sprintf('%s__%s.txt', stem, f));
    todo = filters(~cellfun(@(f) isfile(outFile(f)), filters));
    if isempty(todo), continue; end

    % 'setting', 0: these synthetic .ang files carry no vendor convention,
    % so MTEX must not apply its default coordinate correction (checked: the
    % round trip is exact with this option).
    ebsd = EBSD.load(angFile, CS, 'interface', 'ang', 'setting', 0);
    thr = max(10, factor * thetaMax(k));
    [grains, ebsd] = withGrains(ebsd('indexed'), thr);

    for f = 1:numel(todo)
      t0 = tic;
      name = todo{f};
      F = makeFilter(erase(name, 'clean-'), numNeighbours, hqAlpha);
      if startsWith(name, 'clean-')
        small = grains(grains.numPixel < minPixel);
        ebsdC = ebsd(~ismember(ebsd.grainId, small.id));
        [~, ebsdC] = withGrains(ebsdC, thr);
        ebsdS = smooth(ebsdC, F, 'fill');
      else
        ebsdS = smooth(ebsd, F);
      end
      missing = writeEulerGrid(outFile(name), ebsdS, NY, NX, step);
      if missing > 0
        warning('%s %s: %d pixels have no value (written as NaN)', stem, name, missing);
      end
      fid = fopen(logFile, 'a');
      fprintf(fid, '%d,%s,%.2f,%.2f,%d\n', id, name, thr, toc(t0), missing);
      fclose(fid);
    end
    fprintf('   map %d done (threshold %.1f deg)\n', id, thr);
  end
end
fprintf('\nAll done in %.1f min. Results in %s\n', toc(tAll) / 60, fullfile(pkgDir, 'mtex_out'));


% =========================================================================
function [grains, ebsd] = withGrains(ebsd, thrDeg)
% Grain reconstruction, with the grain id stored in ebsd so that smooth()
% filters each grain separately. MTEX 7 returns the EBSD as the second
% output; older versions return the grain ids.
[grains, second] = calcGrains(ebsd, 'angle', thrDeg * degree);
if isa(second, 'EBSD')
  ebsd = second;
else
  ebsd.grainId = second;
end
end


function F = makeFilter(name, numNeighbours, hqAlpha)
switch name
  case 'mean'
    F = meanFilter;
  case 'median'
    F = medianFilter;
    F.numNeighbours = numNeighbours;
  case 'kuwahara'
    F = KuwaharaFilter;
    F.numNeighbours = numNeighbours;
  case 'spline'
    F = splineFilter;                 % smoothing strength chosen automatically
  case 'halfquad'
    F = halfQuadraticFilter;
    F.alpha = hqAlpha;
  case 'infconv'
    F = infimalConvolutionFilter;
  otherwise
    error('unknown filter %s', name);
end
end


function theta = scatterLevels(angDir, setName, maps)
% theta_max of every map: from the set name, or per map for test_random.
tok = regexp(setName, '^scatter(\d+)', 'tokens', 'once');
if ~isempty(tok)
  theta = repmat(str2double(tok{1}), size(maps));
  return
end
levels = readtable(fullfile(angDir, 'levels.csv'));
theta = zeros(size(maps));
for k = 1:numel(maps)
  row = levels.map == maps(k);
  if ~any(row), error('map %d missing from levels.csv of %s', maps(k), setName); end
  theta(k) = levels.scatter_max_deg(row);
end
end


function missing = writeEulerGrid(outFile, ebsd, NY, NX, step)
% Write Euler angles as NY*NX rows in DREAM.3D raster order (x fastest),
% rebuilt from each pixel's x/y because MTEX does not keep the row order.
% A pixel with no value is written as NaN, never as a made-up orientation.
E = ebsd.orientations.Euler;            % radians, Bunge ZXZ
col = round(ebsd.x(:) / step) + 1;
row = round(ebsd.y(:) / step) + 1;
keep = col >= 1 & col <= NX & row >= 1 & row <= NY & all(isfinite(E), 2);
grid = nan(NY * NX, 3);
grid((row(keep) - 1) * NX + col(keep), :) = E(keep, :);
missing = sum(any(isnan(grid), 2));

fid = fopen(outFile, 'w');
fprintf(fid, 'EulerAngles_0 EulerAngles_1 EulerAngles_2\n');
fprintf(fid, '%.8g %.8g %.8g\n', grid.');
fclose(fid);
end
