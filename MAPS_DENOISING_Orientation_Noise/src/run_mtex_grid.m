function run_mtex_grid(setName, varargin)
% RUN_MTEX_GRID  MTEX denoising filters on every map of one test set.
%
%   run_mtex_grid('scatter3_mis3')
%   run_mtex_grid('test_random', 'maps', 1:10)
%   run_mtex_grid('scatter15', 'filters', {'spline'})
%
% Reads   datasets/noise_scatter_misindexing/noise_<set>/ang_files/map_XXXXX_clean_euler_noisy_noisy.ang
% Writes  datasets/results_test_set_mtex/<set>/map_XXXXX_clean_euler_noisy_noisy__<filter>.txt
%         (Euler angles, DREAM.3D pixel order, the format score_dataset.py reads)
%         datasets/results_test_set_mtex/<set>/mtex_log.csv  (time and unfilled pixels per output)
%
% Filters (default all twelve): mean, median, kuwahara, spline, halfquad,
% infconv applied grain by grain to the raw map, and clean-<filter>: first
% delete grains smaller than minPixel (this removes isolated misindexed
% pixels), then smooth and fill the deleted pixels ('fill'). Settings as in
% run_mtex_baseline.m.
%
% Grains are detected with a threshold angle of max(10, factor * theta_max)
% degrees: two pixels of one grain can differ by up to 2 * theta_max, and a
% threshold below that splits grains apart. MTEX is therefore given the true
% scatter level of each set (per map for test_random, from levels.csv),
% information the network does not get.
%
% Outputs that already exist are skipped, so a stopped run can be restarted.
% Run on the cluster with
%   matlab -batch "addpath('<src>'); run_mtex_grid('scatter3_mis3')"
% where startup has run MTEX's startup_mtex (run_mtex_grid.sh does this).

p = inputParser;
p.addParameter('maps', 1:500);
base = {'mean', 'median', 'kuwahara', 'spline', 'halfquad', 'infconv'};
p.addParameter('filters', [base, strcat('clean-', base)]);
p.addParameter('factor', 2);         % threshold = max(10, factor * theta_max)
p.addParameter('minPixel', 3);       % clean-*: delete grains smaller than this
p.addParameter('numNeighbours', 2);   % median and Kuwahara
p.addParameter('hqAlpha', 0.25);       % half-quadratic
p.parse(varargin{:});
opt = p.Results;

root   = fileparts(fileparts(mfilename('fullpath')));      % src/.. = project
angDir = fullfile(root, 'datasets', 'noise_scatter_misindexing', ['noise_' setName], 'ang_files');
outDir = fullfile(root, 'datasets', 'results_test_set_mtex', setName);
if ~isfolder(angDir), error('no .ang folder %s', angDir); end
if ~isfolder(outDir), mkdir(outDir); end

NY = 128; NX = 128; step = 1.0;
CS = crystalSymmetry('m-3m', [3.524 3.524 3.524], 'mineral', 'Nickel');
setMTEXpref('xAxisDirection', 'east');
setMTEXpref('zAxisDirection', 'intoPlane');

thetaMax = scatterLevels(root, setName, opt.maps);

logFile = fullfile(outDir, 'mtex_log.csv');
if ~isfile(logFile)
  fid = fopen(logFile, 'w');
  fprintf(fid, 'map,filter,threshold_deg,seconds,unfilled_pixels\n');
  fclose(fid);
end

fprintf('%s: %d maps, filters %s\n', setName, numel(opt.maps), strjoin(opt.filters, ', '));
for k = 1:numel(opt.maps)
  id = opt.maps(k);
  stem = sprintf('map_%05d_clean_euler_noisy_noisy', id);
  outFile = @(f) fullfile(outDir, sprintf('%s__%s.txt', stem, f));
  todo = opt.filters(~cellfun(@(f) isfile(outFile(f)), opt.filters));
  if isempty(todo), continue; end

  % 'setting', 0: these synthetic .ang files carry no vendor convention, so
  % MTEX must not apply its default coordinate correction.
  ebsd = EBSD.load(fullfile(angDir, [stem '.ang']), CS, 'interface', 'ang', 'setting', 0);
  thr = max(10, opt.factor * thetaMax(k));
  [grains, ebsd] = withGrains(ebsd('indexed'), thr);

  for f = 1:numel(todo)
    t0 = tic;
    name = todo{f};
    F = makeFilter(erase(name, 'clean-'), opt);
    if startsWith(name, 'clean-')
      small = grains(grains.numPixel < opt.minPixel);
      ebsdC = ebsd(~ismember(ebsd.grainId, small.id));
      [~, ebsdC] = withGrains(ebsdC, thr);
      ebsdS = smooth(ebsdC, F, 'fill', 'extrapolate');   % extrapolate: also fill deleted pixels on the map edge
    else
      ebsdS = smooth(ebsd, F);
    end
    missing = writeEulerGrid(outFile(name), ebsdS, NY, NX, step);
    fid = fopen(logFile, 'a');
    fprintf(fid, '%d,%s,%.2f,%.2f,%d\n', id, name, thr, toc(t0), missing);
    fclose(fid);
  end
  if mod(k, 25) == 0 || k == numel(opt.maps)
    fprintf('  %d/%d maps\n', k, numel(opt.maps));
  end
end
fprintf('done: %s\n', outDir);
end


% -------------------------------------------------------------------------
function [grains, ebsd] = withGrains(ebsd, thrDeg)
% Grain reconstruction, with the grain id stored in ebsd so smooth() filters
% each grain separately. MTEX 7 returns the EBSD as the second output; older
% versions return the grain ids.
[grains, second] = calcGrains(ebsd, 'angle', thrDeg * degree);
if isa(second, 'EBSD')
  ebsd = second;
else
  ebsd.grainId = second;
end
end


function F = makeFilter(name, opt)
switch name
  case 'mean'
    F = meanFilter;
  case 'median'
    F = medianFilter;
    F.numNeighbours = opt.numNeighbours;
  case 'kuwahara'
    F = KuwaharaFilter;
    F.numNeighbours = opt.numNeighbours;
  case 'spline'
    F = splineFilter;                 % smoothing strength chosen automatically
  case 'halfquad'
    F = halfQuadraticFilter;
    F.alpha = opt.hqAlpha;
  case 'infconv'
    F = infimalConvolutionFilter;
  otherwise
    error('unknown filter %s', name);
end
end


function theta = scatterLevels(root, setName, maps)
% theta_max of every map: from the set name, or per map for test_random.
tok = regexp(setName, '^scatter(\d+)', 'tokens', 'once');
if ~isempty(tok)
  theta = repmat(str2double(tok{1}), size(maps));
  return
end
levels = readtable(fullfile(root, 'datasets', 'noise_scatter_misindexing', ['noise_' setName], 'levels.csv'));
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
% Pixels with no value are written as NaN, never as a made-up orientation:
% the scorer then drops the map for that method and the n column shows it.
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
