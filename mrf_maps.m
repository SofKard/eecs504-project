
%% Low-Rank MRF Reconstruction
data_location = pwd;
load(fullfile(data_location,'data/MRXCAT/4CH_slice12/DATA.mat'))
load(fullfile(data_location,'data/MRXCAT/4CH_slice12/dict.mat'))

% DATA: k-space data
% dict: already contains Bloch simulations for several T1/T2 values

% include github: https://github.com/OpenMRF/openmrf-core-matlab/tree/9ac6efe2290599ada19341c8fc57bc76cbe74e0b

path_cmrf = 'openmrf-core-matlab/include_cwru/';

addpath('path_cmrf')
addpath('openmrf-core-matlab/include_miitt/')

[~,SS,V] = svd(dict.','econ');
svals = diag(SS)/SS(1);
svalsEnergy = cumsum(svals.^2./sum(svals.^2));
K = find(svalsEnergy > 0.9999,1,'first'); % K is the rank of the compressed dictionary
Phi = single(V(:,1:K)); % this is a matrix with size [TRs, K]. If we multiply the MRF images (or the MRF k-space data) by Phi, we can project the data onto this low-dimensional subspace.
dictCompressed = (dict.'*Phi).'; % this is the SVD-compressed dictionary with size [K, parameters]
fprintf('   Rank K=%d\n',K);
      

% See these papers for more details...
% Gastao Cruz, MRM 2019. "Sparsity and locally low rank regularization for MR fingerprinting".
% Jesse Hamilton, NMR Biomed 2019. "Simultaneous multislice cardiac magnetic resonance fingerprinting using low rank reconstruction".
FT = NUFFT(kxall/N+1i*kyall/N,wi/max(wi(:)),[0 0],[N N]); % NUFFT operator for fully sampled trajectory
Et = @(x)lowrankMRF2D_adjoint(x,coilmap,idproj,Phi,FT,numSpiralArms); % adjoint operator  (from k-space to image domain)
E = @(x)lowrankMRF2D_forward(x,Phi,FT,idproj,coilmap,[nr numSpiralArms]); % forward operator (from images to k-space)
y0 = E(Et(DATA));
unitv = sum(abs(DATA(:)))/sum(abs(y0(:))); % initial step size
fprintf('   computing initial guess\n');
tic; x0 = Et(DATA); timeAdjoint=toc;
fprintf('      %.2f seconds\n',timeAdjoint);
scaling = max(abs(x0(:))); % normalize image, so we can use the same regularization parameters for different datasets
x0 = x0/scaling; % initial guess for the images
DATA = DATA/scaling;

% Reconstruction settings
params = struct( );
params.block_dim = [6,6];  % locally low-rank patch size
params.block_step = 6;
params.lambdaLLR = 0.02;                % locally low-rank regularization
params.lambdaSpatialTV = 0.003;   % spatial TV regularization
params.lambdaWav = 0;
params.betaMethod = 'Dai-Yuan';
params.beta = 0.6;
params.alpha = 0.01;
params.numIter = 25;   % max number of iterations
params.t0 = unitv;      % initial step
params.stopThresh = 0;
params.updateFreq = 0;      % how often to show intermediate results
params.pixelRangeToShow = 1:N; % crop the central 1x FOV

[images_lowrank,t0,dx,obj,update] = nonlinearCGDescent(x0*0,[],E,Et,DATA,params);

 % Now match the images to the dictionary to get T1, T2, and M0 maps.
fprintf('   matching to dictionary\n');
[T1_lowrank,T2_lowrank,M0_lowrank] = mrf_patternmatch(images_lowrank,ones(N),r,dictCompressed,12);
save(fullfile(savedir,'Tmaps_LR.mat'),'T1_lowrank','T2_lowrank','M0_lowrank');
save(fullfile(savedir,'LR_images.mat'),'images_lowrank');