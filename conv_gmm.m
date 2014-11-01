function conv_gmm(indir, outfile)

PCA = {};
GMM = {};
GMM.trajXY   = {};
GMM.trajHog  = {};
GMM.trajHof  = {};
GMM.trajMbhx = {};
GMM.trajMbhy = {};

infile_traj = [indir filesep 'Trajectory' '.mat'];
infile_hog = [indir filesep 'HOG' '.mat'];
infile_hof = [indir filesep 'HOF' '.mat'];
infile_mbhx = [indir filesep 'MBHx' '.mat'];
infile_mbhy = [indir filesep 'MBHy' '.mat'];

traj = load(infile_traj);
GMM.trajXY.mus      = traj.means;
GMM.trajXY.covs     = traj.covariances;
GMM.trajXY.priors   = traj.priors;
PCA.trajXY.mu       = traj.mu';
PCA.trajXY.sigma    = traj.sigma';
PCA.trajXY.PcaBasis = traj.coeff(:,1:15); 

trajHog = load(infile_hog);
GMM.trajHog.mus      = trajHog.means;
GMM.trajHog.covs     = trajHog.covariances;
GMM.trajHog.priors   = trajHog.priors;
PCA.trajHog.mu       = trajHog.mu';
PCA.trajHog.sigma    = trajHog.sigma';
PCA.trajHog.PcaBasis = trajHog.coeff(:,1:48); 


trajHof = load(infile_hof);
GMM.trajHof.mus      = trajHof.means;
GMM.trajHof.covs     = trajHof.covariances;
GMM.trajHof.priors   = trajHof.priors;
PCA.trajHof.mu       = trajHof.mu';
PCA.trajHof.sigma    = trajHof.sigma';
PCA.trajHof.PcaBasis = trajHof.coeff(:,1:54); 


trajMbhx = load(infile_mbhx);
GMM.trajMbhx.mus      =  trajMbhx.means;
GMM.trajMbhx.covs     =  trajMbhx.covariances;
GMM.trajMbhx.priors   =  trajMbhx.priors;
PCA.trajMbhx.mu       =  trajMbhx.mu';
PCA.trajMbhx.sigma    =  trajMbhx.sigma';
PCA.trajMbhx.PcaBasis =  trajMbhx.coeff(:,1:48);

trajMbhy = load(infile_mbhy);
GMM.trajMbhy.mus      =  trajMbhy.means;
GMM.trajMbhy.covs     =  trajMbhy.covariances;
GMM.trajMbhy.priors   =  trajMbhy.priors;
PCA.trajMbhy.mu       =  trajMbhy.mu';
PCA.trajMbhy.sigma    =  trajMbhy.sigma';
PCA.trajMbhy.PcaBasis =  trajMbhy.coeff(:,1:48);

disp(GMM.trajMbhy);
disp(PCA.trajMbhy);

save(outfile, 'PCA', 'GMM');

end
