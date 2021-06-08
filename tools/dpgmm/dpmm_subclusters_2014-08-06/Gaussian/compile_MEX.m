% this might need to be changed if you have gsl installed somewhere else...
% on linux/unix, you can probably just leave them blank.
GSLFLAGS = ''; %' -I/usr/local/include/ -L/usr/local/lib/ ';

% this might need to be changed if you have eigen somewhere else...
% the following assumes that it's installed one directory up
curdir = pwd;
IFLAGS = '';
IFLAGS = [IFLAGS ' -I' curdir '/../common/ '];
IFLAGS = [IFLAGS ' -I' curdir '/../eigen/ '];

% this might need to be changed.  See here:
% http://openmp.org/wp/openmp-compilers/
OMPFLAG = ' -fopenmp ';

DFLAGS = ' -O ';

% if you are not using linux, the included stopwatch class will not work.
% In this case, please replace stopwatch.h or comment out the following
% line. If it is commented out, the 'time' will just be iteration count
DFLAGS = [DFLAGS ' -DUSESTOPWATCH '];

CFLAGS = [' CXXFLAGS="\$CXXFLAGS ' OMPFLAG '" CFLAGS="\$CFLAGS ' OMPFLAG '" '];
LFLAGS = [' LDFLAGS="\$LDFLAGS ' OMPFLAG '" -lgsl -lgslcblas -lm -lrt'];

cd 'include/';
try
    eval(['mex' DFLAGS IFLAGS CFLAGS LFLAGS GSLFLAGS ' dpgmm_calc_posterior.cpp niw.cpp']);
    eval(['mex' DFLAGS IFLAGS CFLAGS LFLAGS GSLFLAGS ' dpgmm_subclusters.cpp clusters.cpp niw_sampled.cpp normal.cpp']);
    eval(['mex' DFLAGS IFLAGS CFLAGS LFLAGS GSLFLAGS ' dpgmm_FSD.cpp clusters_FSD.cpp niw_sampled.cpp normal.cpp']);
    eval(['mex' DFLAGS IFLAGS CFLAGS LFLAGS GSLFLAGS ' dpgmm_sams.cpp cluster_single.cpp niw.cpp normal.cpp']);
    eval(['mex' DFLAGS IFLAGS CFLAGS LFLAGS GSLFLAGS ' dpgmm_sams_superclusters.cpp cluster_single.cpp niw.cpp normal.cpp']);
catch exception
	 cd '..';
	 rethrow(exception);
end
cd '..';