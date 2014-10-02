#!/bin/sh
#$-cwd
#$-N DenseTraj
#$-j y
#$ -o /nfs/bigeye/sdaptardar/logs/log.$JOB_ID.out
#$ -e /nfs/bigeye/sdaptardar/logs/log.$JOB_ID.err
#$-M sdaptardar@cs.stonybrook.edu
#$-m ea
#########$-pe default 5
#########$-l mf=20G,h_vmem=22G
#$-pe default 8
#$-l mf=12G,h_vmem=16G

export LD_LIBRARY_PATH=/opt/matlab_r2010b/bin/glnxa64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
export DISPLAY=localhost:10.0
echo "Starting job: $JOB_ID"
#####matlab -nodesktop -nosplash -singleCompThread < /nfs/bigeye/sdaptardar/actreg/densetraj/randsamp2.m
#####matlab -nodesktop -nosplash < /nfs/bigeye/sdaptardar/actreg/densetraj/run_all_gmm.m
matlab -nodesktop -nosplash < /nfs/bigeye/sdaptardar/actreg/densetraj/run_all_fisher_encode.m
echo "Ending job: $JOB_ID"
