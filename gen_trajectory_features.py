
# coding: utf-8

# In[1]:

from __future__ import print_function
import os
from glob import glob
import time
import os
import socket
from IPython import parallel
from IPython.parallel.util import interactive


### Compute Dense Trajectories

#### Set up cluster

# In[2]:

c = parallel.Client(profile='sge', sshserver='sdaptardar@130.245.4.230')
view = c[:]
view.block = False
c.ids


# In[2]:




#### Setup the dataset locations

# In[3]:

#src_densetraj = '/home/sdaptardar/actreg/dense_trajectory_release_v1.2'
#bin_densetraj = src_densetraj + os.sep + 'release' + os.sep + 'DenseTrack'
src_densetraj = '/home/sdaptardar/actreg/improved_trajectory_release'
bin_densetraj = src_densetraj + os.sep + 'release' + os.sep + 'DenseTrackStab'

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2'
dset_inp_dir =  dset_dir + os.sep + 'Hollywood2/'
dset_inp_prefix = 'AVIClips' + os.sep + 'actionclip'


# In[4]:

train_files = [vfile for vfile in glob(dset_inp_dir + dset_inp_prefix + 'train' + '*')]
test_files = [vfile for vfile in glob(dset_inp_dir + dset_inp_prefix + 'test' + '*')]

print('#Train: ', len(train_files))
print('#Test: ', len(test_files))


# In[5]:

#dset_out_dir = dset_dir + os.sep + 'HollyWood2_BOF_Results'
dset_out_dir = dset_dir + os.sep + 'Improved_Traj'
dset_out_feat_train_dir = dset_out_dir + os.sep + 'train'
dset_out_feat_test_dir = dset_out_dir + os.sep + 'test'
view.push(dict(doftr_dir = dset_out_feat_train_dir, 
               dofte_dir = dset_out_feat_test_dir, 
               bin_densetraj = bin_densetraj)) 


# In[6]:

if not os.path.exists(dset_out_dir):
    os.mkdir(dset_out_dir)
if not os.path.exists(dset_out_feat_train_dir):
    os.mkdir(dset_out_feat_train_dir)
if not os.path.exists(dset_out_feat_test_dir):
    os.mkdir(dset_out_feat_test_dir)
    


#### For running Dense Trajectory executable 

# In[7]:



def run_dt(video, ffile):
    fout = open(ffile, 'w')
    proc = Popen([bin_densetraj, video],  bufsize=1, stdout=fout, stderr=STDOUT, close_fds=True)
    #for line in iter(proc.stdout.readline, b''):
    #    print(line)
    fout.close()
    #proc.stdout.close()
    proc.wait()

def run_tr_dense_traj(tr_chunk):
    import time
    from subprocess import Popen, STDOUT, PIPE
    import os
    import socket
    
    def run_dt(video, ffile):
        fout = open(ffile, 'w')
        proc = Popen([bin_densetraj, video],  bufsize=1, stdout=fout, stderr=STDOUT, close_fds=True)
        fout.close()
        proc.wait()
    
    M = []
    for tr in tr_chunk:
        #print(tr)
        t1 = time.time()
        feat_file = doftr_dir + os.sep + os.path.splitext(os.path.basename(tr))[0]+ ".txt"
        run_dt(tr, feat_file)
        t2 = time.time()
        time_taken = t2 - t1
        msg = '%s,%d,%s,%f\n' % (tr, os.getpid(), socket.gethostname(), time_taken)
        print(msg)
        M.append(msg)
    return M
 

#def run_te_dense_traj(te_chunk):
#    import time
#    from subprocess import Popen, STDOUT, PIPE
#    import os
#    import socket
#    
#    M = []
#    
#    def run_dt(video, ffile):
#        fout = open(ffile, 'w')
#        proc = Popen([bin_densetraj, video],  bufsize=1, stdout=fout, stderr=STDOUT, close_fds=True)
#        fout.close()
#        proc.wait()
#
#    for te in te_chunk:
#        #print(te)
#        t1 = time.time()
#        feat_file = dofte_dir + os.sep + \
#                    os.path.splitext(os.path.basename(te))[0]+ ".txt"
#        run_dt(te, feat_file)
#        t2 = time.time()
#        time_taken = t2 - t1
#        msg = '%s,%d,%s,%f\n' % (te, os.getpid(), socket.gethostname(), time_taken)
#        print(msg)
#    return M


#### Run

# In[8]:

chunks1 = view.scatter('chunks1', [tr for tr in train_files])
#chunks2 = view.scatter('chunks2', [te for te in test_files])
res1 = view.map(run_tr_dense_traj, view['chunks1'])    
#res2 = view.map(run_te_dense_traj, view['chunks2'])


#### Monitor progress

# In[9]:

import time
while not res1.ready():
    time.sleep(30)
    print(res1.progress)


# In[10]:

print('Ready: ', res1.ready())
print('Progress: ', res1.progress)


# In[11]:

res1.result


# In[12]:

#import time
#while not res2.ready():
#    time.sleep(1)
#    print(res2.progress)


# In[13]:

#print('Ready: ', res2.ready())
#print('Progress: ', res2.progress)


# In[14]:

#res2.result


# In[14]:



