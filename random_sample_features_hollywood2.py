from __future__ import print_function
import mmap
import os
import glob
import cPickle as pickle
import bisect
import numpy.random
import time

feat_base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results'
feat_train_dir = feat_base_dir + os.sep + 'train'
feat_test_dir = feat_base_dir + os.sep + 'test'
feature_folder = feat_train_dir
metadata_folder = '.'
num_features = 256000
dset_name = 'train'


def read_feat(ffile, num):
    feat = None
    if num < 0:
        return
    with open(ffile, 'r+b') as f:
        mm = None
        try:
            mm = mmap.mmap(fileno=f.fileno(), length=0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=0)
            if num > 0:
                for i in range(num):
                    mm.readline()
            feat = mm.readline()
        finally:
            if not mm == None:
                mm.close()
    return map(float, feat.strip().split())


def line_count(ffile):
    line_count = 0
    with open(ffile, 'r') as f:
        mm = None
        line = ''
        try:
            mm = mmap.mmap(fileno=f.fileno(), length=0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ, offset=0)
            while True:
                line = mm.readline()
                if line == '': break
                line_count = line_count + 1
        finally:
            if not mm == None:
                mm.close()
    return line_count


def create_list(folder, pattern, name):
    pat = folder + os.sep + pattern 
    fileid = 0
    file_map      = {}
    file_r_map    = {}
    filesize_map = {}  
    for f in glob.glob(pat):
        f_name = os.path.split(f)[-1]
        file_map[fileid] = f_name
        file_r_map[f_name] = fileid
        filesize_map[f_name] = line_count(f)
        print(f_name, ' ', filesize_map[f_name]) 
        fileid = fileid + 1
    file_map_fn = '%s_filemap.pkl' % name
    file_r_map_fn = '%s_filerevmap.pkl' % name
    filesize_map_fn  = '%s_filesizemap.pkl' % name

    with open(file_map_fn, 'wb') as file_map_f:
        pickle.dump(file_map, file_map_f)

    with open(file_r_map_fn, 'wb') as file_r_map_f:
        pickle.dump(file_r_map, file_r_map_f)

    with open(filesize_map_fn, 'wb') as filesize_map_f:
        pickle.dump(filesize_map, filesize_map_f) 

    
def load_list(folder, name):
    file_map_fn = '%s_filemap.pkl' % name
    file_r_map_fn = '%s_filerevmap.pkl' % name
    filesize_map_fn  = '%s_filesizemap.pkl' % name

    with open(file_map_fn, 'rb') as file_map_f:
        file_map = pickle.load(file_map_f)

    with open(file_r_map_fn, 'rb') as file_r_map_f:
        file_r_map = pickle.load(file_r_map_f)

    with open(filesize_map_fn, 'rb') as filesize_map_f:
        filesize_map = pickle.load(filesize_map_f) 

    return {'file_map': file_map, 'file_r_map': file_r_map, 'filesize_map': filesize_map }


def calc_num_features(folder, name):
    L = load_list(folder, name)
    num_files = len(L['file_map'])
    filesize_sum = [0] * num_files
    filesize_cumsum = []

    for x, v in L['filesize_map'].iteritems():
        filesize_sum[L['file_r_map'][x]] = v

    tot_feat = 0
    for i in range(num_files):
        tot_feat = tot_feat + filesize_sum[i]
        filesize_cumsum.append(tot_feat)

    numfeat_fn = '%s_numfeat.pkl' % name
    filesizecumsum_fn  = '%s_filesizecumsum.pkl' % name

    with open(numfeat_fn, 'wb') as numfeat_f:
        pickle.dump(tot_feat, numfeat_f)

    with open(filesizecumsum_fn, 'wb') as filesizecumsum_f:
        pickle.dump(filesize_cumsum, filesizecumsum_f)

    return tot_feat


def load_metadata(folder, name):
    file_map_fn = '%s_filemap.pkl' % name
    file_r_map_fn = '%s_filerevmap.pkl' % name
    filesize_map_fn  = '%s_filesizemap.pkl' % name
    numfeat_fn = '%s_numfeat.pkl' % name
    filesizecumsum_fn  = '%s_filesizecumsum.pkl' % name
    
    with open(file_map_fn, 'rb') as file_map_f:
        file_map = pickle.load(file_map_f)

    with open(file_r_map_fn, 'rb') as file_r_map_f:
        file_r_map = pickle.load(file_r_map_f)

    with open(filesize_map_fn, 'rb') as filesize_map_f:
        filesize_map = pickle.load(filesize_map_f) 

    with open(numfeat_fn, 'rb') as numfeat_f:
        tot_feat = pickle.load(numfeat_f)

    with open(filesizecumsum_fn, 'rb') as filesizecumsum_f:
        filesize_cumsum = pickle.load(filesizecumsum_f)

    return {'file_map': file_map, \
            'file_r_map': file_r_map, \
            'filesize_map': filesize_map, \
            'tot_feat': tot_feat,  \
            'filesize_cumsum': filesize_cumsum }


def find_gt(a, x):
    i = bisect.bisect_left(a, x)
    if i > 0:
        return (i, a[i-1])
    else:
        return (0, 0)
    raise ValueError

def get_single_feature(feature_folder, mdata, num):

        ub_idx, lb = find_gt(mdata['filesize_cumsum'], num)
        offset = num - lb        
        print('Upper Bound: %d %d %d ' % (ub_idx, lb, offset))
        fn = mdata['file_map'][ub_idx]
        fname = '%s%s%s' % (feature_folder, os.sep, fn)
        print('Filename: %s' % fname)
        return (fn, offset, read_feat(fname, offset))


def gen_random_sample_of_features(feature_folder, metadata_folder, name, num_feat):

    L = load_metadata(metadata_folder, name)
    N = L['tot_feat']

    featureids = numpy.random.permutation(range(N))[0:num_feat]
    # print(featureids)
    # print('Length: %d', len(featureids))

    random_sample_fn = '%s_randomsample.pkl' % name

    random_sample = []
    cnt = 0
    for fid in featureids:
        t1 = time.time()
        (fn, offset, F) = get_single_feature(feature_folder, L, fid)
        random_sample.append((fn, offset, F))
        t2 = time.time()
        cnt = cnt + 1
        print('%d : %f' % (cnt, (t2 - t1)))

    with open(random_sample_fn, 'wb') as random_sample_f:
        pickle.dump(random_sample, random_sample_f)


def load_random_sample_of_features(metadata_folder, name):

    random_sample_fn = '%s_randomsample.pkl' % name
    with open(random_sample_fn, 'rb') as random_sample_f:
        random_sample = pickle.load(random_sample_f)
        print(random_sample)

# feat = read_feat(ffile, 10)
# print(feat)

# print(line_count(ffile))

# create_list('/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results/train', 'actioncliptrain*.txt', 'train')

# L = load_list('.', 'train')

# print(L['file_map'])
# print()
# print()

# print(L['file_r_map'])
# print()
# print()

# print(L['filesize_map'])
# print()
# print()

#print('Total number of features: ', calc_num_features('.', 'train'))
#L = load_metadata('.', 'train')

# print(L['file_map'])
# print()
# print()
# 
# print(L['file_r_map'])
# print()
# print()
# 
# print(L['tot_feat'])
# print()
# print()
# 
# 
# print(L['filesize_cumsum'])
# print()
# print()


gen_random_sample_of_features(feature_folder, metadata_folder, dset_name, num_features)
#load_random_sample_of_features(metadata_folder, dset_name)
