from shutil import copyfile, copy
from os import listdir
from os.path import isfile, join

from tqdm import tqdm


from sklearn.model_selection import train_test_split


mypath = "./"
files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.npy' in f]

full_scan_files = list(filter(lambda f: 'FRUSTUM' not in f, files))
frustum_files = list(filter(lambda f: 'FRUSTUM' in f, files))

full_scan_ids = set((scan_id.split('_')[0] for scan_id in full_scan_files))
frustum_scan_ids = set((scan_id.split('_')[0] for scan_id in frustum_files))

# Split into train and test sets
full_scan_train_ids, full_scan_test_ids = train_test_split(list(full_scan_ids), test_size = .08)
frustum_scan_train_ids, frustum_scan_test_ids = train_test_split(list(frustum_scan_ids), test_size = .08)


# Split train set into train and validation set
full_scan_train_ids, full_scan_val_ids = train_test_split(full_scan_train_ids, test_size = .08)
frustum_scan_train_ids, frustum_scan_val_ids = train_test_split(frustum_scan_train_ids, test_size = .09)



# Write full scans into files
for f in tqdm(full_scan_files):
    f_id = f.split('_')[0]
    if f_id in full_scan_train_ids:
        copyfile(join(mypath, f), join(mypath, 'full_scan', 'train',f))
    elif f_id in full_scan_val_ids:
        copyfile(join(mypath, f), join(mypath, 'full_scan', 'val',f))
    else:
        copyfile(join(mypath, f), join(mypath, 'full_scan', 'test',f))

# Write frustum scans into files
for f in tqdm(frustum_files):
    f_id = f.split('_')[0]
    if f_id in frustum_scan_train_ids:
        copyfile(join(mypath, f), join(mypath, 'frustum_scan', 'train',f))
    elif f_id in frustum_scan_val_ids:
        copyfile(join(mypath, f), join(mypath, 'frustum_scan', 'val',f))
    else:
        copyfile(join(mypath, f), join(mypath, 'frustum_scan', 'test',f))

