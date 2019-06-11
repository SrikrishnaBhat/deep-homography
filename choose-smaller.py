import os
import shutil

def copy_n_files(src_dir, dest_dir, amount):
    src_files = os.listdir(src_dir)
    limit = min(len(src_files), amount)
    for i in range(limit):
        if i%100 == 0:
            print('Done {}'.format(i))
        shutil.copy(os.path.join(src_dir, src_files[i]), dest_dir)

src_train_data = 'train-data'
src_val_data = 'val-data'

dest_train_data = 'train-data-small'
dest_val_data = 'val-data-small'

if not os.path.exists(dest_train_data):
    os.makedirs(dest_train_data)

if not os.path.exists(dest_val_data):
    os.makedirs(dest_val_data)

train_size = 100000
val_size = 10000

# print('Copying {} training files'.format(train_size))
# copy_n_files(src_train_data, dest_train_data, train_size)

print('Copying {} validation files'.format(val_size))
copy_n_files(src_val_data, dest_val_data, val_size)
