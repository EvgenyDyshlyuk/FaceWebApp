import os
files_in_save = os.listdir('tmp/cropped')
filenames_in_save = [os.path.splitext(x)[0] for x in files_in_save]
original_filenames = set([x[0:x.rfind('_')] for x in filenames_in_save])
print(original_filenames)
