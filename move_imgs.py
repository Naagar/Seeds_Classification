import shutil
import os
    
source_dir = 'silkcut'
target_dir = 'all_images'
    
file_names = os.listdir(source_dir)
    
for file_name in file_names:
    shutil.move(os.path.join(source_dir, file_name), target_dir)