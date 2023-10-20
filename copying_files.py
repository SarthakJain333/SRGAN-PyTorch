import os
import shutil

data_path = 'celebA_Dataset/img_align_celeba/img_align_celeba/'
destination_directory = 'celeb_10k'
cnt = 0

for img_path in os.listdir(data_path):
    shutil.copy(data_path+img_path, destination_directory)
    cnt+=1
    if(cnt>10000):
        break

print('Transfer Done Successfully!!')


