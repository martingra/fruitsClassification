
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

dataset_dir = os.path.dirname(os.getcwd()) + '\\DataSet\\'

os.listdir(dataset_dir)

test_path = dataset_dir+'\\test\\'
train_path = dataset_dir+'\\train\\'


# Let's get all the classes we have.
print('# of classes in the dataset: ' + str(len(os.listdir(test_path))))

# Let's check how many images there are.
tot_img_test = 0
for cl in os.listdir(test_path):
    print('# of imgs in the class '+ cl +' (test) : ' + str(len(os.listdir(test_path+"\\"+cl))))
    tot_img_test = tot_img_test + len(os.listdir(test_path+"\\"+cl))
print('Average # of imgs per class (test): ' + str(tot_img_test / len(os.listdir(test_path))))

tot_img_train = 0
for cl in os.listdir(train_path):
    print('# of imgs in the class '+ cl +' (train) : ' + str(len(os.listdir(train_path+"\\"+cl))))
    tot_img_train = tot_img_test + len(os.listdir(train_path+"\\"+cl))
print('Average # of imgs per class (train): ' + str(tot_img_train / len(os.listdir(train_path))))

# Let's check dimension of images in the folders
dim1 = []
dim2 = []
for cl in os.listdir(test_path):
    for image_filename in os.listdir(test_path+'\\'+cl):
        img = imread(test_path+'\\'+cl+'\\'+image_filename)
        d1,d2,colors = img.shape
        dim1.append(d1)
        dim2.append(d2)

sns.jointplot(dim1,dim2)
plt.show()

np.mean(dim1)
np.mean(dim2)