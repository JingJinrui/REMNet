import os
import numpy as np
from PIL import Image
from configs import config_hko


def calculate_dataset_statistic(f_r, f_w):
    rainy_days = [line.split(',')[0] for line in f_r.readlines()]
    for rainy_day in rainy_days:
        rainy_day_root = os.path.join(root, 'radarPNG', rainy_day[0:4], rainy_day[4:6], rainy_day[6:8])
        rainy_day_imgs_paths = [os.path.join(rainy_day_root, img) for img in os.listdir(rainy_day_root)]
        rainy_day_imgs_nonzero_points = []
        rainy_day_imgs_nparr_mean = []
        rainy_day_imgs_nparr_nonzero_mean = []
        for rainy_day_img_path in rainy_day_imgs_paths:
            img = Image.open(rainy_day_img_path)
            img_nparr = np.array(img)
            nonzero_points = len(np.nonzero(img_nparr)[0])
            rainy_day_imgs_nonzero_points.append(nonzero_points)
            rainy_day_imgs_nparr_nonzero_mean.append(np.sum(img_nparr) / nonzero_points)
            rainy_day_imgs_nparr_mean.append(np.mean(img_nparr))
        rainy_day_imgs_nonzero_points = np.mean(np.array(rainy_day_imgs_nonzero_points))
        rainy_day_imgs_nparr_mean = np.mean(np.array(rainy_day_imgs_nparr_mean))
        rainy_day_imgs_nparr_nonzero_mean = np.mean(np.array(rainy_day_imgs_nparr_nonzero_mean))
        f_w.write(rainy_day + ', ' + '{:.4f}'.format(rainy_day_imgs_nonzero_points) + ', ' + '{:.4f}'.format(
            rainy_day_imgs_nparr_mean) + ', ' + '{:.4f}'.format(rainy_day_imgs_nparr_nonzero_mean) + '\n')
    f_r.close()
    f_w.close()


# calculate the nonzero points, mean value of the whole img, and mean value of nonzero points of each raniny day img
# the calculated results will be stored in a .txt file (path defined with f_w)
root = config_hko.dataset_root
f_r = open(os.path.join(root, 'hko7_rainy_train_days.txt'), 'r')
f_w = open(os.path.join(root, 'hko7_rainy_train_days_statistics.txt'), 'w')
calculate_dataset_statistic(f_r, f_w)

f_r = open(os.path.join(root, 'hko7_rainy_test_days.txt'), 'r')
f_w = open(os.path.join(root, 'hko7_rainy_test_days_statistics.txt'), 'w')
calculate_dataset_statistic(f_r, f_w)

f_r = open(os.path.join(root, 'hko7_rainy_valid_days.txt'), 'r')
f_w = open(os.path.join(root, 'hko7_rainy_valid_days_statistics.txt'), 'w')
calculate_dataset_statistic(f_r, f_w)
