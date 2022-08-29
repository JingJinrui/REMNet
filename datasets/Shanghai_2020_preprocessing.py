import os
import random
from configs import config_sh


# divide the datasets into train, valid, and test set
root = config_sh.dataset_root
f_r = os.path.join(root, 'train', 'examples')
f_w = open(os.path.join(root, 'Shanghai_2020_train_examples.txt'), 'w')
f_w1 = open(os.path.join(root, 'Shanghai_2020_valid_examples.txt'), 'w')
f_w2 = open(os.path.join(root, 'Shanghai_2020_test_examples.txt'), 'w')
examples = os.listdir(f_r)
random.seed(42)
random.shuffle(examples)

for index, example in enumerate(examples):
    example_root = os.path.join(root, 'train', 'examples', example)
    if index < 33000:
        f_w.write(example_root + '\n')
    elif index < 35000:
        f_w1.write(example_root + '\n')
    else:
        f_w2.write(example_root + '\n')

f_w.close()
f_w1.close()
f_w2.close()
