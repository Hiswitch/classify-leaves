import pandas as pd

train_data = pd.read_csv('course\ML\lab\classify-leaves\\train.csv')
labels = sorted(list(set(train_data['label'])))

classes_num = len(labels)

class_to_num = dict(zip(labels, range(classes_num)))
num_to_class = {v : k for k, v in class_to_num.items()}