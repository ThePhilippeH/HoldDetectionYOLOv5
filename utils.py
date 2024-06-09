import json
import os
import torch
import torchvision


def convert_json_to_yolo(json_file, output_dir):
  labels_dir = output_dir + 'labels'
  os.makedirs(labels_dir , exist_ok=True)

  with open(json_file) as f:
    data = json.load(f)

    for key in data:
      image = data[key]
      filename = image['filename']
      regions = image['regions']

      if not regions:
        continue
      
      txt_filename = os.path.splitext(filename)[0] + '.txt'
      txt_filepath = os.path.join(labels_dir, txt_filename)

      with open(txt_filepath, 'w') as txt_file:
        for region in regions:
          shape_attributes = region['shape_attributes']
          region_attributes = region['region_attributes']
          class_name = region_attributes['hold_type']

          all_points_x = shape_attributes['all_points_x']
          all_points_y = shape_attributes['all_points_y']

          width = max(all_points_x) - min(all_points_x)
          height = max(all_points_y) - min(all_points_y)
          center_x = min(all_points_x) + width / 2
          center_y = min(all_points_y) + height / 2

          label = f'{class_name} {center_x} {center_y} {width} {height}\n'
    txt_file.write(label)

def split_train_val_test(json_file):

    images_dir = '/content/sm/images'
    labels_dir = '/content/sm/labels'
    train_ratio = 0.8
    test_ratio = 0.2
    
    
    torch.random.manual_seed(42)
    files = os.listdir(images_dir)

    train_size = int(train_ratio * len(files))
    test_size = int(len(files) - train_size)
    

    train_choose = torch.randperm(len(files))[:train_size]
    training_files = [files[i] for i in train_choose]
    test_files = [file for file in files if file not in training_files]


    for file in training_files:
      


