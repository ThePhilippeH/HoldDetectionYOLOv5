import json
import os
import torch
import torchvision
import shutil


def convert_json_to_yolo(json_file, output_dir):
  os.chdir(output_dir)
  labels_dir = 'labels'
  os.makedirs(labels_dir , exist_ok=True)

  with open(json_file) as f:
    data = json.load(f)
    images_metadata = data['_via_img_metadata']

    for key in images_metadata:
      image = images_metadata[key]
      filename = image['filename']
      regions = image['regions']
      

      if not regions:
        continue
      
      txt_filename = os.path.splitext(filename)[0] + '.txt'
      txt_filepath = os.path.join(labels_dir, txt_filename)
      print(filename)

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

def split_train_val_test(dir, train_ratio, val_ratio, test_ratio):

    images_dir = dir +'images'
    labels_dir = dir +'labels'
    
    
    torch.random.manual_seed(42)
    files = os.listdir(images_dir)

    train_size = int(train_ratio * len(files))
    test_size = int(len(files) - train_size)
    

    train_choose = torch.randperm(len(files))[:train_size]
    training_files = [files[i] for i in train_choose]
    test_files = [file for file in files if file not in training_files]


    # Create train and test folders
    train_folder_i = '/content/sm/images/train'
    test_folder_i = '/content/sm/images/valid'
    train_folder_l = '/content/sm/labels/train'
    test_folder_l = '/content/sm/labels/valid'
    os.mkdir(train_folder_i, exist_ok=True)
    os.mkdir(test_folder_i, exist_ok=True)
    os.mkdir(train_folder_l, exist_ok=True)
    os.mkdir(test_folder_l, exist_ok=True)

    # Move images and labels to train and test folders
    for file in training_files:
        os.rename(os.path.join(images_dir, file), os.path.join(train_folder_i, file))
        os.rename(os.path.join(labels_dir, file.replace('.jpg', '.txt')), os.path.join(train_folder_l, file.replace('.jpg', '.txt')))

    for file in test_files:
        os.rename(os.path.join(images_dir, file), os.path.join(test_folder_i, file))
        os.rename(os.path.join(labels_dir, file.replace('.jpg', '.txt')), os.path.join(test_folder_l, file.replace('.jpg', '.txt')))

    print(f'Training set size: {len(training_files)}')
    print(f'Test set size: {len(test_files)}')

def move_images_to_folder(dir):
  os.chdir(dir)
  if os.path.exists('images'):
    return
  os.mkdir('images')
  image_files = [file for file in os.listdir() if file.endswith('.jpg')]
  for file in image_files:
    shutil.move(file, os.path.join(dir, 'images', file))
