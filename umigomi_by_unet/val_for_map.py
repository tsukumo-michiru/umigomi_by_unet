import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
import albumentations as al
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset_for_map import Dataset
from metrics import iou_score
from utils import AverageMeter
from preprocessing import Preprocessing


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main(filename:str) -> None:
    args = parse_args() 

    with open('models/%s/config.yml' % args.name, 'r+') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['dataset'] = filename

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['dataset'], str(c)), exist_ok=True)

    imgs = glob(os.path.join('inputs', filename, 'images', '*'+ config['img_ext']))
    output_path = os.path.join('inputs', config['dataset'], 'images','cut')
    preprocessing = Preprocessing(imgs, output_path, config['input_h'], config['input_w'])
    preprocessing.cut()

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

   
    # Data loading code
    #img_ids = glob(os.path.join('fukui_dataset', config['dataset'], '*' + config['img_ext']))
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images','cut', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
   
    #_, val_img_ids = train_test_split(img_ids, test_size=0.2, shuffle=False)
    val_img_ids = img_ids
    # val_img_ids = glob(os.path.join('inputs', config['dataset'], 'val_images', '*' + config['img_ext']))

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        al.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images', 'cut'),
        img_ext=config['img_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()
    
    """
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['dataset'], str(c)), exist_ok=True)
    """ 
    
    
    with torch.no_grad():
        for inputs, meta in tqdm(val_loader, total=len(val_loader)):
            inputs = inputs.cuda()
            #target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(inputs)[-1]
            else:
                output = model(inputs)

            #iou = iou_score(output, target)
            #avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    
                    cv2.imwrite(os.path.join('outputs', config['dataset'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.3f' % avg_meter.avg)

    torch.cuda.empty_cache()
    print("\n")
    preprocessing.connect(config['dataset'])
    preprocessing.toka(config['dataset'])
    preprocessing.exif(config['dataset'])


if __name__ == '__main__':
    switch = True
    while switch:
        filename = input("inputデータのファイル名を入力してください : User/AI/unet++/inputs/")
        if os.path.isdir(os.path.join('inputs', filename, 'images')):
            switch = False
    main(filename)
