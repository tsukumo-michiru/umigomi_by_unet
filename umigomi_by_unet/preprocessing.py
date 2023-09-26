from fileinput import filename
from itertools import count
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from natsort import natsorted
from PIL import Image, ExifTags
import piexif

class Preprocessing():    
    def __init__(self,imgs,output_path,height,width):
        self.imgs = imgs
        self.output_path = output_path
        self.height = height
        self.width = width

    def prepare_for_cut(self, name, img):
        h, w = img.shape[:2]
        y0 = self.height
        x0 = self.width
        h_count = int(h/self.height) 
        w_count = int(w/self.width)   
        # 分割した画像を内包表記でリスト化
        splited_imgs = [img[y0*y:y0*(y+1), x0*x:x0*(x+1)] for x in range(w_count) for y in range(h_count)]
        

        # 分割された画像をダウンロード
        for i,fig in enumerate(splited_imgs):
            if np.any(fig==[255,255,255]):
                fig[:,:,:] = [0,0,0]
                cv2.imwrite(os.path.join('outputs',filename,'0', os.path.splitext(os.path.basename(name))[0]+'--{}.png'.format(i+1)),fig)
            else:
                cv2.imwrite(os.path.join(self.output_path, os.path.splitext(os.path.basename(name))[0]+'--{}.JPG'.format(i+1)), fig)
        
    def cut(self):
            os.makedirs(self.output_path, exist_ok=True)
            print("画像分割が始まりました")
            for img in tqdm(self.imgs,total=len(self.imgs)):
                img_fig  = cv2.imread(img)
                self.prepare_for_cut(img, img_fig)
        
            print("画像分割が終わりました\n")
            
                
        

    def connect(self, filename):
        print("画像結合が始まりました")
        imgs = natsorted(glob.glob(os.path.join('outputs',filename,'0','*.png')))
        img= cv2.imread(self.imgs[0])
        h, w = img.shape[:2]
        h_count = int(h/self.height) 
        w_count = int(w/self.width)
        cn = 0
        count = int(len(imgs)/int(w_count*h_count))
        
        os.makedirs(os.path.join('detect',filename), exist_ok=True)

        for c in tqdm(range(count)):
            horizontal_connect_list = []
            for i in range(w_count):
                vertical_connect_list = []
                for j in range(h_count):
                    img = cv2.imread(imgs[cn])
                    vertical_connect_list.append(img)
                    cn += 1
                vertical_imgs = cv2.vconcat(vertical_connect_list)
                horizontal_connect_list.append(vertical_imgs)

            connected_img = cv2.hconcat(horizontal_connect_list)
            cv2.imwrite(os.path.join('./detect',filename,os.path.splitext(os.path.basename(imgs[cn-2166]))[0].replace('--0','')+'.png'), connected_img)
        print("画像結合が終わりました\n")

    def exif(self, filename):
        print("地理データを取得しています")
        detects = sorted(glob.glob(os.path.join('detect',filename,'toka','*'+'JPG')))
        for detect, img in zip(tqdm(detects),  self.imgs):
            img = Image.open(img)
            det = Image.open(detect)
            exif_dict = piexif.load(img.info["exif"])
            exif_bytes = piexif.dump(exif_dict)
            det.save(detect, exif=exif_bytes)
            img.close()
        print("地理データを取得し終えました\n")

    def toka(self, filename):
        print("透過処理が始まりました")
        os.makedirs(os.path.join('detect',filename,'toka','JPG'), exist_ok=True)
        os.makedirs(os.path.join('detect',filename,'toka','TIFF'), exist_ok=True)
        
        detects = sorted(glob.glob(os.path.join('detect',filename,'*png')))

        for dt, im in zip(tqdm(detects),self.imgs):
            detect = cv2.imread(dt)
            img = cv2.imread(im)
            # 透過描画したい（マスク）領
            detected = np.where((detect > [100,100,100]).all(axis=2))
    
            # 描画元を赤に塗りつぶす
            img[detected] = [0,0,255]
                
            #cv2.imwrite(os.path.join('detect',filename,'toka','JPG', os.path.splitext(os.path.basename(im))[0]+'.JPG'), img)
            cv2.imwrite(os.path.join('detect',filename,'toka','TIFF', os.path.splitext(os.path.basename(im))[0]+'.tiff'), img)
        print("透過処理が終わりました\n")
    
    