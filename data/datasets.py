import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from random import sample,randint,random
from random import uniform
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
import torchvision
import os
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from .process import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize) # 随机剪裁，默认224
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img) # 不处理
    else:
        crop_func = transforms.CenterCrop(opt.cropSize) # 中心裁剪

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset



class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path




rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img, opt):
    width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    img = torchvision.transforms.Resize(
        (opt.loadSize, opt.loadSize), antialias=None
    )(img)
    return img


def pil_jpeg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img

def pil_webp(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='webp', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img

def pil_blur(img, radius):
    img = Image.fromarray(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    img = np.array(img)
    return img

def pil_gaussian_noise(img, std):
    img = np.array(img).astype(np.float32)                         
    noise = np.random.normal(0, std, img.shape).astype(np.float32) 
    img += noise                                                    
    img = np.clip(img, 0, 255)                                      
    return img.astype(np.uint8)                                   





def pil_brightness(img, factor):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    img = np.array(img)
    return img

def pil_saturation(img, factor):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(factor)
    img = np.array(img)
    return img

def pil_contrast(img, factor):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    img = np.array(img)
    return img

def pil_downup(img, scale=0.5):
    img = Image.fromarray(img)
    w, h = img.size
    down_w, down_h = int(w * scale), int(h * scale)
    
    img = img.resize((down_w, down_h), resample=Image.BICUBIC)
    
    img = img.resize((w, h), resample=Image.BICUBIC)
    return np.array(img)

def pil_resize_to(img, size):
    img = Image.fromarray(img)
    img = img.resize(size, resample=Image.BICUBIC)
    img = np.array(img)
    return img





def custom_augment(img, opt):
    
    # print('height, width:'+str(height)+str(width))
    # resize
    if opt.noise_type=='resize':
        
        height, width = img.height, img.width
        img = torchvision.transforms.Resize(
            (int(height / 2), int(width / 2)), antialias=None
        )(img)

    img = np.array(img)
    # img = img[0:-1:4,0:-1:4,:]

    # if opt.noise_type=='blur':
    #     sig = sample_continuous(opt.blur_sig)
    #     gaussian_blur(img, sig)

    if opt.noise_type=='jpg':
        
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)



    if opt.noise_type == 'jpeg':
        if random() < 0.5:
            pass
        else:
            quality = randint(opt.jpeg_quality_min, opt.jpeg_quality_max)
            img = pil_jpeg(img, quality)


    if opt.noise_type == 'jpeg75-95':
        quality = randint(75, 95)
        img = pil_jpeg(img, quality)


    if opt.noise_type == 'webp':
        if random() < 0.5:
        
            pass
        else:
            quality = randint(75, 95)
            img = pil_webp(img, quality)

    if opt.noise_type == 'blur':
        if random() < 0.5:
       
            pass
        else:
            radius = uniform(0.01, 2)  
            img = pil_blur(img, radius)

    if opt.noise_type == 'gaussian_noise':
        if random() < 0.5:
            pass
        else:
            std = uniform(0.05, 0.25)
            img = pil_gaussian_noise(img, std)

    if opt.noise_type == 'brightness':
        if random() < 0.5:
            pass
        else:
            factor = uniform(0.5, 2.0)  
            img = pil_brightness(img, factor)


    if opt.noise_type == 'saturation':
        if random() < 0.5:
            pass
        else:
            factor = uniform(0.5, 2.0)  
            img = pil_saturation(img, factor)

    if opt.noise_type == 'contrast':
        if random() < 0.5:
            pass
        else:
            factor = uniform(0.5, 2.0)  
            img = pil_contrast(img, factor)


    if opt.noise_type == 'resize_fixed':
        if random() < 0.5:
            pass
        else:
            s = randint(128, 512)  # 随机选一个边长
            img = pil_resize_to(img, (s, s))


    return Image.fromarray(img)


def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths



transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256], antialias=None),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def preprocess_artgate_image(img, opt):
    """Build the CLIP and frequency-branch inputs for one PIL image."""
    img = custom_augment(img.convert('RGB'), opt)
    freq_img = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])(img)
    clip_img = processing(img, opt, 'clip')
    return clip_img, freq_img


class read_data_artgate():
    def __init__(self, opt, max_real_size=None, max_fake_size=None):
        self.opt = opt
        self.root = opt.dataroot
        
        # 加载真实和假图片路径
        real_img_list = loadpathslist(self.root, '0_real')    
        fake_img_list = loadpathslist(self.root, '1_fake')

        # 如果指定了最大数量，从真实和假图片中随机选择
        if max_real_size:
            real_img_list = sample(real_img_list, min(len(real_img_list), max_real_size))
        if max_fake_size:
            fake_img_list = sample(fake_img_list, min(len(fake_img_list), max_fake_size))

        # 创建标签，真实图片标签为0，假图片标签为1
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_label_list = [1 for _ in range(len(fake_img_list))]

        # 合并真实和假图片及标签
        self.img = real_img_list + fake_img_list
        self.label = real_label_list + fake_label_list


    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        clip_img, freq_img = preprocess_artgate_image(img, self.opt)
        return clip_img, target, freq_img

    def __len__(self):
        return len(self.label)


    

