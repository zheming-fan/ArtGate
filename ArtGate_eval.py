from ArtGate_model import ArtGate_CLIP
import os
import csv
import torch
import numpy as np
from validate import validate_artgate
from options import TestOptions
from eval_config import *
from PIL import ImageFile
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

set_random_seed()
# Running tests


opt = TestOptions().parse(print_options=True) 



model_name = os.path.basename(opt.model_path).replace('.pth', '')
results_dir=f"./results/{opt.detect_method}"
mkdir(results_dir)

rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision', 'r_acc', 'f_acc']]

print("{} model testing on...".format(model_name))

all_metrics = []  

for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)


    model = ArtGate_CLIP(name="/home/ubuntu/data/zhemingfan/mllm/openai-clip-vit-large-patch14", num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=True)
    model.cuda()
    model.eval()


    
    try:
        if 'model' in state_dict:
            model.load_state_dict(state_dict['model'], strict=True)
        else:
            model.load_state_dict(state_dict, strict=True)
    except:
        print("[ERROR] model.load_state_dict() error")
    model.cuda()
    model.eval()


    opt.process_device = torch.device("cpu")
    f1, auc, t10, t1, acc, ap, r_acc, f_acc, _, _ ,eer= validate_artgate(model, opt, max_real_size=opt.max_test_image, max_fake_size=opt.max_test_image)

    row = [val, acc, ap, r_acc, f_acc, f1, auc, t10, t1,eer]
    rows.append(row)
    all_metrics.append(row[1:]) 
   
    print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {}; f1: {}; auc: {}; T10: {}; T1: {}; EER: {}".format(
        val, acc, ap, r_acc, f_acc, f1, auc, t10, t1,eer))
    

mean_metrics = np.mean(np.array(all_metrics, dtype=np.float32), axis=0)
rows.append(["Average"] + list(mean_metrics))


print("Average metrics:")
print("acc: {:.4f}; ap: {:.4f}; r_acc: {:.4f}; f_acc: {:.4f}; f1: {:.4f}; auc: {:.4f}; T10: {:.4f}; T1: {:.4f}; EER: {:.4f}".format(
    *mean_metrics))



csv_name = results_dir + '/{}_{}.csv'.format(opt.detect_method, opt.noise_type)
with open(csv_name, 'a+') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
