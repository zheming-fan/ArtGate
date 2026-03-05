import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score,f1_score,roc_auc_score,roc_curve
from data import create_dataloader_test_artgate







def tpr_at_fpr(y_true, y_scores, target_fpr=0.1):
  
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if target_fpr >= fpr[-1]:
        return tpr[-1]
    
    tpr_interp = np.interp(target_fpr, fpr, tpr)
    return tpr_interp


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # Find the threshold where fpr ~= fnr
    eer_threshold_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    return eer





def validate_artgate(model, opt, max_real_size=None, max_fake_size=None):
    
    opt = opt
    
    data_loader = create_dataloader_test_artgate(opt, max_real_size=max_real_size, max_fake_size=max_fake_size)
    y_true, y_pred = [], []

        # with torch.no_grad():
    i = 0
    for img, label,freq_img in data_loader:
        i += 1
        print("batch number {}/{}".format(i, len(data_loader)), end='\r')
        in_tens = img.cuda()
        freq_img = freq_img.cuda()
  
        if opt.fc_class2:
            probs = torch.softmax(model(in_tens,freq_img), dim=1)[:, 1]  # 取类别 1 的概率
        elif opt.detect_method == "EFFORT_sd":
 
            probs = model(in_tens, inference=True)   
            probs = probs["prob"]

        else:
            probs = model(in_tens,freq_img).sigmoid() 

        y_pred.extend(probs.flatten().tolist())

        y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred > 0.5)
    auc = roc_auc_score(y_true, y_pred)
    t10 = tpr_at_fpr(y_true,  y_pred, target_fpr=0.1)
    t1 = tpr_at_fpr(y_true,  y_pred, target_fpr=0.01)
    eer = compute_eer(y_true, y_pred)  
    return f1,auc,t10,t1,acc, ap, r_acc, f_acc, y_true, y_pred,eer


