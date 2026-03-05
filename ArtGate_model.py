import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.nn import functional as F
import torch
from transformers import CLIPModel
from artifact_branch import freq_resnet50
from peft import  get_peft_model, LoraConfig






class ArtGate_CLIP(nn.Module):
    def __init__(self, name='/home/ubuntu/data/zhemingfan/mllm/openai-clip-vit-large-patch14', num_classes=1):
        super(ArtGate_CLIP, self).__init__()

        self.model = CLIPModel.from_pretrained(name)
        clip_state_dict = torch.load('/home/ubuntu/2026/ArtGate/weights/model_clip_progan.pth', map_location='cpu')
        self.model.load_state_dict(clip_state_dict, strict=True)  # 加载模型参数
  
        self.resnetmodel = freq_resnet50()
        state_dict = torch.load('/home/ubuntu/2026/ArtGate/weights/freq_progan.pth', map_location='cpu', weights_only=False)
        self.resnetmodel.load_state_dict(state_dict["model"], strict=True)  # 加载模型参数

        self.art_token_long=32
        self.fc0= nn.Linear(512, self.art_token_long * 1024) 
        self.fc = nn.Linear(768, num_classes)
        self.logit_scale=self.model.logit_scale



        lora_config = LoraConfig(
            init_lora_weights="pissa_niter_4",
            r=8,  
            lora_alpha=16,  
            lora_dropout=0.1,  

             target_modules=[
            "vision_model.encoder.layers.23.self_attn.k_proj",
            "vision_model.encoder.layers.23.self_attn.v_proj",
            "vision_model.encoder.layers.23.self_attn.q_proj",
            "vision_model.encoder.layers.23.self_attn.out_proj",
            "vision_model.encoder.layers.23.mlp.fc1",
            "vision_model.encoder.layers.23.mlp.fc2",
            "visual_projection",
            "text_projection"
    ],  # 明确列出所有目标模块
            bias="none",  # 如何处理目标模块的偏置
            task_type="FEATURE_EXTRACTION"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.lora_parameters = [p for n, p in self.model.named_parameters() if "lora" in n]

    def encode_image(self, img, art_img):

        
        artifact_feature   = self.resnetmodel.get_features(art_img)
        artifact_token = self.fc0(artifact_feature)
        artifact_token = artifact_token.view(artifact_token.size(0), self.art_token_long, 1024) 


        artifact_logits = self.resnetmodel(art_img)
        artifact_logits_flattened = artifact_logits.view(-1)  
        artifact_logits_sigmoid = torch.sigmoid(artifact_logits_flattened)


        if (artifact_logits_sigmoid > 0.5):
            artifact_token = artifact_token
        else:
            artifact_token = None

        vision_outputs = self.model.vision_model(
            pixel_values=img,
            output_attentions    = self.model.config.output_attentions,
            output_hidden_states = self.model.config.output_hidden_states,
            return_dict          = self.model.config.use_return_dict,   
            artifact_token=artifact_token   
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooled_output)
        return image_features    
    
    
    def forward(self, img, img_fa):

        image_embeds = self.encode_image(img,img_fa)

        image_embeds_norm = F.normalize(image_embeds, p=2, dim=-1)

        return self.fc(image_embeds_norm)
