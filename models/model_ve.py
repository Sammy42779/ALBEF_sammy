from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Compute the softmax of the logits
        probs = F.softmax(logits, dim=1)

        # Get the probabilities of the true class labels
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).float()
        true_probs = torch.sum(probs * targets_one_hot, dim=1)

        # Compute the focal loss
        focal_weight = (1 - true_probs).pow(self.gamma)
        focal_loss = -1 * self.alpha * focal_weight * torch.log(true_probs)

        return focal_loss.mean()

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), p_shuffle=False)    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          

        '''three-way classification problem'''
        ## predict the class probabilities using a multi-layer perceptron (MLP) on the multimodal encoder’s representation of the [CLS] token
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 3)
                )

        embed_dim = 256        
        vision_width = 768
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)   
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, 3)
                    )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            
            
    def forward(self, image, text, targets, alpha=0, train=True, p_shuffle=False, weight=False, loss_type=False, lam=0.5):
        
        image_embeds = self.visual_encoder(image, p_shuffle=p_shuffle) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)    

        ## D: to get cls token for computing weights
        image_cls = self.vision_proj(image_embeds[:,0,:].detach()).detach()  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, mode='text')
        text_embeds = text_output.last_hidden_state.detach()   
        text_cls = F.normalize(self.text_proj(text_embeds[:,0,:])).detach()
        
        if train:
            ## multimodal encoder (using image_embeds)
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            ## cls_head: using the [CLS] token (output.last_hidden_state[:,0,:])
            ## using a multi-layer perceptron (MLP)
            ## 3-way classification   
            ## output: class probabilities
            prediction = self.cls_head(output.last_hidden_state[:,0,:])     

            ## D: to assign weights
            if weight == 'label_assign':
                label_weights = torch.tensor([5 if x == 2 else 2 if x == 1 else 1 for x in targets]).to(image.device)
                weights = label_weights

            elif weight == 'l2':
                l2_loss = F.pairwise_distance(image_cls, text_cls, p=2)
                # 计算权重
                l2_weight = 1.0 / torch.pow(l2_loss, 2)
                # 对权重进行归一化
                l2_weight = l2_weight / torch.sum(l2_weight)
                weights = l2_weight

            elif weight == 'kl':
                kl_loss = F.kl_div(image_cls.log_softmax(dim=-1), text_cls.softmax(dim=-1), reduction='none').sum(dim=-1)
                kl_weight = 1.0 / torch.pow(kl_loss, 2)
                # 对权重进行归一化
                kl_weight = kl_weight / torch.sum(kl_weight)
                weights = kl_weight

            elif weight == 'gair_l2':
                l2_loss = F.pairwise_distance(image_cls, text_cls, p=2)
                reweight = ((-1.0+(int(10/2)-l2_loss)*5/(int(10/2))).tanh()+1)/2
                gair_l2_weight = reweight * len(reweight) / reweight.sum()
                weights = gair_l2_weight

            elif weight == 'gair_kl':
                kl_loss = F.kl_div(image_cls.log_softmax(dim=-1), text_cls.softmax(dim=-1), reduction='none').sum(dim=-1)
                reweight = ((-1.0+(int(10/2)-kl_loss)*5/(int(10/2))).tanh()+1)/2
                gair_kl_weight = reweight * len(reweight) / reweight.sum()
                weights = gair_kl_weight

            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = image_embeds_m,
                                               encoder_attention_mask = image_atts,        
                                               return_dict = True
                                              )           
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   
                
                if loss_type == 'focal_loss':
                    fl = self.focal_loss(prediction, targets)
                    loss = (1-alpha)*fl - alpha*torch.sum(
                        F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
                elif loss_type == 'focal_loss_reg':
                    loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                        F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
                    loss = loss + self.focal_loss(prediction, targets) ## as a regularization
                elif loss_type == 'mix_img_text':
                    mixed_cls = lam * image_cls + (1-lam) * text_cls
                    output_cls = output.last_hidden_state[:,0,:]
                    kl_mixed_cls_fusion_loss = F.kl_div(mixed_cls.log_softmax(dim=-1), output_cls.softmax(dim=-1)).sum(dim=-1)
                    loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                        F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean() + kl_mixed_cls_fusion_loss
                else:
                    loss = F.cross_entropy(prediction, targets, reduction='none')
                    loss = loss.mul(weights).mean()
                    loss = (1-alpha)*loss - alpha*torch.sum(
                        F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
                    # loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    #     F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                ## cross-entropy for classification problem
                loss = F.cross_entropy(prediction, targets)    

            return loss 
            
        else:
            '''inference'''
            ## output the probabilites instead of computing loss
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            prediction = self.cls_head(output.last_hidden_state[:,0,:])   # CLS token: shape=[bs, 768]                   
            return prediction
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

