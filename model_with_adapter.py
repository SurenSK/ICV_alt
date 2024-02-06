import torch
from torch.nn import functional as F

def tokenize_each_demonstration(tok, demonstration_list, dataset_name=None):
    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        demonstration_list[exp_id] = (demonstration_list[exp_id][0].strip(" .").strip("."), demonstration_list[exp_id][1].strip(" .").strip("."))

        e_original = tok(demonstration_list[exp_id][0]) 
        e_rewrite = tok(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite)) 
    return tokenized_demonstration_list

class AdapterLayer(torch.nn.Module):
    def __init__(self, icvs, alpha):
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        input_dtype = x.dtype
        if self.icvs is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            alpha = self.alpha
            icv_all_tasks = 0
            for i in range(len(self.icvs)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), self.icvs[i][None,None,:], dim=-1)).unsqueeze(-1)
                icv_all_tasks -= alpha[i] * lambda_sim * F.normalize(self.icvs[i], dim=-1).repeat(1,x.shape[1],1)
            icv_all_tasks = 0.1 * icv_all_tasks/len(self.icvs)
            
            x = F.normalize(F.normalize(x.float(),dim=-1) +  icv_all_tasks, dim=-1) * norm
            return x.type(input_dtype)
        else:
            return x

class model_with_adapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, icvs, alpha):
        for i in range(0, len(self.model.transformer.h)):
            icvs_ = icvs[i]
            self.model.transformer.h[i].mlp = torch.nn.Sequential(self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha))
        return self.model

    def remove_adapter(self):
        weight_all = []
        for i in range(0, len(self.model.transformer.h)):
            weight_all.append(self.model.transformer.h[i].mlp[1].weight_all)
            self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        return weight_all
    
    def reset_adapter(self):
        if isinstance(self.model.transformer.h[0].mlp, torch.nn.modules.container.Sequential):
            for i in range(0, len(self.model.transformer.h)):
                self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        return self.model
    
    def set_adapter(self, icvs, alpha):
        self.model = self.reset_adapter()
        self.model = self.get_model(torch.stack(icvs,dim=1).cuda(), [alpha])
        return self.model