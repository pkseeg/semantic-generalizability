import torch.nn.functional as F
import torch
import numpy as np
from tte_depth import StatDepth

device = "cuda" if torch.cuda.is_available() else "cpu"

def transform_embeds(a_, b_):
    a_ = torch.tensor(a_["embedding"]).float()
    b_ = torch.tensor(b_["embedding"]).float()

    a_ = a_.to(device)
    b_ = b_.to(device)
    return a_, b_

# I need the citation here
def info_gain(a_, b_):
    a_, b_ = transform_embeds(a_, b_)
    distances = torch.cdist(b_.unsqueeze(0), a_.unsqueeze(0)).squeeze(0)
    min_distances = distances.min(dim=1).values
    return min_distances.mean()

def depth(a_, b_, batch_size = 32):
    
    

    F, G = np.array(a_), np.array(b_)


    depth = StatDepth()
    return depth.depth_rank_test(F, G)

    # similarities = []
    # for i in range(0, len(a_), batch_size):
    #     a_batch = a_[i : i + batch_size]
    #     sim = F.cosine_similarity(
    #         a_batch.unsqueeze(1), b_.unsqueeze(0), dim=-1
    #     )  
    #     similarities.append(sim.mean(dim=1).cpu())
    
    # avg_similarity = torch.cat(similarities).mean().item()
    # return avg_similarity