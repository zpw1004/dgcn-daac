import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    random.seed(seed)                            
    np.random.seed(seed)                         
    torch.manual_seed(seed)                      
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)          
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      

    os.environ['PYTHONHASHSEED'] = str(seed)    

# Set seed at the beginning
set_seed(42)
