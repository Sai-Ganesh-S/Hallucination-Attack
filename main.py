from attacker import Attacker
from utils import random_init
import pandas as pd


model_name = 'vicuna' ### [vicuna, llama2, baichuan, internlm, chatglm, ziya]

### 1. OoD Attack (initialized from random tokens)
#init_input = random_init(model_name, length=20)

### 2. Weak Semantic Attack (initialized from the raw sentence)
def attack_cycle(ques,ans):
    init_input = ques
    target = ans
    mini_batch_size = 30 ### If CUDA out of memory, lower the mini_batch_size
    batch_size = 900
    device = 'cuda:0'
    steps = 100
    # topk = 256

    attacker_params = {
        'update_strategy': 'gaussian',
        'early_stop': True,
        # 'is_save': True,
        'save_dir': './result',
    }


    
    attacker = Attacker(
        model_name,
        init_input,
        target,
        device=device,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size,
        **attacker_params
    )
    attacker.run()

ds = pd.read_csv("/content/Hallucination-Attack/QA.csv")
qs = ds ['qs']
ans = ds['ans']

for x in range(len(qs)):
    print(qs[x], ans[x])
    attack_cycle(qs[x],ans[x])
