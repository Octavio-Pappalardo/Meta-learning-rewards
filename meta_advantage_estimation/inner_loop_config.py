

class Config_metaworld:
    def __init__(self):

        self.meta_device='cpu'
        self.il_device='cpu'
        self.seeding=False
        self.seed=1

        self.total_timesteps=6000 
        self.num_steps= 2000  
        self.num_updates = self.total_timesteps // self.num_steps 
        self.num_lifetime_steps= int(self.num_updates*self.num_steps)

        self.learning_rate=3e-4
        self.anneal_lr= False
        self.adam_eps=1e-5  

        self.gamma=0.99 
        self.gae= True
        self.gae_lambda=0.95


        self.ppo={
            'update_epochs' : 64 ,
            'num_minibatches': 16, 
            "normalize_advantage": True, 
            "clip_coef": 0.2,  
            "entropy_coef": 0.0,  
            "clip_grad_norm":True,
            "max_grad_norm": 0.5,
            "target_KL": None
        }


#config to try things out 
class Config_evaluation:
    def __init__(self):

        self.meta_device='cpu'
        self.il_device='cpu'
        self.seeding=False
        self.seed=1

        self.total_timesteps=20000 #10000
        self.num_steps= 1000  #1024
        self.num_updates = self.total_timesteps // self.num_steps ##4000 or 500?
        self.num_lifetime_steps= int(self.num_updates*self.num_steps)

        self.learning_rate=3e-4
        self.anneal_lr= False
        self.adam_eps=1e-5  

        self.gamma=0.99 #!0.99
        self.gae= True
        self.gae_lambda=0.95##  #!0.95


        self.ppo={
            'update_epochs' : 16 ,#,16,  #use 64 for best performance #using 32,32 works similarly well
            'num_minibatches': 16, 
            "normalize_advantage": True, ##
            "clip_coef": 0.2,  #0.25 works worse
            "clip_vloss": False,
            "entropy_coef": 0.000,  ##
            "valuef_coef": 0.5,
            "clip_grad_norm":True,
            "max_grad_norm": 0.5,
            "target_KL": None
        }



def get_config(config_settings):
    if config_settings=='metaworld':
        return Config_metaworld()
    elif config_settings=='evaluation':
        return Config_evaluation()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    
