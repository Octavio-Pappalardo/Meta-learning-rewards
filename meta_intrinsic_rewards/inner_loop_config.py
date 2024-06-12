
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
            "clip_vloss": False,
            "entropy_coef": 0.0,  
            "valuef_coef": 0.5,
            "clip_grad_norm":True,
            "max_grad_norm": 0.5,
            "target_KL": None
        }



def get_config(config_settings):
    if config_settings=='metaworld':
        return Config_metaworld()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    



#INNER LOOP CONFIGURATIONS AND HYPERPARAMETERS

#meta_device: controls on what device the meta_agent runs while making predictions for the inner loop
#il_device:    controls on what device the inner loop agent runs while training
  
#num_steps  :  #number of steps the action agent takes before updating 
#num_updates = total_timesteps // num_steps :   number of times action agent is updated ( !it is actually updated num_updates-1 times)
#num_lifetime_steps= int(num_updates*num_steps) :  number of steps collected per environment during the lifetime(whole training)


#learning_rate: optimizers learning rate
#anneal_lr  :  Toggle learning rate annealing for policy and value networks gradient descent updates



#gamma  : #the gamma considered for the task
#gae   : wether to Use GAE for advantage computation
#gae_lambda : the lambda for the general advantage estimation

#PPO parameters
#update_epochs  :        #number of epochs runned inside the PPO update
#num_minibatches    :      the number of mini-batches created from batch for the PPO update
#normalize_advantage    :  Toggles advantages normalization - if True the advantage estimators are standarized within each minibatch
#clip_coef  :              the surrogate clipping coefficient - controls how big the update by each minibatch inside the ppo update can be.
#clip_vloss :          toggles whether or not to use a clipped loss for the value function
#entropy_coef   :     coefficient of the entropy - controls the weight of the exploration encouraging entropy loss in the total loss with wich agent is updated.
#valuef_coef    :     coefficient of the value function - controls the weight of the value function loss in the total loss with wich agent is updated.
#max_grad_norm  :     the maximum norm the gradient is allowed to have. Parameter for gradient clipping
#target_KL  :      the target KL divergence threshold

