class Config_ML10:
    def __init__(self):
        self.benchmark_name= 'ML10' 
        self.env_name= 'door-close-v2' 

        self.shaped_rewards_available_at_train_time=True
        if self.shaped_rewards_available_at_train_time==True:
            self.sparse_or_shaped_training_target='shaped'
        elif self.shaped_rewards_available_at_train_time==False:
            self.sparse_or_shaped_training_target='sparse'

        self.shaped_rewards_available_at_test_time=False
        if self.shaped_rewards_available_at_test_time==True:
            self.sparse_or_shaped_inputs='shaped' 
        elif self.shaped_rewards_available_at_test_time==False:
            self.sparse_or_shaped_inputs='sparse' 


        self.num_epsiodes_of_validation = 2
        self.num_lifetimes_for_validation = 90

        self.seeding=False
        self.seed=1
        self.ol_device='cuda'
        self.il_device='cpu'

        self.num_outer_loop_updates=5000 
        self.num_inner_loops_per_update = 30
        self.num_il_lifetime_steps=4500  
        

        self.learning_rate=5e-4
        self.anneal_lr= False
        self.adam_eps=1e-5

        self.e_rewards_target_mean= 0.1 
        self.meta_gamma= 0.995 
        self.ae={
            "estimation_method": 'standard GAE',   
            "bootstrapping_lambda" : 0.95,
            "starting_n" : 1000 ,#ignored when using GAE as estimation method
            "num_n_step_estimates" : 10 ,#ignored when using GAE as estimation method
            "skip_rate" : 200 #ignored when using GAE as estimation method
        }

        self.rnn_input_size  = 32 
        self.rnn_type= 'lstm'              
        self.rnn_hidden_state_size= 256 
        self.initial_std=1.0

        self.ppo={
            "k" : 400,
            'update_epochs' : 10, 
            'num_minibatches': 0,
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.0,
            "valuef_coef": 0.5,
            "clip_grad_norm": True, 
            "max_grad_norm": 0.5,
            "target_KL": 0.1
        }

class Config_ML1:
    def __init__(self):
        self.benchmark_name= 'ML1' 
        self.env_name= 'button-press-v2' 

        self.shaped_rewards_available_at_train_time=True
        if self.shaped_rewards_available_at_train_time==True:
            self.sparse_or_shaped_training_target='shaped'
        elif self.shaped_rewards_available_at_train_time==False:
            self.sparse_or_shaped_training_target='sparse'

        self.shaped_rewards_available_at_test_time=False
        if self.shaped_rewards_available_at_test_time==True:
            self.sparse_or_shaped_inputs='shaped' 
        elif self.shaped_rewards_available_at_test_time==False:
            self.sparse_or_shaped_inputs='sparse'


        self.num_epsiodes_of_validation = 2
        self.num_lifetimes_for_validation = 60

        self.seeding=False
        self.seed=1
        self.ol_device='cuda'
        self.il_device='cpu'

        self.num_outer_loop_updates=5000 
        self.num_inner_loops_per_update = 30
        self.num_il_lifetime_steps=4500 
        

        self.learning_rate=5e-4
        self.anneal_lr= False
        self.adam_eps=1e-5

        self.e_rewards_target_mean= 0.1 
        self.meta_gamma= 0.995 
        self.ae={
            "estimation_method": 'standard GAE',   
            "bootstrapping_lambda" : 0.95,
            "starting_n" : 1000 ,#ignored when using GAE as estimation method
            "num_n_step_estimates" : 10 ,#ignored when using GAE as estimation method
            "skip_rate" : 200 #ignored when using GAE as estimation method
        }

        self.rnn_input_size  = 32 
        self.rnn_type= 'lstm'              
        self.rnn_hidden_state_size= 256 
        self.initial_std=1.0

        self.ppo={
            "k" : 400,
            'update_epochs' : 10, 
            'num_minibatches': 0,
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.005,
            "valuef_coef": 0.5,
            "clip_grad_norm": True, 
            "max_grad_norm": 0.5,
            "target_KL": 0.1
        }





def get_config(config_settings):
    if config_settings=='ml10':
        return Config_ML10()
    elif config_settings=='ml1':
        return Config_ML1()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    
