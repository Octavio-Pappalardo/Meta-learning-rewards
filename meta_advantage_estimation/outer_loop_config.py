


class Config_metaworld_ML1:
    def __init__(self):
        self.benchmark_name= 'ML1' 
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

        self.num_epsiodes_of_validation = 4
        self.num_lifetimes_for_validation = 60

        self.seeding=False
        self.seed=1
        self.device='cuda'

        self.num_outer_loop_updates= 500 
        self.num_inner_loops_per_update = 30 

        self.learning_rate= 5e-5 
        self.anneal_lr= False
        self.adam_eps=1e-5


        self.e_rewards_target_mean= 0.0001 
        self.meta_gamma=0.9 
        self.ae={
            "estimation_method": 'bootstrapping skipping uninfluenced future rewards',  
            "bootstrapping_lambda" : 0.85,
            "starting_n" : 2200 ,
            "num_n_step_estimates" : 6 ,
            "skip_rate" : 300 
        }


        self.rnn_input_size  = 32 
        self.rnn_type= 'lstm'              
        self.rnn_hidden_state_size= 128 
        self.initial_std=1.0

        self.ppo={
            "k" : 400,
            'update_epochs' : 12,
            'num_minibatches': 0, 
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.0005,
            "valuef_coef": 0.5,
            "clip_grad_norm": True,
            "max_grad_norm": 0.5,
            "target_KL": 0.01
        }


class Config_metaworld_ML10:
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

        self.num_epsiodes_of_validation = 4
        self.num_lifetimes_for_validation = 60

        self.seeding=False
        self.seed=1
        self.device='cuda'

        self.num_outer_loop_updates= 1500 
        self.num_inner_loops_per_update = 30 

        self.learning_rate= 5e-5 
        self.anneal_lr= False
        self.adam_eps=1e-5


        self.e_rewards_target_mean= 0.0001 
        self.meta_gamma=0.9 
        self.ae={
            "estimation_method": 'bootstrapping skipping uninfluenced future rewards',  
            "bootstrapping_lambda" : 0.85,
            "starting_n" : 2200 ,
            "num_n_step_estimates" : 6 ,
            "skip_rate" : 300 
        }


        self.rnn_input_size  = 32 
        self.rnn_type= 'lstm'              
        self.rnn_hidden_state_size= 128 
        self.initial_std=1.0

        self.ppo={
            "k" : 400,
            'update_epochs' : 12,
            'num_minibatches': 0, 
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.0005,
            "valuef_coef": 0.5,
            "clip_grad_norm": True,
            "max_grad_norm": 0.5,
            "target_KL": 0.01
        }




def get_ol_config(config_settings):
    if config_settings=='ml10':
        return Config_metaworld_ML10()
    elif config_settings=='ml1':
        return Config_metaworld_ML1()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    
