
class Metaworld:
    def __init__(self):
        self.benchmark_name= 'ML1' 
        self.env_name= 'reach-v2' 
        self.intrinsic_rewards= False

        self.shaped_rewards_available_at_train_time=True
        if self.shaped_rewards_available_at_train_time==True:
            self.sparse_or_shaped_training_target='shaped'
        elif self.shaped_rewards_available_at_train_time==False:
            self.sparse_or_shaped_training_target='sparse'

        self.shaped_rewards_available_at_test_time= False
        if self.shaped_rewards_available_at_test_time==True:
            self.sparse_or_shaped_inputs='shaped' 
        elif self.shaped_rewards_available_at_test_time==False:
            self.sparse_or_shaped_inputs='sparse' 

        self.num_epsiodes_of_validation = 4
        self.num_lifetimes_for_validation = 120

        self.seeding=False
        self.seed=1
        self.device='cpu'
        self.device_rew_agent_update='cpu'

        self.num_outer_loop_updates=4000
        self.num_inner_loops_per_update=30

        self.num_adaptation_updates_in_inner_loop=2 
        self.num_env_steps_per_adaptation_update=2000 
        self.num_env_steps_for_estimating_maml_loss=4000
        self.num_lifetime_steps=self.num_adaptation_updates_in_inner_loop * self.num_env_steps_per_adaptation_update + self.num_env_steps_for_estimating_maml_loss

        self.rew_agent_lr=5e-5 
        self.rew_agent_epsilon=1e-5

        self.maml_agent_lr=5e-4 
        self.maml_agent_epsilon=1e-5

        self.adaptation_lr= 7e-2 
        self.adaptation_gamma=0.995 

        #normalization
        self.e_rewards_target_mean_for_reward_agent= 0.0001 
        self.e_rewards_target_mean_for_maml_agent=  0.1 


         
        self.rew_agent_gamma=0.9
        self.rew_agent_ae={
            "estimation_method": 'bootstrapping skipping uninfluenced future rewards',
            "bootstrapping_lambda" : 0.85 ,
            "starting_n" : 2200, 
            "num_n_step_estimates" : 6 ,
            "skip_rate" : 300
        }


        self.maml_agent_gamma=0.995 

        #reward agent network parameters
        self.rnn_input_size  = 32 
        self.rnn_type= 'lstm'              
        self.rnn_hidden_state_size= 128 
        self.initial_std=0.2

        #outer loop updates parameters
        self.rew_agent_ppo={
            "k" : 2000,
            'update_epochs' : 12,
            'num_minibatches': 0,
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.0000,
            "valuef_coef": 0.5,
            "clip_grad_norm": True, 
            "max_grad_norm": 0.5,
            "target_KL": 0.01
        }

        self.maml_TRPO={
            "cg_damping": 1e-2 ,
            "max_kl" :  0.001,
            "cg_iters" : 10,
            "line_search_max_steps": 10,
            "line_search_backtrack_ratio": 0.6
        } 



def get_config(config_settings):
    if config_settings=='metaworld':
        return Metaworld()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")
    
