
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
        self.initial_std=0.2

        self.ppo={
            "k" : 400,
            'update_epochs' : 12,
            'num_minibatches': 0, 
            "normalize_advantage": True,
            "clip_coef": 0.2, 
            "entropy_coef": 0.003,
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
        self.seed=2
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
        self.initial_std=0.2

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
    





#OUTER LOOP CONFIGURATIONS AND HYPERPARAMETERS



#####-----  Dataset-benchmark used -----####
#benchmark_name : Selects which benchmark to run
#env_name  : selects choice of environment when benchmark name is 'ML1' . EJ  'pick-place-v2', 'door-close-v2' , 'soccer-v2'


##########--------  what rewards to use  --------################
#shaped_rewards_available_at_train_time  :  sets wether there is access to shaped or sparse rewards during training. 
                                            #controls value of sparse_or_shaped_training_target
#sparse_or_shaped_training_target  :  controls the objective that the meta agent is trained to maximize. The options are: 
                        #1) 'shaped' to maximize the shaped extrinsic rewards that the inner loop agent receives throughout a lifetime and
                        #2) 'sparse ' to maximize directly a sparse signal of task successes .

#shaped_rewards_available_at_test_time   :   sets wether there is access to shaped or sparse rewards when encountering the test environments. 
                                            #controls value of sparse_or_shaped_inputs
#sparse_or_shaped_inputs :  controls wether the recurrent meta agent receives as input the sparse or the shaped extrinsic reward at each step. 
                            #if shaped rewards are not available at test time, then the recurrent agent has to only take as input the sparse signal (both at train and test time)


######----- model validation performance -----#########
#num_epsiodes_of_validation : controls how many of the last episodes of each lifetime are used for computing the vaildation metrics
#num_lifetimes_for_validation : controls how many of the last lifetimes are used to compute the validation metrics
          #note that validation metrics are just the metrics used for deciding which model version to keep but they arent computed on a different dataset


#device : controls on what device the meta agent runs while training


####----- amount of data -------########
#num_outer_loop_updates : number of times to collect data from inner loops and update the reward agent
#num_inner_loops_per_update  #number of inner loops runned to collect data for the outer loop update



####----learning rate -------#####
#learning_rate: optimizers learning rate
#anneal_lr  :  Toggle learning rate annealing for policy and value network gradient descent updates

####---- advantage estimation ------########
#e_rewards_target_mean  :  the extrinsic rewards from each  environment are normalized such that their new mean value is e_rewards_target_mean
#meta_gamma  : the gamma considered for the meta objective - 
# the other parameters of advantage estimation are explanined in the Lifetime_buffer class .

###-----reward agent parameters----#### 
#rnn_input_size  : size of the imput to the rnn at each step  
#rnn_type  :  The type of rnn network used ,'gru' ,'lstm' ,'rnn'            
#rnn_hidden_state_size : The dimensionality of the rnn hidden state
#initial_std :  the initial standard deviation of the intrinsic rewards given


#####---TBPTT PPO parameters----#####
#k  :  when using TBPTT determines the length at which gradients are cuttof. k=k1=k2 .
#update_epochs  :        #number of epochs runned inside the PPO update
#num_minibatches    :      the number of mini-batches created from each batch for the PPO update - number of gradient step taken per epoch . If 0 one is taken for each k length sequence.
#normalize_advantage    :  Toggles advantages normalization -  if True the advantages of the whole batch are normalized
#clip_coef  :              the PPO clipping coefficient 
#entropy_coef   :     coefficient of the entropy - controls the weight of the exploration encouraging entropy loss in the total loss with wich agent is updated.
#valuef_coef    :     coefficient of the value function - controls the weight of the value function loss in the total loss with wich agent is updated.
#max_grad_norm  :     the maximum norm the gradient of each gradient step is allowed to have. Parameter for gradient clipping
#target_KL  :      the KL divergence threshold