import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torchopt
import metaworld
import time
import random
import sys
sys.path.append('../')

import wandb
import ray


from maml_config import get_config
from maml_utils import Action_agent , collect_data_from_env , policy_loss_for_adaptation , maml_trpo_update
from meta_intrinsic_rewards.outer_loop_utils import OL_buffer, Outer_loop_reward_agent, Outer_loop_TBPTT_PPO
from meta_intrinsic_rewards.inner_loop_utils import Lifetime_buffer
from general_utils import Logger ,Statistics_tracker
from shared_code.shared_utils import Sampler ,Tasks_batch_sampler



config_setting='metaworld'
config=get_config(config_setting)

intrinsic_rewards=config.intrinsic_rewards
benchmark_name= config.benchmark_name
env_name= config.env_name
sparse_or_shaped_training_target=config.sparse_or_shaped_training_target

steps_per_episode=500 #number of steps in a metaworld episode - just used for logging purposes

############ SETUP #############


# Construct the benchmark and construct an iterator that returns batches of tasks. (and sample a random env for setting up some configurations)
if benchmark_name=='ML1':
    benchmark = metaworld.ML1(f'{env_name}', seed=config.seed)
    exp_name= f'{benchmark_name}_{env_name}' 
    task_sampler = Sampler(benchmark.train_tasks, config.num_inner_loops_per_update)
    example_env=benchmark.train_classes[f'{env_name}']()
    
    def create_env(task):
        env = benchmark.train_classes[f'{task.env_name}']()  
        env.set_task(task)  
        env = gym.wrappers.ClipAction(env)
        if config.seeding==True:
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)
        return env

elif benchmark_name=='ML10':
    benchmark= metaworld.ML10(seed=config.seed)
    exp_name= f'{benchmark_name}'
    task_sampler=Tasks_batch_sampler(benchmark,config.num_inner_loops_per_update)
    example_env=next(iter(benchmark.train_classes.items()))[1]()
elif benchmark_name=='ML45':
    benchmark=metaworld.ML45(seed=config.seed)
    task_sampler=Tasks_batch_sampler(benchmark,config.num_inner_loops_per_update)
    exp_name= f'{benchmark_name}'
    example_env=next(iter(benchmark.train_classes.items()))[1]()

if benchmark_name=='ML10' or benchmark_name=='ML45':
    def create_env(task):
        benchmark_envs=benchmark.train_classes.copy()
        benchmark_envs.update(benchmark.test_classes)
        env=benchmark_envs[f'{task.env_name}']()
        env.set_task(task)
        env = gym.wrappers.ClipAction(env)
        if config.seeding==True:
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)
        return env



model_id=int(time.time())
run_name = f"MAML_{exp_name}__{model_id}"
wandb.init(project='project_name',
                name= run_name,
                config=vars(config))


if config.seeding==True:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

if config.device=='auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device=config.device


maml_agent = Action_agent(example_env).to(device)
maml_optimizer = torchopt.Adam(maml_agent.parameters(), lr=config.maml_agent_lr, eps=config.maml_agent_epsilon)

actions_size=example_env.action_space.shape[0]
obs_size=example_env.observation_space.shape[0] 
del example_env
rewardA=Outer_loop_reward_agent(
                    actions_size,          
                    obs_size,
                    rnn_input_size=config.rnn_input_size ,    
                    rnn_type= config.rnn_type,              
                    rnn_hidden_state_size= config.rnn_hidden_state_size ,
                    initial_std=config.initial_std ,
                    input_sparse_or_shaped=config.sparse_or_shaped_inputs
                    ).to(device)
reward_agent_optimizer = optim.Adam(rewardA.parameters(), lr=config.rew_agent_lr, eps=config.rew_agent_epsilon)

meta_buffer=OL_buffer(device=config.device_rew_agent_update)
TBPTT_PPO=Outer_loop_TBPTT_PPO(reward_agent_optimizer, logging=True ,
                k=config.rew_agent_ppo['k'], 
                update_epochs=config.rew_agent_ppo['update_epochs'],
                num_minibatches=config.rew_agent_ppo['num_minibatches'],
                normalize_advantage=config.rew_agent_ppo['normalize_advantage'],
                entropy_coef=config.rew_agent_ppo['entropy_coef'],
                valuef_coef=config.rew_agent_ppo['valuef_coef'],
                clip_grad_norm=config.rew_agent_ppo['clip_grad_norm'],
                max_grad_norm=config.rew_agent_ppo['max_grad_norm'],
                target_KL=config.rew_agent_ppo['target_KL'],
                clip_coef=config.rew_agent_ppo['clip_coef'])

    

data_statistics=Statistics_tracker()

logger=Logger(num_epsiodes_of_validation=config.num_epsiodes_of_validation)

best_rew_agent_model_path= f"../../maml_models/rew_{run_name}__best_model.pth"
best_maml_agent_model_path= f"../../maml_models/{run_name}__best_model.pth"
best_model_performance = 0 

#for determining best model version 
def validation_performance(logger):
    performance= np.array(logger.validation_episodes_success_percentage[-config.num_lifetimes_for_validation:]).mean()
    performance=performance + 1e-6*  np.mean(logger.validation_episodes_return[-config.num_lifetimes_for_validation:]) #for resolving ties
    return performance


############ DEFINE BEHAVIOUR OF INNER LOOPS #############


def inner_loop(base_policy_state_dict,  task , reward_agent ,config,data_statistics, intrinsic_rewards=False): 
    import sys
    sys.path.append('../')

    env=create_env(task)
    
    if intrinsic_rewards==True:
        adaptation_mean_reward_for_baseline= data_statistics.intrinsic_rewards_mean 
    elif intrinsic_rewards==False and config.shaped_rewards_available_at_test_time==True:
        adaptation_mean_reward_for_baseline= data_statistics.e_rewards_mean 
    elif intrinsic_rewards==False and config.shaped_rewards_available_at_test_time==False:
        adaptation_mean_reward_for_baseline= data_statistics.sparse_rewards_mean 

  
    maml_agent=Action_agent(env).to(device)
    torchopt.recover_state_dict(maml_agent, base_policy_state_dict)
    inner_optimizer =torchopt.MetaSGD(maml_agent, lr=config.adaptation_lr)

    lifetime_buffer=Lifetime_buffer(config.num_lifetime_steps ,config.num_env_steps_per_adaptation_update, env, device ,env_name=f'{task.env_name}')

    # ADAPTATION - K gradient steps
    # for each adaptation step : collect data with current policy (starting with the base policy) for doing the inner loop adaptation. Use this data to
    # Compute gradient of the tasks's policy objective function  with respect to the current policy parameters.
    # finally use this gradient to update the parameters. Repeat the process config.num_adaptation_steps times .

    information={'current_state':torch.tensor(env.reset()[0],dtype=torch.float32).to(device) , 'prev_done' :torch.ones(1).to(device) ,
            'current_episode_step_num':0, 'current_episode_success':False ,'current_episode_return':0,
            'current_lifetime_step':0 , 'hidden_state':None}

    for _ in range(config.num_adaptation_updates_in_inner_loop):
        buffer, information =collect_data_from_env(agent=maml_agent,reward_agent=reward_agent,env=env,num_steps=config.num_env_steps_per_adaptation_update , information=information,
                                    config=config, lifetime_buffer=lifetime_buffer ,intrinsic_rewards=intrinsic_rewards,
                                    training=True , mean_reward_for_baseline=adaptation_mean_reward_for_baseline)

        pg_loss = policy_loss_for_adaptation(agent=maml_agent, buffer=buffer)
        inner_optimizer.step(pg_loss)


    #META LOSS CALCULATION
    # collect data with adapted policy for the outer loop update (for computing the loss that is used for the maml meta update - updating the base policy) . 
    maml_buffer, _ =collect_data_from_env(agent=maml_agent ,reward_agent=reward_agent,env=env, num_steps=config.num_env_steps_for_estimating_maml_loss, information=information,
                                    config=config, lifetime_buffer=lifetime_buffer ,env_name=f'{task.env_name}' ,
                                    for_maml_loss=True, intrinsic_rewards=intrinsic_rewards,training=True )

    adapted_policy_state_dict = torchopt.extract_state_dict(maml_agent)

    return lifetime_buffer , maml_buffer, adapted_policy_state_dict





############ OUTER LOOP #############

if ray.is_initialized:
    ray.shutdown()
ray.init()

remote_inner_loop=ray.remote(inner_loop)
config_ref=ray.put(config)

for i in range(config.num_outer_loop_updates):

    policy_state_dict = torchopt.extract_state_dict(maml_agent) #extract the state of the base maml agent (before being adapted to a specific env)

    rewardA=rewardA.to(device) 
    rewardA_ref=ray.put(rewardA)
    policy_state_dict_ref=ray.put(policy_state_dict)
    data_statistics_ref=ray.put(data_statistics)

    inputs=[ [policy_state_dict_ref, task , rewardA_ref , config_ref, data_statistics_ref, intrinsic_rewards] for task in next(task_sampler)]

    ##--------------COLLECTING DATA --------------
    lifetime_buffers ,maml_buffers, adapted_policy_state_dicts = zip(*ray.get([remote_inner_loop.options(num_cpus=1).remote(*i) for i in inputs])) 
    
    # take maml_agent back to its original state 
    torchopt.recover_state_dict(maml_agent, policy_state_dict)

    ##--------------PROCESSING STAGE --------------
    #use collected data to compute statistics (used for normalization of rewards)
    for lifetime_data in lifetime_buffers:
        data_statistics.update_statistics(lifetime_data)

    #preprocess collected data
    for lifetime_data in lifetime_buffers:
        lifetime_data.preprocess_data(data_stats=data_statistics , objective_mean=config.e_rewards_target_mean_for_reward_agent) #normalize extrinsic rewards
        lifetime_data.compute_meta_advantages_and_returns_to_go(sparse_or_shaped=sparse_or_shaped_training_target, gamma=config.rew_agent_gamma, estimation_method=config.rew_agent_ae["estimation_method"],
                                                            e_lambda=config.rew_agent_ae["bootstrapping_lambda"], starting_n= config.rew_agent_ae["starting_n"], 
                                                            num_to_average_over=config.rew_agent_ae["num_n_step_estimates"] ,skip_rate=config.rew_agent_ae["skip_rate"])
        logger.collect_per_lifetime_metrics(lifetime_data,episodes_till_first_adaptation_update= config.num_env_steps_per_adaptation_update//steps_per_episode, episodes_after_adaptation=config.num_env_steps_for_estimating_maml_loss//steps_per_episode)
        
        meta_buffer.collect_lifetime_data(lifetime_data)

    for maml_buffer in maml_buffers:
        if config.shaped_rewards_available_at_train_time==True:
            maml_buffer.preprocess_data(data_stats=data_statistics , objective_mean=config.e_rewards_target_mean_for_maml_agent)
            mean_reward_for_control_variate=config.e_rewards_target_mean_for_maml_agent
        else:
            mean_reward_for_control_variate=data_statistics.sparse_rewards_mean
        maml_buffer.calculate_returns_and_advantages(mean_reward=config.e_rewards_target_mean_for_maml_agent ,gamma=config.maml_agent_gamma)


    meta_buffer.combine_data()
    #---log and print metrics
    logger.log_per_update_metrics(num_inner_loops_per_update=config.num_inner_loops_per_update)
    print(f'completed meta update {i} , base policy return and success percentage={np.array(logger.base_maml_agent_return[-config.num_inner_loops_per_update:]).mean()} ,{np.array(logger.base_maml_agent_success_percentage[-config.num_inner_loops_per_update:]).mean()} , adapted policy return and success percentage={np.array(logger.adapted_maml_agent_return[-config.num_inner_loops_per_update:]).mean()} ,{np.array(logger.adapted_maml_agent_success_percentage[-config.num_inner_loops_per_update:]).mean()}')



    #-------Save model if required----
    
    model_performance=validation_performance(logger)
    if model_performance > best_model_performance:
        best_model_performance=model_performance
        print(f'best model performance= {best_model_performance}')
        torch.save(maml_agent.state_dict(), best_maml_agent_model_path)
        rew_state_dict=rewardA.state_dict()
        statistics = {
            'i_reward_mean': data_statistics.intrinsic_rewards_mean,
            'e_reward_mean': data_statistics.e_rewards_mean,
            'sparse_reward_mean':data_statistics.sparse_rewards_mean
            }
        data_to_save = {
            'rew_model_state_dict': rew_state_dict,
            'statistics': statistics
        }
        torch.save(data_to_save, best_rew_agent_model_path)


    # --------------  UPDATE MODELS ------------------------
        
    rewardA=rewardA.to(config.device_rew_agent_update) 

    if config.intrinsic_rewards==False:
        maml_trpo_update(maml_agent=maml_agent, data_buffers=maml_buffers,adapted_policies_states=adapted_policy_state_dicts ,config=config, logging=True)
    else:
        TBPTT_PPO.update(reward_agent=rewardA,buffer=meta_buffer ,dont_consider_last=config.num_env_steps_for_estimating_maml_loss)
        maml_trpo_update(maml_agent=maml_agent, data_buffers=maml_buffers,adapted_policies_states=adapted_policy_state_dicts ,config=config, logging=True)


    meta_buffer.clean_buffer()


wandb.finish()