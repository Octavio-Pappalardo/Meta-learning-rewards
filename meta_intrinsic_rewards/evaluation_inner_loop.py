import random
import numpy as np
import torch
import torch.optim as optim
from inner_loop_utils import Inner_loop_action_agent ,IL_buffer, Lifetime_buffer, PPO_class
from inner_loop_config import get_config
from outer_loop_utils import Outer_loop_reward_agent
import metaworld
import gymnasium as gym



def run_eval_inner_loop(arguments ,intrinsic_rewards=True, num_deterministic_evaluation_episodes=10):
    ''' Collects data for evaluating meta agents. Has two steps:
    1) Trains an inner loop agent with the intrinsic rewards from the meta agent according to the il_configuratins and records the return obtained by the agent on
    each episode an whether it succeeded. This step is the same as running run_inner_loop from inner_loop.py whith training=False argument.
    2) In the second step it doesnt keep training the inner loop agent . It makes the inner loop action agent deterministic and runs it for 
    num_deterministic_evaluation_episodes recording the episodes returns and successes.

    '''

    ########## Settings and preparations ########
    ol_config=arguments[0]
    il_config_setting=arguments[1]
    model_path= arguments[2]
    benchmark_name= arguments[3]
    task= arguments[4]


    config=get_config(il_config_setting)

    # Construct the appropriate benchmark and a method to create an environment with the task specified in the argument 'task'
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{task.env_name}', seed=ol_config.seed)
        def create_env(task):
            env = benchmark.train_classes[f'{task.env_name}']()  
            env.set_task(task)  
            env = gym.wrappers.ClipAction(env)
            if ol_config.seeding==True:
                env.action_space.seed(ol_config.seed)
                env.observation_space.seed(ol_config.seed)
            return env
    elif benchmark_name=='ML10' or benchmark_name=='ML45':
        benchmark = metaworld.ML10(seed=ol_config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=ol_config.seed)
        def create_env(task):
            benchmark_envs=benchmark.train_classes.copy()
            benchmark_envs.update(benchmark.test_classes)
            env=benchmark_envs[f'{task.env_name}']()
            env.set_task(task)
            env = gym.wrappers.ClipAction(env)
            if ol_config.seeding==True:
                env.action_space.seed(ol_config.seed)
                env.observation_space.seed(ol_config.seed)
            return env
  

    env= create_env(task)    
    actions_size=env.action_space.shape[0]
    obs_size=env.observation_space.shape[0]
    rewardA=Outer_loop_reward_agent(
                    actions_size,          
                    obs_size,           
                    rnn_input_size=ol_config.rnn_input_size ,    
                    rnn_type= ol_config.rnn_type,              
                    rnn_hidden_state_size= ol_config.rnn_hidden_state_size ,
                    initial_std=ol_config.initial_std ,
                    input_sparse_or_shaped=ol_config.sparse_or_shaped_inputs
                    )
    
    rewardA.load_state_dict(torch.load(model_path)) 

    #----
    if config.seeding==True:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.il_device=='auto':
        il_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        il_device=config.il_device
    if config.meta_device=='auto':
        meta_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        meta_device=config.meta_device

    actionA = Inner_loop_action_agent(env).to(il_device)
    optimizer = optim.Adam(actionA.parameters(), lr=config.learning_rate, eps=config.adam_eps)

    actionA_buffer=IL_buffer(config.num_steps, env, il_device)

    rewardA=rewardA.to(meta_device)
    lifetime_buffer=Lifetime_buffer(config.num_lifetime_steps ,config.num_steps, env, meta_device, env_name=f'{task.env_name}')

    PPO=PPO_class()


    ####### Inner loop #############3

    global_step = 0
    episode_step_num=0
    max_episode_steps=500
    episode_return=0
    episodes_lengths=[]
    episodes_returns=[]
    episodes_successes=[] 
    succeded_in_episode=False 

    next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)


    for update in range(1, config.num_updates + 1):
        if config.anneal_lr==True:
            frac = 1.0 - (update - 1.0) / config.num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        #DATA COLLECTION STAGE
        for step in range(0, config.num_steps):

            obs, prev_done = next_obs, done
            with torch.no_grad():
                action, logprob, _, value = actionA.get_action_and_value(obs.unsqueeze(0))

            next_obs, e_reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            done= torch.max(torch.tensor([terminated,truncated],dtype=torch.float32))

            e_reward= torch.as_tensor(e_reward,dtype=torch.float32).to(il_device)
            lifetime_buffer.store_step_data(global_step=global_step, obs=obs.to(meta_device), act=action.to(meta_device),
                            e_reward=e_reward.to(meta_device), logp=logprob.to(meta_device),prev_done=prev_done.to(meta_device))
            
            if info['success'] == 1.0 and succeded_in_episode==False:
                lifetime_buffer.success_signal[global_step]= 1.0 - 0.7*(episode_step_num/max_episode_steps)
            elif episode_step_num==(max_episode_steps-1) and succeded_in_episode==False:
                lifetime_buffer.success_signal[global_step]= -0.2
            else:
                lifetime_buffer.success_signal[global_step]=0.0

            if intrinsic_rewards==True:
                #-------OUTER LOOP agent involvement ---------------
                if global_step==0:
                    hidden_state=rewardA.initialize_state(batch_size=1)
                    if isinstance(hidden_state, tuple):
                        hidden_state = tuple(hs.to(meta_device) for hs in hidden_state)
                    else:
                        hidden_state.to(meta_device)
                    
                hidden_state=rewardA.rnn_next_state(lifetime_buffer ,lifetime_timestep=global_step ,
                                                    rnn_current_state=hidden_state) 
                with torch.no_grad():
                    meta_value=torch.ones(1) 
                    i_reward= rewardA.get_deterministic_action(hidden_state)
                    i_reward=i_reward.squeeze(0).squeeze(0)  
                    logp_i_reward=torch.zeros(1) 

                lifetime_buffer.store_outer_loop_predictions(global_step=global_step, i_reward=i_reward,
                                                            logp_i_reward=logp_i_reward,meta_value=meta_value)
                #-----------------------------------------------
            else:
                i_reward=lifetime_buffer.success_signal[global_step]#e_reward


            actionA_buffer.store_inner_loop_update_data(step_index=step, obs=obs, act=action, reward=i_reward.to(il_device), val=value
                                                        , logp=logprob,prev_done=prev_done)


            #prepare for next step
            next_obs, done =torch.as_tensor(next_obs,dtype=torch.float32).to(il_device) ,torch.as_tensor(done,dtype=torch.float32).to(il_device)
            global_step += 1 
            episode_step_num+=1 
            episode_return+= e_reward
            if info['success'] == 1.0: 
                succeded_in_episode=True 


            #deal with the case where the episode ends 
            if episode_step_num==max_episode_steps :
                episodes_returns.append(episode_return)
                episodes_lengths.append(episode_step_num)
                if succeded_in_episode==True:
                    episodes_successes.append(1.0) 
                else:
                    episodes_successes.append(0.0)

                done=torch.ones(1).to(il_device) 
                next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device) 
                episode_step_num=0
                episode_return=0
                succeded_in_episode=False

            
        with torch.no_grad():
            next_value = actionA.get_value(next_obs)
            actionA_buffer.calculate_returns_and_advantages(done,next_value,gae=config.gae,gamma=config.gamma,gae_lambda=config.gae_lambda)

        #UPDATE STAGE. 
        if update != config.num_updates:
            PPO.get_data_from_buffer(actionA_buffer,env)
            PPO.update(agent=actionA, optimizer=optimizer, 
                        update_epochs = config.ppo['update_epochs'],
                        num_minibatches = config.ppo['num_minibatches'],
                        normalize_advantage = config.ppo['normalize_advantage'],
                        clip_coef = config.ppo['clip_coef'],
                        clip_vloss = config.ppo['clip_vloss'],
                        entropy_coef = config.ppo['entropy_coef'],
                        valuef_coef = config.ppo['valuef_coef'],
                        clip_grad_norm=config.ppo["clip_grad_norm"],
                        max_grad_norm = config.ppo['max_grad_norm'],
                        target_KL = config.ppo['target_KL'],
                        )


    ##############################  FINISHED STEP 1   #####################################
    ##############################                    #####################################

    eval_episode_num=0
    episode_step_num=0
    episode_return=0
    succeded_in_episode=False 


    #get an initial state from the agent to start training
    next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)

    #Run the inner loop agent deterministically for num_deterministic_evaluation_episodes episodes.
    while eval_episode_num != num_deterministic_evaluation_episodes:

        obs= next_obs

        with torch.no_grad():
            action = actionA.get_deterministic_action(obs.unsqueeze(0)) 
        next_obs, e_reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())

        #prepare for next step
        next_obs =torch.as_tensor(next_obs,dtype=torch.float32).to(il_device)
        episode_step_num+=1 
        episode_return+= e_reward
        if info['success'] == 1.0: 
            succeded_in_episode=True 

        #deal with episode ending .
        if episode_step_num==500 :
            eval_episode_num+=1
            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_step_num)
            if succeded_in_episode==True:
                episodes_successes.append(1.0) 
            else:
                episodes_successes.append(0.0)

            next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device) 
            episode_step_num=0
            episode_return=0
            succeded_in_episode=False


    ####################### FINISHED STEP 2  ###########################
                
    return episodes_returns,episodes_successes