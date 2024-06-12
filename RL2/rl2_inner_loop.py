import random
import numpy as np
import torch
from rl2_inner_loop_utils import Lifetime_buffer
from rl2_outer_loop_utils import Outer_loop_action_agent
import metaworld
import gymnasium as gym

def run_inner_loop(arguments , training=True, run_deterministically=False):

    config = arguments[0]
    model_path= arguments[1]
    benchmark_name= arguments[2]
    task= arguments[3]

    ######## SETUP #########

    # Construct the appropriate benchmark and a method to create an environment with the task specified in the argument 'task'
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{task.env_name}', seed=config.seed)
        def create_env(task):
            env = benchmark.train_classes[f'{task.env_name}']()  
            env.set_task(task)  
            env = gym.wrappers.ClipAction(env)
            if config.seeding==True:
                env.action_space.seed(config.seed)
                env.observation_space.seed(config.seed)
            return env
    elif benchmark_name=='ML10' or benchmark_name=='ML45':
        benchmark = metaworld.ML10(seed=config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=config.seed)
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
        
    #create env and load meta agent
    env= create_env(task)    
    actions_size=env.action_space.shape[0]
    obs_size=env.observation_space.shape[0] 

    meta_agent=Outer_loop_action_agent(       
                actions_size,          
                obs_size,         
                rnn_input_size=config.rnn_input_size ,    
                rnn_type= config.rnn_type,              
                rnn_hidden_state_size= config.rnn_hidden_state_size,
                initial_std=config.initial_std ,
                input_sparse_or_shaped=config.sparse_or_shaped_inputs
                )
    meta_agent.load_state_dict(torch.load(model_path))

    #----
    if config.seeding==True:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.il_device=='auto':
        il_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        il_device=config.il_device

    meta_agent=meta_agent.to(il_device)
    lifetime_buffer=Lifetime_buffer(config.num_il_lifetime_steps , env, il_device, env_name=f'{task.env_name}')
    

    ####### Inner loop #############
    
    episode_step_num=0
    max_episode_steps=500 

    episode_return=0
    episodes_lengths=[]
    episodes_returns=[]
    episodes_successes=[] 
    succeded_in_episode=False 


    #get an initial state from the agent to start training
    next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)


    #####-- MAIN LOOP ----########

    for global_step in range(0, config.num_il_lifetime_steps):


        obs, prev_done = next_obs, done

        #create dummy elements to store in the first step
        if global_step==0:
            action=torch.from_numpy(np.zeros(env.action_space.shape[0])) #(have to do this because there is no action at timestep -1)
            logprob= torch.zeros(1)
            reward=torch.zeros(1)# pass 'reward' of 0 for timestep -1
            success_signal= torch.zeros(1)# pass 'success_signal' of 0 for timestep -1

        #store (s_t ,a_{t-1} , logprob_{t-1}, d_t ,r_{t-1})
        lifetime_buffer.store_step_data(global_step=global_step, obs=obs.to(il_device), prev_act=action.to(il_device),
            prev_reward=reward.to(il_device), prev_success_signal=success_signal.to(il_device), prev_logp=logprob.to(il_device),prev_done=prev_done.to(il_device))

        #-------OUTER LOOP agent involvement ---------------
        # get meta agent predictions and state value estimates conditioning on the lifetime history
        if global_step==0:
            hidden_state=meta_agent.initialize_state(batch_size=1)
            if isinstance(hidden_state, tuple):
                hidden_state = tuple(hs.to(il_device) for hs in hidden_state)
            else:
                hidden_state.to(il_device)
            
        hidden_state=meta_agent.rnn_next_state(lifetime_buffer ,lifetime_timestep=global_step ,
                                            rnn_current_state=hidden_state)  #(1,1,hidden_state_size)
        
        if run_deterministically==False:
            with torch.no_grad():
                meta_value=meta_agent.get_value(hidden_state).squeeze(0).squeeze(0) #(1)
                action, logprob, _ = meta_agent.get_action(hidden_state)
                action=action.squeeze(0).squeeze(0)  #(action_size)
                logprob=logprob.squeeze(0).squeeze(0) #(1)

        #have meta agent take only deterministic actions. Should only be used in an evaluation setting (when not training it) .
        elif run_deterministically==True:
            with torch.no_grad():
                meta_value=torch.ones(1) #(1)
                action = meta_agent.get_deterministic_action(hidden_state)
                action=action.squeeze(0).squeeze(0)  #(action_size)
                logprob=torch.zeros(1) #(1)


        lifetime_buffer.store_meta_value(global_step=global_step, meta_value=meta_value)
        #-----------------------------------------------
        #execute the action and get environment response.
        next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done= torch.max(torch.Tensor([terminated,truncated]))

        if info['success'] == 1.0 and succeded_in_episode==False:
            success_signal = 1.0 - 0.7*(episode_step_num/max_episode_steps)
        elif episode_step_num==(max_episode_steps-1) and succeded_in_episode==False:
            success_signal = -0.2
        else:
            success_signal= 0.0
        
        #prepare for next step
        reward= torch.tensor(reward).to(il_device)
        success_signal=torch.tensor(success_signal).to(il_device)
        done = torch.Tensor(done).to(il_device)
        next_obs=torch.Tensor(next_obs).to(il_device)
        
        episode_step_num+=1 
        episode_return+= reward
        if info['success'] == 1.0: 
            succeded_in_episode=True 

        #When in last step save the last action taken and reward received in a extra timeslot
        if global_step== config.num_il_lifetime_steps-1:
            dummy_obs= torch.from_numpy(np.zeros(env.observation_space.shape[0])) #isnt used for anything ,could also save obs of timestep T+1 instead of zeros
            lifetime_buffer.store_step_data(global_step=global_step+1, obs=dummy_obs.to(il_device), prev_act=action.to(il_device),
            prev_reward=reward.to(il_device), prev_success_signal=success_signal.to(il_device), prev_logp=logprob.to(il_device),prev_done=done.to(il_device))
            

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


    lifetime_buffer.episodes_returns=episodes_returns
    lifetime_buffer.episodes_successes = episodes_successes 

    #if its run for evaluation there is no need to return all the lifetime's data
    if training==False:
        return episodes_returns,episodes_successes
    elif training==True:
        return lifetime_buffer