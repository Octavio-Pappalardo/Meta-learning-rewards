import random
import numpy as np
import torch
import torch.optim as optim


from inner_loop_utils import Inner_loop_action_agent ,IL_buffer, Lifetime_buffer, PPO_class
from inner_loop_config import get_config
from outer_loop_utils import Outer_loop_reward_agent
import metaworld
import gymnasium as gym


def run_inner_loop(arguments ,intrinsic_rewards=True,training=True,return_agent=False):
    ol_config=arguments[0]
    il_config_setting=arguments[1]
    model_path= arguments[2]
    benchmark_name= arguments[3]
    task= arguments[4]

    config=get_config(il_config_setting)

    ######## SETUP #########

    # Construct the appropriate benchmark and a method to create an environment with the task specified in the argument 'task'
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{task.env_name}', seed=ol_config.seed)
        def create_env(task):
            env = benchmark.train_classes[f'{task.env_name}']()  
            env.set_task(task)  # Set task
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

   
    ####### Inner loop #############

    global_step = 0
    episode_step_num=0
    max_episode_steps=500 #maximum number of steps the agent can take in an episode
    episode_return=0
    episodes_lengths=[]
    episodes_returns=[]
    episodes_successes=[] # it keeps track of wether the goal was achieved in each episode
    succeded_in_episode=False #keeps track of wether the agent has achieved succes in current episode


    #get an initial state from the agent to start training
    next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)


    for update in range(1, config.num_updates + 1):

        if config.anneal_lr==True:
            frac = 1.0 - (update - 1.0) / config.num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        ##---DATA COLLECTION STAGE-----

        for step in range(0, config.num_steps):

            #prepare for new step: next_obs is the new step obs , the done flag which indicates wether the episode finished in the last step becomes this episodes prev_done.
            obs, prev_done = next_obs, done

            # get actionA action predictions and state value estimates
            with torch.no_grad():
                action, logprob, _, value = actionA.get_action_and_value(obs.unsqueeze(0))

            #execute the action and get environment response.
            next_obs, e_reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            done= torch.max(torch.tensor([terminated,truncated],dtype=torch.float32))

            #preprocess and store data that you get from a step in a standard RL algo and store it in lifetime buffer
            e_reward= torch.as_tensor(e_reward,dtype=torch.float32).to(il_device)
            lifetime_buffer.store_step_data(global_step=global_step, obs=obs.to(meta_device), act=action.to(meta_device),    
                            e_reward=e_reward.to(meta_device), logp=logprob.to(meta_device),prev_done=prev_done.to(meta_device))
            
            #save the sparse signal of success
            if info['success'] == 1.0 and succeded_in_episode==False:
                lifetime_buffer.success_signal[global_step]= 1.0 - 0.7*(episode_step_num/max_episode_steps)
            elif episode_step_num==(max_episode_steps-1) and succeded_in_episode==False:
                lifetime_buffer.success_signal[global_step]= -0.2
            else:
                lifetime_buffer.success_signal[global_step]=0.0
            


            if intrinsic_rewards==True:
                #-------OUTER LOOP agent involvement ---------------
            #get an encoding of the whole lifetime up to current step
                if global_step==0:
                    hidden_state=rewardA.initialize_state(batch_size=1)
                    if isinstance(hidden_state, tuple):
                        hidden_state = tuple(hs.to(meta_device) for hs in hidden_state)
                    else:
                        hidden_state.to(meta_device)
                    
                hidden_state=rewardA.rnn_next_state(lifetime_buffer ,lifetime_timestep=global_step ,
                                                    rnn_current_state=hidden_state)  #(1,1,hidden_state_size)
                
                #compute reward agent predictions conditioning on the lifetime history
                if training==True:
                    with torch.no_grad():
                        meta_value=rewardA.get_value(hidden_state).squeeze(0).squeeze(0) #(1)
                        i_reward, logp_i_reward , _ = rewardA.get_action(hidden_state)
                        i_reward=i_reward.squeeze(0).squeeze(0)  #(1)
                        logp_i_reward=logp_i_reward.squeeze(0).squeeze(0) #(1)
                elif training==False:
                    with torch.no_grad():
                        meta_value=torch.ones(1) #(1)
                        i_reward= rewardA.get_deterministic_action(hidden_state)
                        i_reward=i_reward.squeeze(0).squeeze(0)  #(1)
                        logp_i_reward=torch.zeros(1) #(1)

                # store the predictions from the outer loop agent
                lifetime_buffer.store_outer_loop_predictions(global_step=global_step, i_reward=i_reward,
                                                            logp_i_reward=logp_i_reward,meta_value=meta_value)
                
                #-----------------------------------------------
            else:
                #i_reward=e_reward #for shaped extrinsic rewards
                i_reward=lifetime_buffer.success_signal[global_step]

            #store data used by the inner loop learning algorithm
            actionA_buffer.store_inner_loop_update_data(step_index=step, obs=obs, act=action, reward=i_reward.to(il_device), val=value
                                                        , logp=logprob,prev_done=prev_done)


            #prepare for next step
            next_obs, done =torch.as_tensor(next_obs,dtype=torch.float32).to(il_device) ,torch.as_tensor(done,dtype=torch.float32).to(il_device)
            global_step += 1 
            episode_step_num+=1 
            episode_return+= e_reward
            if info['success'] == 1.0: 
                succeded_in_episode=True 
            

            #deal with the case where the episode ends for  the environment when working with metaworld (doesnt return terminated or truncated)
            if episode_step_num==max_episode_steps :
                #log metrics
                episodes_returns.append(episode_return)
                episodes_lengths.append(episode_step_num)
                if succeded_in_episode==True:
                    episodes_successes.append(1.0) 
                else:
                    episodes_successes.append(0.0)

                #Prepare for next episode
                done=torch.ones(1).to(il_device) #set done manually to True
                next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(il_device) #reset environment
                episode_step_num=0
                episode_return=0
                succeded_in_episode=False

        
        with torch.no_grad():
            #get state value estimate of the new state
            next_value = actionA.get_value(next_obs)
            #calculate the advantages and to go returns for each state in the buffer. 
            actionA_buffer.calculate_returns_and_advantages(done,next_value,gae=config.gae,gamma=config.gamma,gae_lambda=config.gae_lambda)


        ###----  UPDATE STAGE  ----###
            
        if update != config.num_updates: # so that life doesnt end in an update
            #retrieve data needed for update from buffer
            PPO.get_data_from_buffer(actionA_buffer,env)
            #perform the update
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

    lifetime_buffer.episodes_returns=episodes_returns
    lifetime_buffer.episodes_successes = episodes_successes 


    #if it is run for evaluation there is no need to return all the lifetime's data
    if training==False and return_agent==False:
        return episodes_returns,episodes_successes
    elif training==False and return_agent==True:
        return (episodes_returns,episodes_successes), actionA
    
    elif training==True and return_agent==True:
        return lifetime_buffer , actionA
    else:
        return lifetime_buffer