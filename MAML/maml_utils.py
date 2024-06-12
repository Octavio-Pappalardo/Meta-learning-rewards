import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import wandb
import torchopt

from torch.nn.utils import parameters_to_vector ,vector_to_parameters
from torch.distributions.normal import Normal



#-- 4 sections : data collection  , adaptation loss , maml neural net  , maml trpo ---#



############################################################################


#################---------   DATA COLLECTION   ---------###############

def collect_data_from_env(agent,reward_agent,env,num_steps , information, config,
                          lifetime_buffer , env_name=None, intrinsic_rewards=True,training=True,for_maml_loss=False , mean_reward_for_baseline=None
                          ,device='cpu'):

    episodes_returns=[]
    max_episode_steps=500
    episode_step_num=information['current_episode_step_num']
    episode_return= information['current_episode_return']
    episodes_successes=[] #keeps track of wether the goal was achieved in each episode
    succeded_in_episode=information['current_episode_success'] #keeps track of wether the agent has achieved succes in current episode
    current_lifetime_step=information['current_lifetime_step']
    hidden_state=information['hidden_state']

    #instantiate a buffer to save the data in
    actionA_buffer= Action_agent_data_buffer(num_steps=num_steps ,env=env ,device=device,env_name=env_name)
    
    #get an initial state from the environment 
    next_obs=information['current_state']
    done = information['prev_done']

    for step in range(0, num_steps):

        obs, prev_done = next_obs, done

        with torch.no_grad():
            action, logprob, _ = agent.get_action(obs.unsqueeze(0))  

        #execute the action and get environment response.
        next_obs, e_reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())  
        done= torch.max(torch.tensor([terminated,truncated],dtype=torch.float32))

        #preprocess and store data 
        e_reward= torch.as_tensor(e_reward,dtype=torch.float32).to(device)
        lifetime_buffer.store_step_data(global_step=current_lifetime_step , obs=obs.to(device), act=action.to(device),
                            e_reward=e_reward.to(device), logp=logprob.to(device),prev_done=prev_done.to(device))
            
        #extrinsic sparse reward signal
        if info['success'] == 1.0 and succeded_in_episode==False:
            success_signal= 1.0 - 0.7*(episode_step_num/max_episode_steps)
        elif episode_step_num==(max_episode_steps-1) and succeded_in_episode==False:
            success_signal= -0.2
        else:
            success_signal=0.0
        lifetime_buffer.success_signal[current_lifetime_step]=success_signal


        if intrinsic_rewards==True:
            #-------intrinsic reward agent involvement ---------------
            
            #get an encoding of the whole agent's lifetime up to this point
            if current_lifetime_step==0:
                hidden_state=reward_agent.initialize_state(batch_size=1)
                if isinstance(hidden_state, tuple):
                    hidden_state = tuple(hs.to(device) for hs in hidden_state)
                else:
                    hidden_state.to(device)

            hidden_state=reward_agent.rnn_next_state(lifetime_buffer ,lifetime_timestep=current_lifetime_step ,
                                                rnn_current_state=hidden_state)  #(1,1,hidden_state_size)
            
            #compute reward agent predictions conditioning on the lifetime history
            if training==True:
                with torch.no_grad():
                    meta_value=reward_agent.get_value(hidden_state).squeeze(0).squeeze(0) #(1)
                    i_reward, logp_i_reward , _ = reward_agent.get_action(hidden_state)
                    i_reward=i_reward.squeeze(0).squeeze(0)  #(1)
                    logp_i_reward=logp_i_reward.squeeze(0).squeeze(0) #(1)
            elif training==False:
                with torch.no_grad():
                    meta_value=torch.ones(1) #(1)
                    i_reward= reward_agent.get_deterministic_action(hidden_state)
                    i_reward=i_reward.squeeze(0).squeeze(0)  #(1)
                    logp_i_reward=torch.zeros(1) #(1)

            # store the predictions from the outer loop agent
            lifetime_buffer.store_outer_loop_predictions(global_step=current_lifetime_step, i_reward=i_reward,
                                                        logp_i_reward=logp_i_reward,meta_value=meta_value)
            i_reward=i_reward.to(device)
            #-----------------------------------------------

        if intrinsic_rewards==False:
            #If not using intrinsic rewards then the buffer that collects data for adaptation should store the extrinsic shaped or sparse rewards 
            #which is determined by the value of shaped_rewards_available_at_test_time 
            if config.shaped_rewards_available_at_test_time==True:
                i_reward=e_reward
            elif config.shaped_rewards_available_at_test_time==False:
                i_reward=success_signal
        if for_maml_loss==True:
            # If collecting data after adaptation to evaluate adapted parameters performance , then the stored rewards should be the extrinsic shaped or sparse rewards
            # which is determined by the value of shaped_rewards_available_at_train_time.  
            if config.shaped_rewards_available_at_train_time==True:
                i_reward=e_reward
            elif config.shaped_rewards_available_at_train_time==False:
                i_reward=success_signal

        actionA_buffer.store_inner_loop_update_data(step_index=step, obs=obs, act=action, reward=i_reward
                                                    , logp=logprob,prev_done=prev_done)
 

        #prepare for next step
        next_obs, done =torch.as_tensor(next_obs,dtype=torch.float32).to(device) ,torch.as_tensor(done,dtype=torch.float32).to(device)
        current_lifetime_step+=1
        episode_step_num+=1
        episode_return+= e_reward
        if info['success'] == 1.0: 
            succeded_in_episode=True 



        #deal with the case where the episode ends 
        if episode_step_num==max_episode_steps :
            episodes_returns.append(episode_return)
            if succeded_in_episode==True:
                episodes_successes.append(1.0) 
            else:
                episodes_successes.append(0.0)
            done=torch.ones(1).to(device) 
            next_obs = torch.tensor(env.reset()[0],dtype=torch.float32).to(device) 
            episode_step_num=0
            episode_return=0
            succeded_in_episode=False

                
    if not for_maml_loss:
        with torch.no_grad():
            #calculate the advantages and to go returns for each state visited in the data. 
            actionA_buffer.calculate_returns_and_advantages(mean_reward=mean_reward_for_baseline,gamma=config.adaptation_gamma,)

    lifetime_buffer.episodes_returns=lifetime_buffer.episodes_returns+episodes_returns
    lifetime_buffer.episodes_successes =lifetime_buffer.episodes_successes+ episodes_successes 

    information={'current_state':next_obs , 'prev_done' :done  ,
                'current_episode_step_num':episode_step_num , 'current_episode_success':succeded_in_episode ,'current_episode_return':episode_return,
                'current_lifetime_step':current_lifetime_step ,'hidden_state':hidden_state}
    
    return actionA_buffer , information



#----



class Action_agent_data_buffer:
    def __init__(self,num_steps ,env,device,env_name='none'):
        '''buffer for storing the data needed for an adaptation update'''
        self.observations= torch.zeros( (num_steps , env.observation_space.shape[0])).to(device)
        self.actions = torch.zeros((num_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions = torch.zeros((num_steps)).to(device)
        self.rewards = torch.zeros((num_steps)).to(device)
        self.dones = torch.zeros((num_steps)).to(device)
        self.advantages = torch.zeros((num_steps)).to(device)
        self.returns = torch.zeros((num_steps)).to(device)

        self.batch_size = int( num_steps)
        self.num_steps=num_steps
        self.env_name=env_name


    def store_inner_loop_update_data(self,step_index, obs, act, reward, logp,prev_done):
        self.observations[step_index]=obs
        self.actions[step_index]=act
        self.logprob_actions[step_index]=logp
        self.rewards[step_index]=reward
        self.dones[step_index]=prev_done


    def preprocess_data(self, data_stats , objective_mean):
        ''' Normalizes rewards using environment dependant statistics. It multiplies the e_rewards by a factor that makes the mean equal to objective_mean
        Args:
            data_stats: An object that keeps track of the mean extrinsic reward given by each environment type
            objective_mean : 
        '''
        if f'{self.env_name}' in data_stats.e_rewards_means.keys():
            self.rewards= (self.rewards/ (data_stats.e_rewards_means[f'{self.env_name}'] +1e-7)) *objective_mean


    def calculate_returns_and_advantages(self, mean_reward=None , gamma=0.95):
        '''calculate an advantage estimate and a return to go estimate for each state in the batch .
          It estimates it using montecarlo and adds a baseline that is calculated using a measure of the mean reward the agent receives at each step  '''
        baseline=torch.zeros((self.num_steps))

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_baseline=0
                nextnonterminal=0
                next_return=0
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                next_return = self.returns[t + 1]
                next_baseline=baseline[t+1]

            baseline[t]= mean_reward + gamma *  nextnonterminal * next_baseline
            self.returns[t] = self.rewards[t] + gamma * nextnonterminal * next_return
        
        self.advantages = self.returns - baseline



############################################################################


#################---------   ADAPTATION LOSS  ---------###############

def policy_loss_for_adaptation(agent,buffer):
    '''Computes loss for the adaptation steps in the inner loops  '''

    _ , newlogprob, entropy = agent.get_action(buffer.observations, buffer.actions) 

    normalize_advantage=True
    if normalize_advantage==True:
        buffer.advantages= (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    pg_loss = -(newlogprob * buffer.advantages).mean()

    return pg_loss



############################################################################


#################---------   MAML AGENT NEURAL NET  ---------###############



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Action_agent(nn.Module):
    def __init__(self, env):
        super(Action_agent, self).__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, np.prod(env.action_space.shape)), std=0.01),
        )

        self.actor_logstd=nn.Sequential(
            layer_init(nn.Linear(np.prod(env.observation_space.shape), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=0.01) )

    def get_action(self, x, action=None ,return_distribution=False):
        action_mean = self.actor_mean(x)

        action_logstd= self.actor_logstd(x)
        action_std = torch.exp(action_logstd )
        distribution = Normal(action_mean, action_std)
        if action is None:
            action= distribution.sample() 
            logprob= distribution.log_prob(action).sum(1)
        else:
            logprob= distribution.log_prob(action).sum(1)

        if not return_distribution:
            return action, logprob, distribution.entropy().sum(1) 
        elif return_distribution==True:
            return action, logprob, distribution.entropy().sum(1),  distribution
        

    def get_deterministic_action(self,x):
        action_mean = self.actor_mean(x)
        return action_mean
    




############################################################################


#################---------   MAML AGENT TRPO UPDATE  ---------###############
    

def combine_gradients(gradient_list):
    """Combines gradients from a list by averaging them for each parameter.

    Args:
        gradient_list: A list of gradients, where each element is a list of
            tensors containing gradients for a set of parameters.

    Returns:
        A list of tensors containing the combined gradients, where each tensor
        has the same shape as the corresponding tensor in the input gradients.
    """

    combined_gradients = []

    for gradients_per_param in zip(*gradient_list):
        combined_gradient = torch.mean(torch.stack(gradients_per_param), dim=0)
        combined_gradients.append(combined_gradient)

    return combined_gradients

#----

def surrogate_loss(agent,buffer,old_distribution=None,logs_dict=None):
    

    _ , newlogprob, entropy, distribution = agent.get_action(buffer.observations, buffer.actions ,return_distribution=True) 

    normalize_advantage=True
    if normalize_advantage==True:
        buffer.advantages= (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

    if old_distribution==None:
        if isinstance(distribution, Normal):
            old_distribution = Normal(loc=distribution.loc.detach(), scale=distribution.scale.detach())


    logratio = newlogprob - old_distribution.log_prob(buffer.actions).sum(1)
    ratio = logratio.exp()

    loss=  -(ratio * buffer.advantages).mean()

    kl=torch.distributions.kl.kl_divergence(distribution, old_distribution).mean()

    if logs_dict:
        logs_dict['policy gradient loss'].append(loss.item())
        logs_dict['entropy'].append(entropy.mean().item())
        logs_dict['aprox KL'].append(kl.item())
        
    return loss ,kl ,old_distribution


#----

def hessian_vector_product(kls, adapted_policies_states, maml_agent, damping=1e-2):

    kls=kls
    adapted_policies_states=adapted_policies_states
    maml_agent=maml_agent
    
    def _hv_product(vector,retain_graph=True):
        grads_grad_kl_v=[]
        for kl, adapted_policy_states in zip(kls,adapted_policies_states):
            torchopt.recover_state_dict(maml_agent, adapted_policy_states)

            kl_grad = torch.autograd.grad(kl, maml_agent.parameters(),
                                                create_graph=True)
            flat_kl_grad = parameters_to_vector(kl_grad)
            
            #_product
            grad_kl_v = torch.dot(flat_kl_grad, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                            maml_agent.parameters(),
                                            retain_graph=retain_graph)
            grads_grad_kl_v.append(grad2s)

        grad_grad_kl_v = combine_gradients(grads_grad_kl_v)
        flat_grads_grad_kl_v = parameters_to_vector( grad_grad_kl_v )

        return flat_grads_grad_kl_v + damping * vector
    
    return _hv_product


#----

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()


#-----

def maml_trpo_update(maml_agent, data_buffers, adapted_policies_states,  config, logging=True):

    base_policy_state_dict = torchopt.extract_state_dict(maml_agent) #extract the state of the base maml agent (before being adapted to a specific env)

    logs_dict={'policy gradient loss':[] ,'entropy' :[]  ,'aprox KL':[]}

    old_loss=0
    kls=[]
    surrogate_loss_gradients=[]
    old_distributions=[]

    #obtain surrogate losses (loss of adapted policies) , the kl divergences and save the probability distribution over actions of each adapted policy for the states in the collected data
    for buffer ,adapted_policy_state in zip(data_buffers ,adapted_policies_states):
        torchopt.recover_state_dict(maml_agent, adapted_policy_state)
        
        adapted_policy_losss , kl ,distribution = surrogate_loss(agent=maml_agent,buffer=buffer) 

        surrogate_loss_gradients.append( torch.autograd.grad(adapted_policy_losss,
                                            maml_agent.parameters(),
                                            retain_graph=True))

        old_loss+=adapted_policy_losss
        kls.append(kl)
        old_distributions.append(distribution)
    old_loss=old_loss/len(data_buffers)

    policy_loss_grads=combine_gradients(surrogate_loss_gradients)
    flat_policy_loss_grads=parameters_to_vector(policy_loss_grads)

    #save old parameters of base policy
    torchopt.recover_state_dict(maml_agent, base_policy_state_dict)
    old_base_params = parameters_to_vector(maml_agent.parameters())
    # Save the old parameters of each adapted policy
    old_params=[]
    for adapted_policy_state in adapted_policies_states:
        torchopt.recover_state_dict(maml_agent, adapted_policy_state)
        vec_params = parameters_to_vector(maml_agent.parameters())
        old_params.append(vec_params)
    

    #compute the direction of the update with conjugate gradient
    hess_vec_product = hessian_vector_product(kls,adapted_policies_states,maml_agent,damping=config.maml_TRPO["cg_damping"])
    stepdir = conjugate_gradient(hess_vec_product,
                                        flat_policy_loss_grads,
                                        cg_iters=config.maml_TRPO["cg_iters"])

    # Compute the Lagrange multiplier
    shs = 0.5 * torch.dot(stepdir,
                            hess_vec_product(stepdir,retain_graph=False))
    lagrange_multiplier = torch.sqrt(shs / config.maml_TRPO['max_kl'])

    step = stepdir / lagrange_multiplier




    ## Line search to find how much the parameters of the base policy are moved in the update direction
    with torch.no_grad():
        step_size = 3.0

        line_search_succeeded=False

        for _ in range(config.maml_TRPO['line_search_max_steps']):
            surrogate_policy_loss=0
            kl=0

            for old_adapted_policy_params,buffer,old_distribution in zip(old_params ,data_buffers, old_distributions):

                updated_vec_params= old_adapted_policy_params - step_size * step
                
                vector_to_parameters(updated_vec_params , maml_agent.parameters())

                s_loss , kl_div ,_= surrogate_loss(buffer=buffer,agent=maml_agent,old_distribution=old_distribution ,logs_dict=logs_dict)

                surrogate_policy_loss+=s_loss
                kl+=kl_div

            surrogate_policy_loss=surrogate_policy_loss/len(data_buffers)
            kl=kl/len(data_buffers)

            # Check if the proposed update satisfies the constraints that
            #  we improve with respect to the surrogate policy objective while also staying close enough (in term of kl div) to the old policy
            if (surrogate_policy_loss<old_loss) and (kl.item()<config.maml_TRPO['max_kl']):
                line_search_succeeded=True
                updated_vec_params= old_base_params - step_size * step
                vector_to_parameters(updated_vec_params , maml_agent.parameters())
                break
            
            else:
                ## Reduce step size if line-search wasn't successful
                step_size *= config.maml_TRPO["line_search_backtrack_ratio"]
                logs_dict={'policy gradient loss':[] ,'entropy' :[]  ,'aprox KL':[]}

        if line_search_succeeded==False:
            # If the line-search wasn't successful we revert to the original parameters
            vector_to_parameters(old_base_params, maml_agent.parameters())
        

    if logging==True:
        wandb.log({ 'maml policy gradient loss': np.array(logs_dict['policy gradient loss']).mean(),
                    'maml entropy ': np.array(logs_dict['entropy']).mean(),
                    'maml aprox KL ' :  np.array(logs_dict['aprox KL']).mean() } , commit=False )





