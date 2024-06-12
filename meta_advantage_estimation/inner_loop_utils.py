import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import time
####----------BUFFER FOR COLLECTING ALL THE LIFETIME'S DATA ---------------------------


class Lifetime_buffer:
    def __init__(self,num_lifetime_steps ,il_batch_size, env, device ,env_name='none'):

        self.observations=torch.zeros( (num_lifetime_steps , env.observation_space.shape[0])).to(device)
        self.actions= torch.zeros((num_lifetime_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions= torch.zeros((num_lifetime_steps)).to(device)
        self.advantages= torch.zeros((num_lifetime_steps)).to(device)
        self.dones= torch.zeros((num_lifetime_steps)).to(device)

        self.extrinsic_rewards= torch.zeros((num_lifetime_steps)).to(device)
        self.success_signal = torch.zeros((num_lifetime_steps)).to(device)
        self.logprob_advantages= torch.zeros((num_lifetime_steps)).to(device)
        self.meta_values= torch.zeros((num_lifetime_steps)).to(device)

        self.meta_advantages=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_returns_to_go=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_return=0 


        self.device=device
        self.il_batch_size=il_batch_size 
        self.num_lifetime_steps=num_lifetime_steps
        self.current_row=0
        self.episodes_returns=[]
        self.episodes_successes=[] 
  
        self.env_name=env_name

    def store_step_data(self,global_step, obs, act,e_reward, logp,prev_done):
        self.observations[global_step]=obs.to(self.device)
        self.actions[global_step]=act.to(self.device)
        self.logprob_actions[global_step]=logp.to(self.device)
        self.dones[global_step]=prev_done.to(self.device)
        self.extrinsic_rewards[global_step]=e_reward.to(self.device)

    def store_outer_loop_predictions(self,global_step, advantage ,logp_advantage ,meta_value):
        self.advantages[global_step]=advantage
        self.logprob_advantages[global_step]=logp_advantage
        self.meta_values[global_step]=meta_value


    def preprocess_data(self, data_stats , objective_mean):

        self.normalized_e_rewards= (self.extrinsic_rewards.clone().detach() / (data_stats.e_rewards_means[f'{self.env_name}'] +1e-7)) *objective_mean

    def compute_meta_advantages_and_returns_to_go(self,sparse_or_shaped='shaped',gamma=0.98, estimation_method='bootstrapping' , e_lambda=0.95 ,starting_n=500 ,num_to_average_over=10 ,skip_rate=2 ):

        if sparse_or_shaped=='shaped':
            objective_rewards=self.normalized_e_rewards
        elif sparse_or_shaped=='sparse':
            objective_rewards=self.success_signal

        episode_numbers= torch.cumsum(self.dones,dim=0) 
    
        if estimation_method=='MC':
            discounted_rewards= objective_rewards* (gamma ** episode_numbers)
            discounted_meta_returns_to_go = torch.cumsum(discounted_rewards.flip(0), dim=0).flip(0)
            self.meta_returns_to_go= discounted_meta_returns_to_go / (gamma**episode_numbers) #this line corrects so that the disounting starts anew from each state's returns_to_go
            
            self.meta_advantages = self.meta_returns_to_go - self.meta_values
        

        elif estimation_method=='MC skipping uninfluenced future rewards':
            discounted_rewards= objective_rewards* (gamma ** episode_numbers)
            discounted_meta_returns_to_go = torch.cumsum(discounted_rewards.flip(0), dim=0).flip(0)
            self.meta_returns_to_go= discounted_meta_returns_to_go / (gamma**episode_numbers) #this line corrects so that the disounting starts anew from each state's returns_to_go
            for t in range(self.num_lifetime_steps-1):
                steps_to_next_il_update= self.il_batch_size - (t%self.il_batch_size)
                if (t+steps_to_next_il_update)< self.num_lifetime_steps:
                    self.meta_returns_to_go[t]= self.meta_returns_to_go[t+steps_to_next_il_update]

            self.meta_advantages = self.meta_returns_to_go - self.meta_values
        

        elif estimation_method=='bootstrapping':
            ns= [n*skip_rate+starting_n for n in range(num_to_average_over)] #n's to be considered for calculating n-step advantage estimates
            ns2= [n*skip_rate+1 for n in range(num_to_average_over)] # the same but is used in the case where all n's in ns lie beyond the horizon of the data
            self.meta_advantages[self.num_lifetime_steps-1] = objective_rewards[self.num_lifetime_steps-1]-self.meta_values[self.num_lifetime_steps-1] # there is no n-step bootstrapped estimation possible for last step
            for t in range(self.num_lifetime_steps-1):
                advantage_estimates=[ self._n_step_advantage_estimate(objective_rewards,t,n,gamma,episode_numbers).item() for n in ns if (t+n)<self.num_lifetime_steps]
                if len(advantage_estimates)==0:
                    advantage_estimates=[ self._n_step_advantage_estimate(objective_rewards,t,n,gamma,episode_numbers).item() for n in ns2 if (t+n)<self.num_lifetime_steps] 
                self.meta_advantages[t]=self._exponentially_weighted_average(advantage_estimates,e_lambda)

            self.meta_returns_to_go= self.meta_advantages + self.meta_values

        elif estimation_method=='bootstrapping skipping uninfluenced future rewards':
            ns= [n*skip_rate+starting_n for n in range(num_to_average_over)] #n's to be considered for calculating n-step advantage estimates
            ns2= [n*skip_rate+1 for n in range(num_to_average_over)] # the same but is used in the case where all n's in ns lie beyond the horizon of the data
            self.meta_advantages[self.num_lifetime_steps-1] = objective_rewards[self.num_lifetime_steps-1]-self.meta_values[self.num_lifetime_steps-1] # there is no n-step bootstrapped estimation possible for last step
            for t in range(self.num_lifetime_steps-1):
                steps_to_next_il_update= self.il_batch_size - (t%self.il_batch_size)
                advantage_estimates=[ self._n_step_advantage_estimate_skipping(objective_rewards,t,n,gamma,episode_numbers).item() for n in ns if (t+n)<self.num_lifetime_steps and (steps_to_next_il_update<n)]
                if len(advantage_estimates)==0:
                    advantage_estimates=[ self._n_step_advantage_estimate_skipping(objective_rewards,t,n,gamma,episode_numbers).item() for n in ns2 if ((t+n)<self.num_lifetime_steps) and (steps_to_next_il_update<n)]
                self.meta_advantages[t]=self._exponentially_weighted_average(advantage_estimates,e_lambda)

            self.meta_returns_to_go= self.meta_advantages + self.meta_values
        

        elif estimation_method=='standard GAE':
            self.calculate_returns_and_advantages_with_standard_GAE(objective_rewards=objective_rewards,gamma=gamma,gae_lambda=0.99)
        
        ep_nums=np.arange(len(self.episodes_returns))
        meta_return= self.episodes_returns * (gamma**ep_nums)
        self.meta_return= np.sum(meta_return)

    def _n_step_advantage_estimate(self,objective_rewards,t,n,gamma,episode_numbers):
        returns_up_to_t_plus_n= torch.sum( objective_rewards[t:t+n] * (gamma**(episode_numbers[t:t+n]-episode_numbers[t]))  )
        return returns_up_to_t_plus_n + (gamma**(episode_numbers[t+n]-episode_numbers[t]))* self.meta_values[t+n] -self.meta_values[t]
    
    def _n_step_advantage_estimate_skipping(self,objective_rewards,t,n,gamma,episode_numbers):
        steps_to_next_il_update= self.il_batch_size - (t%self.il_batch_size) #It is the number of steps that were remaining till the next inner loop update at timestep t. it is used so that the value of the state is only considered in terms of the rewards after this (since the intrinsic r given at timestep t has no effect on the e_rew obtained in between t and when the next update takes place)
        returns_up_to_t_plus_n= torch.sum( objective_rewards[t+steps_to_next_il_update:t+n] * (gamma**(episode_numbers[t+steps_to_next_il_update:t+n]-episode_numbers[t+steps_to_next_il_update]))  )
        return returns_up_to_t_plus_n + (gamma**(episode_numbers[t+n]-episode_numbers[t+steps_to_next_il_update]))* self.meta_values[t+n] -self.meta_values[t] 
       
    def _exponentially_weighted_average(self,values, alpha):

        values_tensor = torch.tensor(values, dtype=torch.float32)
        weights = torch.pow(alpha, torch.arange(len(values_tensor), dtype=torch.float32))
        weighted_average = torch.sum(values_tensor * weights) / torch.sum(weights)
        return weighted_average
    
    def calculate_returns_and_advantages_with_standard_GAE(self,objective_rewards,gamma=0.99,gae_lambda=0.95):

        lastgaelam = 0
        for t in reversed(range(self.num_lifetime_steps)):
            if t == self.num_lifetime_steps - 1:
                nextnonterminal = 0.0 
                nextvalue = 0.0 
            else:
                nextnonterminal = 1.0 
                nextvalue = self.meta_values[t + 1]
            delta = objective_rewards[t] + gamma * nextvalue * nextnonterminal - self.meta_values[t]
            self.meta_advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        #Estimate returns to go
        self.meta_returns_to_go = self.meta_advantages + self.meta_values



#---------------------------------------------------------------

#---------------BUFFER FOR INNER LOOP ALGORITHM ----------------

class IL_buffer:
    def __init__(self,num_steps ,env,device):

        self.observations= torch.zeros( (num_steps , env.observation_space.shape[0])).to(device)
        self.actions = torch.zeros((num_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions = torch.zeros((num_steps)).to(device)
        self.advantages = torch.zeros((num_steps)).to(device)


        self.batch_size = int( num_steps)
        self.num_steps=num_steps

    def store_inner_loop_update_data(self,step_index, obs, act, logp ,advantage):
        self.observations[step_index]=obs
        self.actions[step_index]=act
        self.logprob_actions[step_index]=logp
        self.advantages[step_index]=advantage



#-------------------------------------------

#----------------- AGENT--------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Inner_loop_action_agent(nn.Module):
    def __init__(self, env):
        super(Inner_loop_action_agent, self).__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )
        initial_std = 1.0
        self.actor_logstd = nn.Parameter(torch.full((np.prod(env.action_space.shape),),  np.log(initial_std)))

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd )
        distribution = Normal(action_mean, action_std)
        if action is None:
            action= distribution.sample() 
            logprob= distribution.log_prob(action).sum(1)
        else:
            logprob= distribution.log_prob(action).sum(1)
        return action, logprob, distribution.entropy().sum(1)

    def get_deterministic_action(self,x):
        action_mean = self.actor_mean(x)
        return action_mean



#-------------------------------------------------------------

#--------------------   UPDATE METHOD ------------------------

class PPO_class:

    def get_data_from_buffer(self,buffer,env):
 
        self.b_obs = buffer.observations.reshape((-1, env.observation_space.shape[0]))
        self.b_logprob_actions = buffer.logprob_actions.reshape(-1)
        self.b_actions = buffer.actions.reshape((-1,env.action_space.shape[0]))
        self.b_advantages = buffer.advantages.reshape(-1)

    def update(self,agent,optimizer,update_epochs=10,num_minibatches=32,normalize_advantage=True,entropy_coef=0,clip_grad_norm=True,max_grad_norm=0.5,target_KL=None,clip_coef=0.2 ):

        batch_size= len(self.b_logprob_actions)
        minibatch_size=int(batch_size//num_minibatches)
        b_indices = np.arange(batch_size)

        for epoch in range(update_epochs):
            np.random.shuffle(b_indices)
            for start in range(0,  batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = b_indices[start:end]

                #----calculate policy gradient loss for the minimabtch of data-----#
 
                _, newlogprob, entropy = agent.get_action(self.b_obs[mb_indices], self.b_actions[mb_indices])
                logratio = newlogprob - self.b_logprob_actions[mb_indices]
                ratio = logratio.exp()

                mb_advantages = self.b_advantages[mb_indices]
                if normalize_advantage==True:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() 

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean() 

                #----entropy loss---
                entropy_loss = entropy.mean()

                ##Combine de previous computations to create the total loss and backpropagate it
                loss = pg_loss - entropy_coef * entropy_loss 

                optimizer.zero_grad()
                loss.backward()
                if clip_grad_norm:
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_KL is not None:
                if approx_kl > target_KL:
                    break
        




