import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


####----------BUFFER FOR COLLECTING ALL THE LIFETIME'S DATA ---------------------------


class Lifetime_buffer:
    def __init__(self,num_lifetime_steps ,il_batch_size, env, device ,env_name='none'):
        '''class for storing all the data the agent collects throughout an inner loop . It is used in outer loop updates'''

        #data that is useed for training in the inner loop
        self.observations=torch.zeros( (num_lifetime_steps , env.observation_space.shape[0])).to(device)
        self.actions= torch.zeros((num_lifetime_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions= torch.zeros((num_lifetime_steps)).to(device)
        self.intrinsic_rewards= torch.zeros((num_lifetime_steps)).to(device)
        self.dones= torch.zeros((num_lifetime_steps)).to(device)

        #Also store data that is not needed for the inner loop but is needed to update the outer loop
        self.extrinsic_rewards= torch.zeros((num_lifetime_steps)).to(device)
        self.success_signal = torch.zeros((num_lifetime_steps)).to(device)
        self.logprob_i_rewards= torch.zeros((num_lifetime_steps)).to(device)
        self.meta_values= torch.zeros((num_lifetime_steps)).to(device)
        #extrinsic_rewards[i] is the true reward the agent got for doing action[i] from observation[i] by the environment
        #success_signal[i] informs wether the agent succeded in completing the task in step i. In case it did it has 1 - 0.7 * (episode_step_num / max_steps_in_episode)
        #logprob_i_rewards[i] stores the log of the probability of the intrinsic reward given by the reward agent to the action agent at step i
        #meta_values[i] stores an estimate by the reward agent of the expected meta-return to go given the training history up to the info in step i

        self.meta_advantages=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_returns_to_go=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_return=0 #stores the lifetimes total meta discounted return


        self.device=device
        self.il_batch_size=il_batch_size #batch_size used in the inner loop
        self.num_lifetime_steps=num_lifetime_steps
        self.current_row=0
        self.episodes_returns=[] #list that contains the returns of each episode in the lifetime
        self.episodes_successes=[] #list that contains wether each episode in the lifetime succeded in completing the task 
  
        self.env_name=env_name

    def store_step_data(self,global_step, obs, act,e_reward, logp,prev_done):
        '''stores the data collected in a step of a standard RL algorithm'''
        self.observations[global_step]=obs.to(self.device)
        self.actions[global_step]=act.to(self.device)
        self.logprob_actions[global_step]=logp.to(self.device)
        self.dones[global_step]=prev_done.to(self.device)
        self.extrinsic_rewards[global_step]=e_reward.to(self.device)

    def store_outer_loop_predictions(self,global_step, i_reward,logp_i_reward,meta_value):
        '''stores data that is only obtained due to the outer loop agent'''
        self.intrinsic_rewards[global_step]=i_reward
        self.logprob_i_rewards[global_step]=logp_i_reward
        self.meta_values[global_step]=meta_value


    def preprocess_data(self, data_stats , objective_mean):
        ''' Normalizes extrinsic rewards using environment dependant statistics. It multiplies the e_rewards by a factor that makes the mean equal to objective_mean
        Args:
            data_statistics : An object that keeps track of the mean extrinsic reward given by each environment type
            objective_mean : 
        '''
        self.normalized_e_rewards= (self.extrinsic_rewards.clone().detach() / (data_stats.e_rewards_means[f'{self.env_name}'] +1e-7)) *objective_mean

    def compute_meta_advantages_and_returns_to_go(self,sparse_or_shaped='shaped',gamma=0.98, estimation_method='bootstrapping' , e_lambda=0.95 ,starting_n=500 ,num_to_average_over=10 ,skip_rate=2 ):
        '''calculate an advantage estimate and a return to go estimate for each meta state .gamma indicates the episodic discounting (discounting is applied per episode , not per step ,except with estimation_method='standard GAE ).
         There are Several options:
        1) use the montecarlo estimate. select estimation_method='MC' for this.
        2) use a combination of several n-step estimates , select estimation_method='bootstrapping'for this .
             The scheme used for GAE cant be used when considering episodic (instead of stepwise) discounting. Several parameters of the estimation can be selected : 
                a) num_to_average_over: the number of n-step estimates to average over . ( Complexity of the estimation method is O(num_lifetime_steps*num_to_average_over) ) 
                b) skip rate to consider one every skip_rate estimates. ej if skip_rate=3 it would average over the 1,4,7,10-step estimates
                c) starting_n: gives the n at which to start with the n step estimates . if starting_n=10 and skip_rate=3 it would average over the 10,13,16,..-step estimates .  ( for steps where the chosen value lies beyond the lifes horizon it defaults back to 1)
                d) e-lambda: controls the decay factor in the exponential average of the estimates
        3) Both of the mentiones estimation methods also have a 'skipping uninfluenced rewards' version . This versions dont take into account the rewards received from the environment
           until the actions of the meta agent had any influence in the actions taken

        sparse_or_shaped: controls the objective to which state's values ,returns_to_go and advantages are calculated with respect to. The options are: 
        - 'shaped' to calculate them with respect to shaped extrinsic rewards that the inner loop agent receives throughout a lifetime and
        - 'sparse ' to calculate them with respect to a sparse signal of successes that the inner loop agent receives throughout a lifetime'''
 
        if sparse_or_shaped=='shaped':
            objective_rewards=self.normalized_e_rewards
        elif sparse_or_shaped=='sparse':
            objective_rewards=self.success_signal

        episode_numbers= torch.cumsum(self.dones,dim=0) #stores for each step to which episode it belongs starting at zero

        #get montecarlo estimate of the meta returns to go for each state and use it to also obtain an advantage estimate
        if estimation_method=='MC':
            discounted_rewards= objective_rewards* (gamma ** episode_numbers)
            discounted_meta_returns_to_go = torch.cumsum(discounted_rewards.flip(0), dim=0).flip(0)
            self.meta_returns_to_go= discounted_meta_returns_to_go / (gamma**episode_numbers) #this line corrects so that the disounting starts anew from each state's returns_to_go

            self.meta_advantages = self.meta_returns_to_go - self.meta_values
        

        ###
        elif estimation_method=='MC skipping uninfluenced future rewards':
            discounted_rewards= objective_rewards* (gamma ** episode_numbers)
            discounted_meta_returns_to_go = torch.cumsum(discounted_rewards.flip(0), dim=0).flip(0)
            self.meta_returns_to_go= discounted_meta_returns_to_go / (gamma**episode_numbers) #this line corrects so that the disounting starts anew from each state's returns_to_go
            for t in range(self.num_lifetime_steps-1):
                steps_to_next_il_update= self.il_batch_size - (t%self.il_batch_size)
                if (t+steps_to_next_il_update)< self.num_lifetime_steps:
                    self.meta_returns_to_go[t]= self.meta_returns_to_go[t+steps_to_next_il_update]

            self.meta_advantages = self.meta_returns_to_go - self.meta_values
        

        #estimate the advantage for each state visited using a combination of n-step estimates .
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

        #estimate the advantage for each state visited using a combination of n-step estimates . Also when estimating the value of a state it ignores future rewards that were received before the current action could have had any effect (i_rewards give only influence after an update of the il_agent takes place)
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
        

        #Finally , compute the lifetimes meta discounted return (not necessary , just for having the metric)  
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
        '''calculate an advantage estimate and a return to go estimate for each state using gae. 
        '''
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
        '''buffer for storing the data needed for an inner loop update'''
        #Initialize space for storing data for a batch.
        self.observations= torch.zeros( (num_steps , env.observation_space.shape[0])).to(device)
        self.actions = torch.zeros((num_steps, env.action_space.shape[0])).to(device)
        self.logprob_actions = torch.zeros((num_steps)).to(device)
        self.rewards = torch.zeros((num_steps)).to(device)
        self.dones = torch.zeros((num_steps)).to(device)
        self.values = torch.zeros((num_steps)).to(device)

        # the advantages and to go returns can be computed from the previous data
        self.advantages = torch.zeros((num_steps)).to(device)
        self.returns = torch.zeros((num_steps)).to(device)

        #observations[i] stores the state at which the agent was in step i
        #action[i] stores the action the agent took in step i
        #logprob_actions[i] stores 
        #rewards[i] stores the reward the agent got for doing action[i] from observation[i] by the reward agent
        #values[i] stores the state value estimate the agent estimated for state obsrvation[i]
        #Dones[i] stores whether when taking step[i-1] the env was terminated or tuncated.
        #in other words, it says wether the env was reset before step i. In which case observation[i] is the first obs from the new episode

        self.batch_size = int( num_steps)
        self.num_steps=num_steps

    def store_inner_loop_update_data(self,step_index, obs, act, reward, val, logp,prev_done):
        '''stores the data collected in a step that is needed to make an update by the inner loop algoritm'''
        self.observations[step_index]=obs
        self.actions[step_index]=act
        self.logprob_actions[step_index]=logp
        self.rewards[step_index]=reward
        self.values[step_index]=val
        self.dones[step_index]=prev_done


    def calculate_returns_and_advantages(self, done,next_value,gae,gamma=0.95,gae_lambda=0.5):
        '''calculate an advantage estimate and a return to go estimate for each state 
        in the batch of observed data using gae or montecarlo. 
        
        gae: wether to use gae or monte carlo
        done:  wether the episode was 'done' after taking the last step 
        next_value: The state value esimate of the observed state after taking the last action 

        '''
        #calculation if GAE is used
        if gae==True:
            #calculate advantage estimation
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            #Estimate returns to go
            self.returns = self.advantages + self.values

        #Calculation if using monte carlo
        else:
            #calculate returns to go as the montecarlo sample (but bootstraping when last step taken in data collected didnt reach an end of episode)
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    next_return = self.returns[t + 1]
                self.returns[t] = self.rewards[t] + gamma * nextnonterminal * next_return
            # obtain advantage estimation using the returns to go
            self.advantages = self.returns - self.values







#-------------------------------------------

#----------------- AGENT--------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Inner_loop_action_agent(nn.Module):
    def __init__(self, env):
        super(Inner_loop_action_agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.action_space.shape)), std=0.01),
        )

        initial_std = 1.0
        self.actor_logstd = nn.Parameter(torch.full((np.prod(env.action_space.shape),),  np.log(initial_std)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd )
        distribution = Normal(action_mean, action_std)
        if action is None:
            action= distribution.sample() 
        logprob= distribution.log_prob(action).sum(1)
 
        return action, logprob, distribution.entropy().sum(1), self.critic(x)

    def get_deterministic_action(self,x):
        action_mean = self.actor_mean(x)
        return action_mean


#-------------------------------------------------------------

#--------------------   UPDATE METHOD ------------------------

class PPO_class:
    
    def get_data_from_buffer(self,buffer,env):
        '''retrieves from a buffer all the batch of data it needs to perform an update
        Note that before calling it, all the batch should have been filled and the returns
        and advantages calculated. Also have to pass the env on which the buffer is collecting data'''
        self.b_obs = buffer.observations.reshape((-1, env.observation_space.shape[0]))
        self.b_logprob_actions = buffer.logprob_actions.reshape(-1)
        self.b_actions = buffer.actions.reshape((-1,env.action_space.shape[0]))
        self.b_advantages = buffer.advantages.reshape(-1)
        self.b_returns = buffer.returns.reshape(-1)
        self.b_values = buffer.values.reshape(-1) 

    def update(self,agent,optimizer,update_epochs=10,num_minibatches=32,normalize_advantage=True,clip_vloss=False,entropy_coef=0,valuef_coef=0.5,clip_grad_norm=True,max_grad_norm=0.5,target_KL=None,clip_coef=0.2 ):
        '''for update_epochs iterations: generate random minibatches from the batch of collected data
        and do a PPO update with each of this minibatches. If after an epoch the approximate KL divergence between the
        updated policy and the policy before starting updating goes beyond a treshold, the update terminates.'''
        
        batch_size= len(self.b_values)
        minibatch_size=int(batch_size//num_minibatches)
        b_indices = np.arange(batch_size)
        
        for epoch in range(update_epochs):
            np.random.shuffle(b_indices)
            for start in range(0,  batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = b_indices[start:end]

                #----calculate policy gradient loss for the minimabtch of data-----#
                #calculate necessary values with current new policy
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(self.b_obs[mb_indices], self.b_actions[mb_indices])
                
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


                #------- Value loss for minibatch of data----#
                newvalue = newvalue.view(-1)
                if clip_vloss==True:
                    # value loss with clipping. 
                    v_loss_unclipped = (newvalue - self.b_returns[mb_indices]) ** 2
                    v_clipped = self.b_values[mb_indices] + torch.clamp(
                        newvalue - self.b_values[mb_indices],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.b_returns[mb_indices]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    #standard value loss calculation
                    v_loss = 0.5 * ((newvalue - self.b_returns[mb_indices]) ** 2).mean()

                #----entropy loss---
                entropy_loss = entropy.mean()

                ##Combine de previous computations to create the total loss and backpropagate it
                loss = pg_loss - entropy_coef * entropy_loss + v_loss * valuef_coef

                optimizer.zero_grad()
                loss.backward()
                if clip_grad_norm:
                    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_KL is not None:
                if approx_kl > target_KL:
                    break
      



