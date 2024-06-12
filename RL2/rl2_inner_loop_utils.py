import torch
import numpy as np

####----------BUFFER FOR COLLECTING ALL THE LIFETIME'S DATA ---------------------------


'''When deciding what to store at position t in the lifetime buffer there is are at least two options. Here it was decided to store at 
position t the last information needed for the agent to make the prediction a_t ; this simplifies how the inputs to the rnn at each timestep are obtained and
the making of predictions during running the inner loop . The tradeoff is that this means storing at position t information
about the timestep t-1 (like a_{t-1}) ; this makes the loss calculations and advantage estimation for the outer loop update a bit less easy to read because it means certain information
belonging to a certain timestep is displaced by one.

At position t it is stored: (s_t ,a_{t-1} , logprob_{t-1}, d_{t-1} ,r_{t-1}  , metaValue_t ,metaAdvantage_t ,metaReturnsToGo_t).
A dummy a,logprob,d,r is stored at position 0 and a dummy s is stored in the last position'''

class Lifetime_buffer:
    def __init__(self,num_lifetime_steps , env, device,env_name='none'):
        '''class for storing all the data the agent collects throughout an inner loop . It is used in outer loop updates'''

        self.observations=torch.zeros( (num_lifetime_steps+1 , env.observation_space.shape[0])).to(device)
        self.prev_actions= torch.zeros((num_lifetime_steps+1, env.action_space.shape[0])).to(device)
        self.prev_logprob_actions= torch.zeros((num_lifetime_steps+1)).to(device)
        self.prev_rewards= torch.zeros((num_lifetime_steps+1)).to(device)
        self.prev_success_signals= torch.zeros((num_lifetime_steps+1)).to(device)
        self.dones= torch.zeros((num_lifetime_steps+1)).to(device)
        #they have size num_lifetime_steps +1 because we actually want to store 1 extra step (refer to the discussion above)

        self.meta_values= torch.zeros((num_lifetime_steps)).to(device)

        #observations[i] stores the state at which the agent was in step i
        #prev_actions[i] stores the action the agent took in step i-1
        #prev_logprob_actions[i] stores the log probabilty prev_actions[i] had of being taken
        #prev_rewards[i] stores the reward the agent got for doing prev_actions[i] at observation[i-1] - the reward obtained in timestep i-1
        #prev_success_signals[i] informs wether the agent succeded in completing the task in step i-1. In case it did it has 1 - 0.7 * (episode_step_num / max_steps_in_episode)
        #meta_values[i] stores the state meta value estimate (an estimate of the expected meta-return to go) the agent estimated taking into account all lifetime up to and including observation[i]
        #Dones[i] stores whether when taking step[i-1] the env was terminated or tuncated.
        #in other words, it says wether the env was reset befor step i. In which case observation[i] is the first obs from the new episode

        self.meta_advantages=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_returns_to_go=torch.zeros((num_lifetime_steps)).to(device)
        self.meta_return=0 

        self.device=device

        self.num_lifetime_steps=num_lifetime_steps
        self.current_row=0
        self.episodes_returns=[] 
        self.episodes_successes=[] 

        self.env_name=env_name

    def store_step_data(self,global_step, obs, prev_act, prev_reward, prev_success_signal, prev_logp,prev_done):
        self.observations[global_step]=obs.to(self.device)
        self.prev_actions[global_step]=prev_act.to(self.device)
        self.prev_logprob_actions[global_step]=prev_logp.to(self.device)
        self.dones[global_step]=prev_done.to(self.device)
        self.prev_rewards[global_step]=prev_reward.to(self.device)
        self.prev_success_signals[global_step]=prev_success_signal.to(self.device)

    def store_meta_value(self,global_step, meta_value):
        self.meta_values[global_step]=meta_value
    def preprocess_data(self, data_stats , objective_mean):

        self.normalized_prev_rewards= (self.prev_rewards.clone().detach() / (data_stats.e_rewards_means[f'{self.env_name}'] +1e-7)) *objective_mean

    def compute_meta_advantages_and_returns_to_go(self,sparse_or_shaped='shaped',gamma=0.95,  estimation_method='bootstrapping'  ,e_lambda=0.95 ,starting_n=500 ,num_to_average_over=10 ,skip_rate=2):#, done,next_value,gae,gamma=0.95,gae_lambda=0.5)

        if sparse_or_shaped=='shaped':
            prev_objective_rewards=self.normalized_prev_rewards 
        elif sparse_or_shaped=='sparse':
            prev_objective_rewards=self.prev_success_signals
        episode_numbers= torch.cumsum(self.dones,dim=0) 
        if estimation_method=='MC':
            discounted_rewards= prev_objective_rewards[1:]* (gamma ** episode_numbers[:-1])
            discounted_meta_returns_to_go = torch.cumsum(discounted_rewards.flip(0), dim=0).flip(0)
            self.meta_returns_to_go= discounted_meta_returns_to_go / (gamma**episode_numbers[:-1]) 

            self.meta_advantages = self.meta_returns_to_go - self.meta_values
        
        elif estimation_method=='bootstrapping':
            ns= [n*skip_rate+starting_n for n in range(num_to_average_over)] 
            ns2= [n*skip_rate+1 for n in range(num_to_average_over)] 
            self.meta_advantages[self.num_lifetime_steps-1] = prev_objective_rewards[self.num_lifetime_steps]-self.meta_values[self.num_lifetime_steps-1] 
            for t in range(self.num_lifetime_steps-1):
                advantage_estimates=[ self._n_step_advantage_estimate(prev_objective_rewards,t,n,gamma,episode_numbers).item() for n in ns if (t+n)<self.num_lifetime_steps]
                if len(advantage_estimates)==0:
                    advantage_estimates=[ self._n_step_advantage_estimate(prev_objective_rewards,t,n,gamma,episode_numbers).item() for n in ns2 if (t+n)<self.num_lifetime_steps] 
                self.meta_advantages[t]=self._exponentially_weighted_average(advantage_estimates,e_lambda)
            self.meta_returns_to_go= self.meta_advantages + self.meta_values

        elif estimation_method=='standard GAE':
            self.calculate_returns_and_advantages_with_standard_GAE(prev_objective_rewards=prev_objective_rewards,gamma=gamma,gae_lambda=e_lambda)
        else:
            print('advantage estimation method not supported')

        
        ep_nums=np.arange(len(self.episodes_returns))
        meta_return= self.episodes_returns * (gamma**ep_nums)
        self.meta_return= np.sum(meta_return)


    def _n_step_advantage_estimate(self,prev_objective_rewards,t,n,gamma,episode_numbers):
        returns_up_to_t_plus_n= torch.sum( prev_objective_rewards[t+1:t+n+1] * (gamma**(episode_numbers[t:t+n]-episode_numbers[t]))  )
        return returns_up_to_t_plus_n + (gamma**(episode_numbers[t+n]-episode_numbers[t]))* self.meta_values[t+n] -self.meta_values[t]
    

    def _exponentially_weighted_average(self,values, alpha):
        values_tensor = torch.tensor(values, dtype=torch.float32)
        weights = torch.pow(alpha, torch.arange(len(values_tensor), dtype=torch.float32))
        weighted_average = torch.sum(values_tensor * weights) / torch.sum(weights)
        return weighted_average
                                            

    def calculate_returns_and_advantages_with_standard_GAE(self,prev_objective_rewards,gamma=0.99,gae_lambda=0.95):

        lastgaelam = 0
        for t in reversed(range(self.num_lifetime_steps)):
            if t == self.num_lifetime_steps - 1:
                nextnonterminal = 0.0 
                nextvalue = 0.0
            else:
                nextnonterminal = 1.0 
                nextvalue = self.meta_values[t + 1]
            delta = prev_objective_rewards[t+1] + gamma * nextvalue * nextnonterminal - self.meta_values[t]
            self.meta_advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        self.meta_returns_to_go = self.meta_advantages + self.meta_values

