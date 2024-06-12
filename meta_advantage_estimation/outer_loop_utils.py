import torch
import numpy as np
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.data import BatchSampler, SubsetRandomSampler 
import wandb

#-------------------------------------------------------------

#-------------- OUTER LOOP BUFFER ----------------------------

class OL_buffer:
    def __init__(self,device):
        self.num_lifetimes = 0

        self.observations = []
        self.actions = []
        self.logprob_actions = []
        self.advantages = []
        self.logprob_advantages = []
        self.dones = []
        self.extrinsic_rewards= []
        self.success_signals= []

        self.meta_values= []
        self.meta_returns_to_go = []
        self.meta_advantages = []

        self.device=device


    def collect_lifetime_data(self, lifetime_buffer):
        # Append data from a given lifetime_buffer to the combined buffer
        self.num_lifetimes += 1

        self.observations.append(lifetime_buffer.observations)
        self.actions.append(lifetime_buffer.actions)
        self.logprob_actions.append(lifetime_buffer.logprob_actions)
        self.advantages.append(lifetime_buffer.advantages)
        self.logprob_advantages.append(lifetime_buffer.logprob_advantages)
        self.dones.append(lifetime_buffer.dones)
        self.extrinsic_rewards.append(lifetime_buffer.extrinsic_rewards)
        self.success_signals.append(lifetime_buffer.success_signal)
   
        self.meta_values.append(lifetime_buffer.meta_values)
        self.meta_returns_to_go.append(lifetime_buffer.meta_returns_to_go)
        self.meta_advantages.append(lifetime_buffer.meta_advantages) 

    def combine_data(self):
        self.observations=torch.nn.utils.rnn.pad_sequence(self.observations, batch_first=False,padding_value=0.0).to(self.device)
        self.actions=torch.nn.utils.rnn.pad_sequence(self.actions, batch_first=False,padding_value=0.0).to(self.device)
        self.logprob_actions=torch.nn.utils.rnn.pad_sequence(self.logprob_actions, batch_first=False,padding_value=0.0).to(self.device)
        self.advantages=torch.nn.utils.rnn.pad_sequence(self.advantages, batch_first=False,padding_value=0.0).to(self.device)
        self.logprob_advantages=torch.nn.utils.rnn.pad_sequence(self.logprob_advantages, batch_first=False,padding_value=0.0).to(self.device)
        self.dones=torch.nn.utils.rnn.pad_sequence(self.dones, batch_first=False,padding_value=0.0).to(self.device)
        self.extrinsic_rewards=torch.nn.utils.rnn.pad_sequence(self.extrinsic_rewards, batch_first=False,padding_value=0.0).to(self.device)
        self.success_signals=torch.nn.utils.rnn.pad_sequence(self.success_signals, batch_first=False,padding_value=0.0).to(self.device)


        self.meta_values=torch.nn.utils.rnn.pad_sequence(self.meta_values, batch_first=False,padding_value=0.0).to(self.device)
        self.meta_returns_to_go=torch.nn.utils.rnn.pad_sequence(self.meta_returns_to_go, batch_first=False,padding_value=0.0).to(self.device)
        self.meta_advantages=torch.nn.utils.rnn.pad_sequence(self.meta_advantages, batch_first=False,padding_value=0.0).to(self.device)



    def clean_buffer(self):
        self.num_lifetimes = 0
        self.observations = []
        self.actions = []
        self.logprob_actions = []
        self.advantages = []
        self.logprob_advantages = []
        self.dones = []
        self.extrinsic_rewards= []
        self.success_signals= []

        self.meta_values= []
        self.meta_returns_to_go = []
        self.meta_advantages = []



#-------------------------------------------------------------

#----------META AGENT , NETWORK --------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#--
class Adv_agent_actor(nn.Module):
    def __init__(self, hidden_size,initial_std=1):
        super(Adv_agent_actor, self).__init__()
        self.initial_std=initial_std

        self.mean= nn.Sequential(
            layer_init(nn.Linear(hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=0.01)
            )
        self.log_std=nn.Sequential(
            layer_init(nn.Linear(hidden_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=0.01) )


    def forward(self, hidden_state): 
        # If the hidden state is from an LSTM, unpack the tuple
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]  

        mean = torch.arctan(self.mean(hidden_state)) 
        log_std=self.log_std(hidden_state)
        std = torch.exp(log_std ) *self.initial_std 
        return mean ,std 

#--

class Adv_agent_critic(nn.Module):
    def __init__(self, hidden_size):
        super(Adv_agent_critic, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0) )
        

    def forward(self,hidden_state):
        if isinstance(hidden_state, tuple):
            hidden_state = hidden_state[0]  

        value=self.critic(hidden_state)
        return value
    


class Outer_loop_advantage_agent(nn.Module):
    def __init__(self,
                 actions_size,
                 obs_size,               
                 rnn_input_size,        #dimensionality of the input that goes into the rnn at each step
                 rnn_type,              #the type of recurrent network to be used to encode the history of an inner loop life. 'lstm' , 'gru' or 'rnn'
                 rnn_hidden_state_size,  #the dimensionality of the hidden state of the recurrent network used
                 initial_std,            #the initial standard deviation of the advantages given
                 input_sparse_or_shaped='sparse' ,#controls wether the network takes as input the shaped extrinsic rewarrd signal or the shaped signal.
                 ):
        super(Outer_loop_advantage_agent, self).__init__()
        self.input_sparse_or_shaped=input_sparse_or_shaped
        #--------- LIFETIME ENCODER - RNN-------------------
        #Layers for preprocessing input that goes into the rnn at each step
        rnn_input_size1 = obs_size + actions_size+ 1+1+1+1 #(obs_encoding,action_encoding,action_prob,success_signal,prev_done,prev_advantage) 
        rnn_input_size2 = 128
        self.rnn_input_size = rnn_input_size
        self.rnn_input_linear_encoder1= nn.Sequential(layer_init(nn.Linear(rnn_input_size1 , rnn_input_size2-4)), nn.ReLU()) 
        self.rnn_input_linear_encoder2= nn.Sequential(layer_init(nn.Linear(rnn_input_size2 , self.rnn_input_size-4)), nn.ReLU()) 

        #set some factors that scale different inputs
        self.succes_signal_strength= 3
        self.episode_termination_signal_strength= 2
        self.past_advantage_signal_strength= 1
        self.past_action_prob_signal_strength =1


        self.rnn_hidden_state_size=rnn_hidden_state_size
        self.rnn_type=rnn_type

        #instantiating the recurent network
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(self.rnn_input_size, self.rnn_hidden_state_size, batch_first=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(self.rnn_input_size, self.rnn_hidden_state_size, batch_first=False)
            
        #initializing parameters of rnn
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        #----------ACTOR and CRITIC--- 
        self.actor= Adv_agent_actor(self.rnn_hidden_state_size , initial_std=initial_std)

        self.critic =Adv_agent_critic(self.rnn_hidden_state_size)
        #-------------------------------------------------------------

    #define function that initializes the hidden state of the rnn 
    def initialize_state(self, batch_size):
        if self.rnn_type == "lstm":
            rnn__initial_state = (
                torch.zeros(1, batch_size, self.rnn_hidden_state_size),
                torch.zeros(1, batch_size, self.rnn_hidden_state_size),
            )
        elif self.rnn_type == "gru":
            rnn__initial_state = torch.zeros(1, batch_size, self.rnn_hidden_state_size)

        return rnn__initial_state


    def get_rnn_timestep_t_input(self, lifetime_buffer, lifetime_timestep):
        t=lifetime_timestep
        obs_encoding = lifetime_buffer.observations[t]   #shape(obs_size)
        action_encoding = lifetime_buffer.actions[t]  #shape(actions_size)
        action_prob= torch.exp(lifetime_buffer.logprob_actions[t]).unsqueeze(0) * self.past_action_prob_signal_strength  #shape (1)
        prev_done= lifetime_buffer.dones[t].unsqueeze(0) * self.episode_termination_signal_strength#shape(1)

        if t==0:
            prev_advantage=torch.zeros(1).to(prev_done.device)
        else:
            prev_advantage= lifetime_buffer.advantages[t-1].unsqueeze(0) * self.past_advantage_signal_strength  #shape (1)

        if self.input_sparse_or_shaped=='shaped':
            extrinsic_reward= lifetime_buffer.extrinsic_rewards[t].unsqueeze(0) #shape(1)
            rnn_input= torch.cat( (obs_encoding,action_encoding,action_prob,extrinsic_reward,prev_done,prev_advantage) ) #( rnn_input_size1)
            rnn_input=torch.cat( (self.rnn_input_linear_encoder1(rnn_input) ,action_prob,extrinsic_reward,prev_done,prev_advantage)  ) #( rnn_input_size+4)
            rnn_input=torch.cat( (self.rnn_input_linear_encoder2(rnn_input) ,action_prob,extrinsic_reward,prev_done,prev_advantage)  )
        elif self.input_sparse_or_shaped=='sparse':
            success_signal=lifetime_buffer.success_signal[t].unsqueeze(0) * self.succes_signal_strength #shape(1)
            rnn_input= torch.cat( (obs_encoding,action_encoding,action_prob,success_signal,prev_done,prev_advantage) ) #( rnn_input_size1)
            rnn_input=torch.cat( (self.rnn_input_linear_encoder1(rnn_input) ,action_prob,success_signal,prev_done,prev_advantage)  ) #( rnn_input_size+4)
            rnn_input=torch.cat( (self.rnn_input_linear_encoder2(rnn_input) ,action_prob,success_signal,prev_done,prev_advantage)  )

        return rnn_input


    
    def get_subset_of_input_sequence(self, indices, ol_buffer):

        obs_encoding = ol_buffer.observations[indices]    #(len(indices), batch_size, obs_size)
        action_encoding = ol_buffer.actions[indices]   #(len(indices), batch_size, action_size) 
        action_prob= torch.exp(ol_buffer.logprob_actions[indices]).unsqueeze(2)  * self.past_action_prob_signal_strength      #(len(indices), batch_size, 1)
        prev_done=ol_buffer.dones[indices].unsqueeze(2) * self.episode_termination_signal_strength          #(len(indices), batch_size, 1)

        #at time t the intrinsic reward of time t-1 goes in as input.
        advantages=ol_buffer.advantages[indices].unsqueeze(2) 
        num_tasks=advantages.shape[1] #batch_size
        zeros= torch.zeros(1,num_tasks,1).to(prev_done.device)
        prev_advantage=torch.cat( (zeros,advantages[:-1,:,:]),dim=0) * self.past_advantage_signal_strength #(len(indices), batch_size, 1)

        if self.input_sparse_or_shaped=='shaped':
            extrinsic_reward= ol_buffer.extrinsic_rewards[indices].unsqueeze(2)               #(len(indices), batch_size, 1)
            rnn_input= torch.cat((obs_encoding,action_encoding,action_prob,extrinsic_reward,prev_done,prev_advantage) ,dim=2) #(len(indices), batch_size, rnn_input_size1) 
            rnn_input=torch.cat( (self.rnn_input_linear_encoder1(rnn_input) ,action_prob,extrinsic_reward,prev_done,prev_advantage) ,dim=2  )
            rnn_input=torch.cat( (self.rnn_input_linear_encoder2(rnn_input) ,action_prob,extrinsic_reward,prev_done,prev_advantage) ,dim=2 ) #(len(indices), batch_size,rnn_input_size+4)

        elif self.input_sparse_or_shaped=='sparse':
            success_signal=ol_buffer.success_signals[indices].unsqueeze(2) * self.succes_signal_strength       #(len(indices), batch_size, 1)
            rnn_input= torch.cat((obs_encoding,action_encoding,action_prob,success_signal,prev_done,prev_advantage) ,dim=2) #(len(indices), batch_size, rnn_input_size1) 
            rnn_input=torch.cat( (self.rnn_input_linear_encoder1(rnn_input) ,action_prob,success_signal,prev_done,prev_advantage) ,dim=2  )
            rnn_input=torch.cat( (self.rnn_input_linear_encoder2(rnn_input) ,action_prob,success_signal,prev_done,prev_advantage) ,dim=2 ) #(len(indices), batch_size,rnn_input_size+4)

        return rnn_input

        

    def rnn_next_state(self,lifetime_buffer ,lifetime_timestep ,rnn_current_state):
        rnn_step_input=self.get_rnn_timestep_t_input(lifetime_buffer,lifetime_timestep ) 
        rnn_step_input=rnn_step_input.unsqueeze(0).unsqueeze(0) 
        _ , rnn_new_state = self.rnn(rnn_step_input, rnn_current_state)

        return rnn_new_state 

    def get_value(self, hidden_states):
        values= self.critic(hidden_states)
        return values

    def get_action(self, hidden_states, action=None):
        mean ,std  = self.actor(hidden_states)  
        prob_distribution = Normal(mean, std) 
        if action is None:
            action = prob_distribution.sample()
        log_prob=prob_distribution.log_prob(action)
        entropy=prob_distribution.entropy()

        return action, log_prob , entropy
    
    def get_deterministic_action(self, hidden_states):
        mean ,std  = self.actor(hidden_states)
        return mean



#--------------- TBPTT PPO -----------------------------------


class Outer_loop_TBPTT_PPO():
    def __init__(self, optimizer,logging=False,k=100,
                 update_epochs=2,num_minibatches=10,normalize_advantage=True,
                 entropy_coef=0,valuef_coef=0.5,
                 clip_grad_norm=False,max_grad_norm=0.5,target_KL=None,clip_coef=0.2): 


        self.optimizer = optimizer
        self.logging=logging

        self.k = k 
        self.update_epochs=update_epochs
        self.num_minibatches=num_minibatches
        self.normalize_advantage=normalize_advantage
        self.entropy_coef=entropy_coef
        self.valuef_coef=valuef_coef
        self.clip_grad_norm=clip_grad_norm
        self.max_grad_norm=max_grad_norm
        self.target_KL=target_KL
        self.clip_coef=clip_coef

        assert self.k>=self.update_epochs, 'cant have update_epochs be larger than k in tbptt implementation '

        #instantiate space for metrics logged after each update
        self.update_number=0  
        self.learning_rates=[]
        self.value_losses=[] 
        self.policy_gradient_losses=[]
        self.entropies=[]
        self.aprox_KL=[]  
        self.fractions_of_clippings=[] 



    def update(self,meta_agent,buffer,dont_consider_last=0):
        '''for update_epochs iterations: Divide the sequence of collected data into k length subsequences cutting gradient propagation between them.
        Do num_minibatches gradient steps that update the networks parameters with a PPO loss . Each of this steps considers the gradients from a subset of the k length subsequences .
        If after an epoch the approximate KL divergence between the
        updated policy and the policy before starting updating goes beyond a treshold 'target_KL', the update terminates.'''
  

        #normalize advantages 
        if self.normalize_advantage==True:
            buffer.meta_advantages[:-dont_consider_last] = (buffer.meta_advantages[:-dont_consider_last] - buffer.meta_advantages[:-dont_consider_last].mean()) / (buffer.meta_advantages[:-dont_consider_last].std() + 1e-8)

        #randomly choose different displacements for each training epoch to use
        displacements=np.random.choice(a=self.k ,size=self.update_epochs ,replace=False)+1 

        for epoch_num in range(self.update_epochs):
            displacement=displacements[epoch_num]

            self._one_TBPTT_through_sequence(buffer,meta_agent,displacement=displacement,dont_consider_last=dont_consider_last)


            if (self.target_KL!=None) and (np.array(self.aprox_KL).mean()>self.target_KL): 
                break
            #empty metrics saved for logging except in last epoch.
            if epoch_num != (self.update_epochs-1):
                self.value_losses=[] 
                self.policy_gradient_losses=[]
                self.entropies=[]
                self.aprox_KL=[]  
                self.fractions_of_clippings=[] 

        #Log metrics
        if self.logging==True:
            lr=self.optimizer.param_groups[0]["lr"]
            wandb.log({'learning_rate': lr ,
                       'policy gradient loss':np.array(self.policy_gradient_losses).mean(),
                        'value loss':np.array(self.value_losses).mean(),
                        'entropy ': np.array(self.entropies).mean(),
                        'aprox KL' :  np.array(self.aprox_KL).mean(),
                        'fraction of clippings':np.array(self.fractions_of_clippings).mean()} , commit=False )

        self.update_number+=1 

    def _one_TBPTT_through_sequence(self, buffer , meta_agent ,displacement , dont_consider_last=0):

        ###-- preparations -- ###
        sequence_length=buffer.dones.shape[0]
        indices=np.arange(sequence_length-dont_consider_last) 
        num_subsequences=int(len(indices)/self.k) 
        current_subsequence_number=0 
        if self.num_minibatches == 0:
            subsequences_at_which_to_update=np.arange(0,num_subsequences+1) #num_minibatches=0 indicated performing a gradient step for each subsequence
        else:
            assert self.num_minibatches+2<=num_subsequences, 'sequence to short for specified number of minibatches'
            subsequences_at_which_to_update=np.linspace(0,num_subsequences,self.num_minibatches+1).astype(np.int64)[1:] 

        self.optimizer.zero_grad()



        ###---perform the first backpropagation which is done over the first 'diplacement' steps----###
        b_indices=indices[:displacement]
        input_subsequence=meta_agent.get_subset_of_input_sequence( b_indices, buffer)
        #obtain intitial state
        batch_size=input_subsequence.shape[1]
        rnn_initial_state= meta_agent.initialize_state(batch_size=batch_size) # (1,batch_size,rnn_hidden_state_size)
        if isinstance(rnn_initial_state, tuple):
            rnn_initial_state = tuple(hs.to(input_subsequence.device) for hs in rnn_initial_state)
        else:
            rnn_initial_state= rnn_initial_state.to(input_subsequence.device) 
        #compute loss for the subsequence
        rnn_hidden_states , last_hidden= meta_agent.rnn(input_subsequence, rnn_initial_state) 
        mean_entropy , policy_gradient_loss = self._entropy_and_policy_grad_loss(indices=b_indices , hidden_states=rnn_hidden_states ,buffer=buffer,meta_agent=meta_agent)
        value_loss = self._value_loss(indices=b_indices , hidden_states=rnn_hidden_states ,buffer=buffer,meta_agent=meta_agent)
        loss = policy_gradient_loss - self.entropy_coef * mean_entropy + value_loss * self.valuef_coef
        loss.backward()

        current_subsequence_number+=1
        if current_subsequence_number in subsequences_at_which_to_update:
            if self.clip_grad_norm: 
                nn.utils.clip_grad_norm_(meta_agent.parameters(), self.max_grad_norm)
            self.optimizer.step() 
            self.optimizer.zero_grad() 

        #Save metrics for logging
        self.value_losses.append(value_loss.item())
        self.policy_gradient_losses.append(policy_gradient_loss.item())
        self.entropies.append(mean_entropy.item())



        ###---perform the backpropagations over the rest of the sequence  , each subsequence considered takes into account k steps---- ###
        indices=indices[displacement:]#start from a different timestep according to the 'displacement' argument
        for b_indices in np.array_split(indices,indices_or_sections=num_subsequences-1) :

 
            input_subsequence=meta_agent.get_subset_of_input_sequence( b_indices, buffer) 
 
            if isinstance(last_hidden, tuple):
                last_hidden = tuple(hs.detach() for hs in last_hidden)
            else:
                last_hidden= last_hidden.detach()
            rnn_hidden_states , last_hidden= meta_agent.rnn(input_subsequence, last_hidden) 

            #compute loss for the subsequence
            mean_entropy , policy_gradient_loss = self._entropy_and_policy_grad_loss(indices=b_indices , hidden_states=rnn_hidden_states ,buffer=buffer,meta_agent=meta_agent)
            value_loss = self._value_loss(indices=b_indices , hidden_states=rnn_hidden_states ,buffer=buffer,meta_agent=meta_agent)
            loss = policy_gradient_loss - self.entropy_coef * mean_entropy + value_loss * self.valuef_coef
            loss.backward()

            current_subsequence_number+=1
            if current_subsequence_number in subsequences_at_which_to_update:
                if self.clip_grad_norm: 
                    nn.utils.clip_grad_norm_(meta_agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            #Save metrics for logging
            self.value_losses.append(value_loss.item())
            self.policy_gradient_losses.append(policy_gradient_loss.item())
            self.entropies.append(mean_entropy.item())

    def _entropy_and_policy_grad_loss(self, indices, hidden_states ,buffer ,meta_agent):

    
        _, new_logp_advantage , entropy = meta_agent.get_action(hidden_states ,action=buffer.advantages[indices].unsqueeze(2)) #(k,batch_size,1) , // 
        new_logp_advantage,entropy =new_logp_advantage.squeeze(2) ,entropy.squeeze(2) #(k,batch_size) , //

        logratio = new_logp_advantage - buffer.logprob_advantages[indices] #shape (k,batch_size)
        ratio = logratio.exp() 

        m_advantage = buffer.meta_advantages[indices] #shape (k,batch_size)

        pg_loss1 = -m_advantage * ratio
        pg_loss2 = -m_advantage * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = (torch.max(pg_loss1, pg_loss2)).mean() 


        mean_entropy = entropy.mean() 

        #calculate metrics for logging before returning losses
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean().item() 
            clipfracs = [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
            self.aprox_KL.append(approx_kl)
            self.fractions_of_clippings.append(clipfracs)


        return mean_entropy , pg_loss

    def _value_loss(self, indices , hidden_states ,buffer,meta_agent):
        new_meta_value= meta_agent.get_value(hidden_states).squeeze(2) #(k,batch_size)
        v_loss = 0.5 * ((new_meta_value - buffer.meta_returns_to_go[indices]) ** 2).mean()

        return v_loss


