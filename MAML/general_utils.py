      
import numpy as np
import torch
import wandb

# ------------- Logger . For logging metrics ---------------
class Logger:
    def __init__(self , num_epsiodes_of_validation=2):

        self.lifetimes_mean_episode_return= [] #stores the mean episode return of the lifetimes used during the outer loop training
        self.lifetimes_meta_returns = [] #stores the meta return of the lifetimes used during the outer loop training , #it only makes sense if i use the same K and alpha always for computing the metalearning objective.
        self.lifetimes_success_percentage =[] #for all lifetimes used during the outer loop training ,it stores the percentage of episodes where the agent succedded in the lifetime

        self.per_env_total_return={}  #stores the total return of the lifetimes but for each different type of env used (instead of a single list for all envs)
        self.per_env_success_percentage={}

        self.base_maml_agent_success_percentage=[]
        self.base_maml_agent_return=[]
        self.adapted_maml_agent_success_percentage=[]
        self.adapted_maml_agent_return=[]

        self.validation_episodes_return=[]  #stores the return in the last 'num_epsiodes_of_validation' episodes of each lifetime
        self.validation_episodes_success_percentage=[]
        self.num_epsiodes_of_validation=num_epsiodes_of_validation


        self.lifetimes_episodes_returns= [] #stores a list of lists. each inner list contains the episode returns of a lifetime
        self.lifetimes_episodes_successes = [] #stores a list of lists. each inner list contains information on the success of each episode in a given lifetime

        
    def collect_per_lifetime_metrics(self, lifetime_buffer ,episodes_till_first_adaptation_update=0 , episodes_after_adaptation=0):
        self.lifetimes_mean_episode_return.append(np.array(lifetime_buffer.episodes_returns).mean())
        self.lifetimes_meta_returns.append(lifetime_buffer.meta_return)
        self.lifetimes_success_percentage.append(np.sum(lifetime_buffer.episodes_successes) /len(lifetime_buffer.episodes_successes))
        

        self.base_maml_agent_return.append(np.mean(lifetime_buffer.episodes_returns[0:episodes_till_first_adaptation_update]))
        self.base_maml_agent_success_percentage.append(np.mean(lifetime_buffer.episodes_successes[0:episodes_till_first_adaptation_update]))
        self.adapted_maml_agent_return.append(np.mean(lifetime_buffer.episodes_returns[-episodes_after_adaptation:]))
        self.adapted_maml_agent_success_percentage.append(np.mean(lifetime_buffer.episodes_successes[-episodes_after_adaptation:]))
    
        self.validation_episodes_return.append(lifetime_buffer.episodes_returns[-self.num_epsiodes_of_validation:])
        self.validation_episodes_success_percentage.append(np.sum(lifetime_buffer.episodes_successes[-self.num_epsiodes_of_validation:]) / self.num_epsiodes_of_validation  )


        if lifetime_buffer.env_name not in self.per_env_total_return:
            self.per_env_total_return[f'{lifetime_buffer.env_name}']= []
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_returns))
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}']= []
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_successes) /len(lifetime_buffer.episodes_successes))
        else:
            self.per_env_total_return[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_returns))
            self.per_env_success_percentage[f'{lifetime_buffer.env_name}'].append(np.sum(lifetime_buffer.episodes_successes) /len(lifetime_buffer.episodes_successes))


    def log_per_update_metrics(self , num_inner_loops_per_update):

        #log per environment metrics
        for env_name in self.per_env_total_return:
            env_return= np.array(self.per_env_total_return[env_name][-10:]).mean()
            env_success= np.array(self.per_env_success_percentage[env_name][-10:]).mean()
            wandb.log({env_name+' returns': env_return ,env_name+' success':env_success}, commit=False)
       

        #log metrics taking a mean over all the lifetimes considered for the update

        wandb.log({'base maml agent success percentage': np.array(self.base_maml_agent_success_percentage[-num_inner_loops_per_update:]).mean() ,'base maml agent return':np.array(self.base_maml_agent_return[-num_inner_loops_per_update:]).mean() ,
                   'adapted maml agent success percentage': np.array(self.adapted_maml_agent_success_percentage[-num_inner_loops_per_update:]).mean()
                   ,'adapted maml agent return':np.array(self.adapted_maml_agent_return[-num_inner_loops_per_update:]).mean() }, commit=False)

        validation_episodes_return=np.array(self.validation_episodes_return[-num_inner_loops_per_update:]).mean()
        validation_episodes_success_percentage=np.array(self.validation_episodes_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'validation episodes return': validation_episodes_return ,'validation episodes success percentage':validation_episodes_success_percentage}, commit=False)


        mean_episode_return=np.array(self.lifetimes_mean_episode_return[-num_inner_loops_per_update:]).mean()
        lifetime_meta_return= np.array(self.lifetimes_meta_returns[-num_inner_loops_per_update:]).mean()
        lifetime_success_percentage=np.array(self.lifetimes_success_percentage[-num_inner_loops_per_update:]).mean()
        wandb.log({'mean episode return': mean_episode_return ,'total lifetime meta return':lifetime_meta_return,
                    'lifetime success percentage':lifetime_success_percentage})



#---------- Statistics tracker  ----------
#keeps track of reward statistics for normalization puproses and for estimating control variates

class Statistics_tracker:
    def __init__(self ):
        self.e_rewards_means={}  #keeps track of the mean extrinsic reward given by each environment type
        self.e_rewards_vars={}

        self.intrinsic_rewards_mean=0  #keeps track of the mean intrinsic reward given 
        self.intrinsic_rewards_means=[] #keeps track of the mean intrinsic reward given in the last n lifetimes
        self.e_rewards_mean=0  #keeps track of the total mean extrinsic reward
        self.list_e_rewards_means=[] #keeps track of the mean extrinsic reward given in the last n lifetimes
        self.sparse_rewards_mean=0 #keeps track of the total mean sparse rewards
        self.sparse_rewards_means=[]

        #for calculating a running mean of extrinsic rewards
        self.num_lifetimes_processed={}
        self.means_sums={}


    def update_statistics(self,lifetime_buffer):
        #update extinsic rewards statistics
        sample_mean= torch.mean(lifetime_buffer.extrinsic_rewards)
        #first time that environment type is encountered
        if lifetime_buffer.env_name not in self.e_rewards_means:
            self.e_rewards_means[f'{lifetime_buffer.env_name}']= sample_mean 
            self.num_lifetimes_processed[f'{lifetime_buffer.env_name}']=1
            self.means_sums[f'{lifetime_buffer.env_name}'] =sample_mean
        
        else:
            self.num_lifetimes_processed[f'{lifetime_buffer.env_name}']+=1
            self.means_sums[f'{lifetime_buffer.env_name}'] += sample_mean
            self.e_rewards_means[f'{lifetime_buffer.env_name}']= self.means_sums[f'{lifetime_buffer.env_name}'] / self.num_lifetimes_processed[f'{lifetime_buffer.env_name}']


        self.list_e_rewards_means.append(sample_mean)
        if len(self.list_e_rewards_means)>60:
            self.list_e_rewards_means=self.list_e_rewards_means[-60:]
        self.e_rewards_mean=np.array(self.list_e_rewards_means).mean()


        #update intrinsic rewards statistics
        sample_mean=torch.mean(lifetime_buffer.intrinsic_rewards)
        self.intrinsic_rewards_means.append(sample_mean)
        if len(self.intrinsic_rewards_means)>60:
            self.intrinsic_rewards_means=self.intrinsic_rewards_means[-60:]
        self.intrinsic_rewards_mean=np.array(self.intrinsic_rewards_means).mean()


        #update mean sparse reward statistic
        sample_mean=torch.mean(lifetime_buffer.success_signal)
        self.sparse_rewards_means.append(sample_mean)
        if len(self.sparse_rewards_means)>60:
            self.sparse_rewards_means=self.sparse_rewards_means[-60:]
        self.sparse_rewards_mean=np.array(self.sparse_rewards_means).mean()