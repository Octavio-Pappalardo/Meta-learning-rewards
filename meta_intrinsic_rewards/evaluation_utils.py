import numpy as np

import metaworld
from outer_loop_config import get_ol_config
from inner_loop import run_inner_loop
from evaluation_inner_loop import run_eval_inner_loop


class Evaluation_buffer:
    def __init__(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]

    def collect_data(self, lifetime_return_per_episode ,lifetime_success_per_episode):
        self.lifetimes_returns_per_episode.append(lifetime_return_per_episode)
        self.lifetimes_success_per_episode.append(lifetime_success_per_episode)

    def combine_data(self):
        self.lifetimes_returns_per_episode=np.array(self.lifetimes_returns_per_episode)
        self.lifetimes_success_per_episode=np.array(self.lifetimes_success_per_episode)
        # after running each should be a numpy array of size (num_lifetimes , num_episodes_per_lifetime)

    def clean_buffer(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]



def collect_evaluation_data(model_path, n , ol_config , il_config , benchmark_name='ML1' ,env_name= 'door-close-v2',
                            intrinsic_rewards=True, post_training_evaluation=False , num_post_training_evaluation_episodes=10,
                             evaluate_test_tasks=True):
    ''' For a given benchmark and environment name it collects all the test(or train) tasks corresponding to that environment ,
    evaluates the performance of the meta agent when being applied to those tasks and returns data of that evaluation.

    Args:
        model_path : path to the weights of the model to be evaluated
        n : amount of times each evaluation task is runned
        ol_config : configurations for outer loop. should match those used when training the agent to be evaluated so that it can be loaded properly
        il_config : configurations for running the inner loops on which the meta agent will be tested
        benchmark_name : name of the benchmark to evaluate on .
        env_name : name of the environment to evaluate on .
        intrinsic_rewards : If false then the evaluation data is collected for the standard RL algorithm (inner loop trained with extrinsic rewards instead of intrinsic rewards from meta agent)
        post_training_evaluation: If true , after a phase of training the inner loop models with intrinsic rewards ,
                                 the inner loop action models are run for num_post_training_evaluation_episodes more iterations in a deterministic fashion
        num_post_training_evaluation_episodes:
        evaluate_test_tasks : if false the evaluation is done over the tasks with which the meta agent trained instead of with the test tasks

    Returns:
        lifetimes_returns_per_episode : np.array of size (num_lifetimes , num_episodes_per_lifetime) containing the return of each episode
        lifetimes_success_per_episode : np.array of size (num_lifetimes , num_episodes_per_lifetime) containing whether each episode succeded
    '''
    import ray
    if ray.is_initialized:
        ray.shutdown()
    ray.init()
    ol_config_setting=ol_config
    il_config_setting=il_config 
    ol_config=get_ol_config(ol_config_setting)

    # Construct the benchmark and get all the evaluation tasks (50)
    if benchmark_name=='ML1':
        benchmark = metaworld.ML1(f'{env_name}', seed=ol_config.seed)
        if evaluate_test_tasks==True:
            eval_tasks= benchmark.test_tasks
        else:
            eval_tasks= benchmark.train_tasks

    elif benchmark_name=='ML10' or benchmark_name=='ML45':
        benchmark = metaworld.ML10(seed=ol_config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=ol_config.seed)
        if evaluate_test_tasks==True:
            eval_tasks=[task for task in benchmark.test_tasks
                        if task.env_name == env_name]
        else:
            eval_tasks=[task for task in benchmark.train_tasks
                        if task.env_name == env_name]

    # set the number of times each task will be evaluated
    eval_tasks=eval_tasks * n
    eval_buffer=Evaluation_buffer() 


    if post_training_evaluation==False:
        remote_inner_loop=ray.remote(run_inner_loop)
        return_action_agent=False
        ol_training=False
        inputs= [ ((ol_config,il_config_setting,model_path,benchmark_name,task),intrinsic_rewards,ol_training,return_action_agent) for task in eval_tasks]
        results =ray.get([remote_inner_loop.options(num_cpus=1).remote(*args) for args in inputs])

    elif post_training_evaluation==True:
        remote_eval_inner_loop=ray.remote(run_eval_inner_loop)
        inputs= [ ((ol_config,il_config_setting,model_path,benchmark_name,task),intrinsic_rewards,num_post_training_evaluation_episodes) for task in eval_tasks]
        results =ray.get([remote_eval_inner_loop.options(num_cpus=1).remote(*args) for args in inputs])

    for episodes_returns,episodes_successes in results:
        eval_buffer.collect_data(lifetime_return_per_episode=episodes_returns ,lifetime_success_per_episode=episodes_successes )


    eval_buffer.combine_data()

    return eval_buffer.lifetimes_returns_per_episode , eval_buffer.lifetimes_success_per_episode
    



