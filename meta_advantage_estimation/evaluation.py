import numpy as np
import metaworld
from outer_loop_config import get_ol_config
from inner_loop import run_inner_loop


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

    def clean_buffer(self):
        self.lifetimes_returns_per_episode=[]
        self.lifetimes_success_per_episode=[]



def collect_evaluation_data(model_path, n , ol_config , il_config , benchmark_name='ML1' ,env_name= 'door-close-v2',
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

    eval_tasks=eval_tasks * n
    eval_buffer=Evaluation_buffer() 



    remote_inner_loop=ray.remote(run_inner_loop)
    return_action_agent=False
    ol_training=False
    inputs= [ ((ol_config,il_config_setting,model_path,benchmark_name,task),ol_training,return_action_agent) for task in eval_tasks]
    results =ray.get([remote_inner_loop.options(num_cpus=1).remote(*args) for args in inputs])



    for episodes_returns,episodes_successes in results:
        eval_buffer.collect_data(lifetime_return_per_episode=episodes_returns ,lifetime_success_per_episode=episodes_successes )


    eval_buffer.combine_data()

    return eval_buffer.lifetimes_returns_per_episode , eval_buffer.lifetimes_success_per_episode
    



if __name__=='__main__':

    import re
    import os
    import time

    # define parameters of the evaluation run
    evaluate_test_tasks=True   #wether to perform the evaluation over the training or test tasks
    model_path= "../../adv_models/ADV_ML1_button-press-v2__589651235__best_model.pth"
    benchmark_name='ML1' 
    ml1_env_name= 'button-press-v2' #only relevant when using ML1 benchmark
    ol_config_setting= 'ml1'
    il_config_setting= 'metaworld' 
    n= 10

    
    # Extract the time from the file name (identifies the model) using regular expression . assumes model_path has a format like "path/example123__1638439876__best_model.pth"
    model_id = re.search(r'(?<=__)\d+', os.path.basename(model_path)).group()

    
    #get a list of the names of all the environments that will be evaluated (each of them contains 50 tasks with parametric variation)
    if benchmark_name=='ML1':
        envs_to_evaluate=[ml1_env_name]
    else:
        ol_config=get_ol_config(ol_config_setting)
        benchmark = metaworld.ML10(seed=ol_config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=ol_config.seed)
        if evaluate_test_tasks==True:
            envs_to_evaluate= [name for name,env_cls in benchmark.test_classes.items()]
        else:
            envs_to_evaluate= [name for name,env_cls in benchmark.train_classes.items()]

    for env_name in envs_to_evaluate:
        start_time=time.time()

        exp_name= f'{benchmark_name}_{env_name}'
        run_name = f"{exp_name}__{model_id}" #__{config.seed}
        returns_data_path = f"../../adv_eval_data/{run_name}__test_eval_returns.npy"
        successes_data_path = f"../../adv_eval_data/{run_name}__test_eval_successes.npy"

        lifetimes_returns_per_episode_data ,lifetimes_success_per_episode_data= collect_evaluation_data(model_path=model_path , 
                                                            n=n , ol_config=ol_config_setting , il_config=il_config_setting ,
                                                            benchmark_name=benchmark_name ,env_name= env_name,
                                                            evaluate_test_tasks=evaluate_test_tasks)

        np.save(returns_data_path ,lifetimes_returns_per_episode_data)
        np.save(successes_data_path ,lifetimes_success_per_episode_data )

        print(f'evaluation data of experiment : {run_name} ready')
        print(f'succes percentage: {np.mean(lifetimes_success_per_episode_data) }')
        print(f'succes percentage: {np.mean(lifetimes_success_per_episode_data[:,-8:-4]) }')
        print(f'time take in minutes ={(time.time()-start_time)/60}')



