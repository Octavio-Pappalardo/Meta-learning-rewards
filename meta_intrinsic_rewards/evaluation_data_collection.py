


if __name__=='__main__':
    from evaluation_utils import collect_evaluation_data
    from outer_loop_config import get_ol_config
    import metaworld

    import re
    import os
    import numpy as np
    import time
    

    # define parameters of the evaluation run
    intrinsic_rewards= True
    evaluate_test_tasks= False  #wether to perform the evaluation over the training or test tasks
    model_path= "../../i_rew_models/REW_ML1_door-close-v2__5896987454__best_model.pth"
    benchmark_name='ML1' 
    ml1_env_name= 'door-close-v2' #only relevant when using ML1 benchmark
    ol_config_setting= 'ml1'
    il_config_setting= 'metaworld' 
    n= 10
    post_training_evaluation= True
    num_post_training_evaluation_episodes= 4

    ol_config=get_ol_config(ol_config_setting)
    # Extract the time from the file name (identifies the model) using regular expression . assumes model_path has a format like "path/example123__1638439876__best_model.pth"
    model_id = re.search(r'(?<=__)\d+', os.path.basename(model_path)).group()
    if intrinsic_rewards==False:
        model_id= f'baseline{ol_config.seed}' #if not actually using the meta agent -> save file with a 'baseline ' identifier.
    

    #get a list of the names of all the environments that will be evaluated (each of them contains 50 tasks with parametric variation)
    if benchmark_name=='ML1':
        envs_to_evaluate=[ml1_env_name]
    else:
        benchmark = metaworld.ML10(seed=ol_config.seed) if benchmark_name=='ML10' else metaworld.ML45(seed=ol_config.seed)
        if evaluate_test_tasks==True:
            envs_to_evaluate= [name for name,env_cls in benchmark.test_classes.items()]
        else:
            envs_to_evaluate= [name for name,env_cls in benchmark.train_classes.items()]

    for env_name in envs_to_evaluate:
        start_time=time.time()

        exp_name= f'{benchmark_name}_{env_name}'
        run_name = f"{exp_name}__{model_id}" 
        returns_data_path = f"../../sparse_baseline_eval_data/{run_name}__train_eval_returns.npy"
        successes_data_path = f"../../sparse_baseline_eval_data/{run_name}__train_eval_successes.npy"

        lifetimes_returns_per_episode_data ,lifetimes_success_per_episode_data= collect_evaluation_data(model_path=model_path , 
                                                            n=n , ol_config=ol_config_setting , il_config=il_config_setting ,
                                                            benchmark_name=benchmark_name ,env_name= env_name,
                                                            intrinsic_rewards=intrinsic_rewards , 
                                                            post_training_evaluation=post_training_evaluation , num_post_training_evaluation_episodes=num_post_training_evaluation_episodes,
                                                            evaluate_test_tasks=evaluate_test_tasks)

        np.save(returns_data_path ,lifetimes_returns_per_episode_data)
        np.save(successes_data_path ,lifetimes_success_per_episode_data )

        print(f'evaluation data of experiment : {run_name} ready')
        print(f'succes percentage: {np.mean(lifetimes_success_per_episode_data[:,-4:]) }')
        print(f'time take in minutes ={(time.time()-start_time)/60}')




    