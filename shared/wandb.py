import os
import wandb


class W:  # W for wandb
    @staticmethod
    def get_wandb_project_name(h: dict):
        assert h['method'] in ['uq_through_redundancy', 'ensemble']

        project_name = f"{h['method']}_{h['redundancy_method']}_{h['dataset']}_{h['wandb_project_name']}"
        run_name = f"m={h['method']} alpha={h['alpha']} nb_views={h['num_views']}"  # f"{family}_kld={options.encoder_config.kld_weight}_lr={options.encoder_config.learning_rate}_{int(time.time())}"
        entity = h['wandb_entity']
        # replace empty string with None
        entity = entity if entity != "" else None
        return entity, project_name, run_name

    @staticmethod
    def initialize_wandb(h: dict, entity, project_name, run_name):
        # conf = vars(h)
        conf = h.copy()
        # store conf._classifier_configs them as string instead of type Dict[ClassifierKey, ...]
        # Fixes ClassifierKey not serializable error for wandb
        # conf["_classifier_configs"] = str(conf["_classifier_configs"])

        wandb.init(entity=entity, project=project_name, name=run_name, config=conf)
        # After initializing the wandb run, get the run id
        run_id = wandb.run.id
        # Save the run id to a file in the logs directory
        with open(os.path.join(h['log_path'], 'wandb_run_id.txt'), 'w') as f:
            f.write(run_id)
            # write project name to file
            f.write(f"\n{project_name}")

    @staticmethod
    def retrieve_existing_wandb_run_id(h: dict):
        # Save the run id to a file in the logs directory
        if os.path.exists(os.path.join(h['log_path'], 'wandb_run_id.txt')):
            with open(os.path.join(h['log_path'], 'wandb_run_id.txt'), 'r') as f:
                text = f.read()
                # first line is the run id, second line is the project name (second line is optional)
                run_id = text.split('\n')[0]
                project_name = text.split('\n')[1] if len(text.split('\n')) > 1 else None

        # if file doesn't exist, return None
        else:
            run_id = None
            project_name = None

        assert run_id is not None, "Run id not found, set use_wandb to False in the config file to disable wandb logging"
        assert project_name is not None, "Project name not found, set use_wandb to False in the config file to disable wandb logging"

        return run_id, project_name

    @staticmethod
    def log_im(h: dict, im, name: str):
        if h['use_wandb']:
            wandb.log({name: [wandb.Image(im)]})

        # if options.use_wandb:
        #     wandb_section = get_wandb_audio_classific_key(opt, classifier_config)
        #     wandb.log({f"{wandb_section}_softmax/variance_distribution_combined": wandb.Image(fig)})
