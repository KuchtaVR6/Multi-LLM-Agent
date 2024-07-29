supportedModels = {
    'llama': 'meta-llama/Llama-2-7b-hf',
    'dev': 'EleutherAI/pythia-160m',
}


def get_model_path_on_suffix(model_suffix, trained_on_all=False):
    if not model_suffix:
        model_suffix = 'caller'

    if model_suffix in supportedModels:
        return supportedModels[model_suffix]
    else:
        base_folder = 'saved_models/'
        if '[' in model_suffix and trained_on_all:
            base_folder += 'trained_on_all/'

        return base_folder + model_suffix
