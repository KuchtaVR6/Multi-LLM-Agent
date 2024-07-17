supportedModels = {
    'llama': 'meta-llama/Llama-2-7b-hf',
    'dev': 'EleutherAI/pythia-160m',
}


def get_model_path_on_suffix(model_suffix):
    if not model_suffix:
        model_suffix = 'caller'

    if model_suffix in supportedModels:
        return supportedModels[model_suffix]
    else:
        return f'saved_models/{model_suffix}'
