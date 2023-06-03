from common_imports import *

def create_subset(dataset, num_examples):
    indices = random.sample(range(len(dataset)), num_examples)
    return dataset.select(indices)

def filter_datasets_for_use_case(datasets, use_case):
    filtered_datasets = {}
    for key, value in datasets.items():
        if value[use_case]:
            filtered_datasets[key] = value[use_case]
    return filtered_datasets

def split_datasets(data_dict, ratio=0.7, random_state=None):
    train_data = {}
    valid_data = {}
    validation_indices = {}

    for key, value in data_dict.items():
        train, valid, train_indices, valid_indices = train_test_split(value, range(len(value)), train_size=ratio, random_state=random_state)
        train_data[key] = train
        valid_data[key] = valid
        validation_indices[key] = valid_indices

    return train_data, valid_data, validation_indices

def unique_elements(lst):
    result = []
    seen = set()
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

class PerplexityLoggingCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                    metrics: Dict[str, float], prefix=None, **kwargs):
        if prefix is None:
            prefix = "eval"
        eval_loss_key = f"{prefix}_loss"
        if eval_loss_key in metrics:
            loss = metrics[eval_loss_key]
            metrics[f"{prefix}_perplexity"] = math.exp(loss)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class CustomDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __getitem__(self, idx):
        return self.tensor_list[idx]

    def __len__(self):
        return len(self.tensor_list)
        
def get_sequences(text, tokenizer, seq_length=768, stride_ratio=0.5):
    all_token_ids = tokenizer.encode(text)

    #Generate sequences using sliding window approach
    stride_length = int(seq_length * stride_ratio)
    sequences = []
    for i in range(0, len(all_token_ids) - seq_length +1, stride_length):
        input_ids = all_token_ids[i:i+seq_length]
        sequences.append(input_ids)
    
    #Truncate the last sequence if it less than seq_length
    last_sequence = sequences[-1]
    if len(last_sequence) < seq_length:
        last_sequence = last_sequence + [tokenizer.pad_token_id] * (seq_length - len(last_sequence))
        sequences[-1] = last_sequence

    #Drop any remaining sequences that are less than seq_length
    sequences = [sequence for sequence in sequences if len(sequence) == seq_length]

    return sequences

def evaluate(model, dataloader, device, max_eval_steps):
    model.eval()
    losses = []
    for step, batch in enumerate(dataloader):
        # Extract input_ids and convert them to tensors
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else None

        with torch.no_grad():
            input_dict = {'input_ids': input_ids, 'labels': labels}
            outputs = model(**input_dict)
         
        loss = outputs.loss.repeat(input_ids.shape[0])
        losses.append(loss.detach())
        if max_eval_steps > 0 and step >= max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()

class CustomTrainer(Trainer):
    def __init__(self, *args, max_eval_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_perplexity = float("inf")
        self.best_model_state_dict = None
        self.no_improvement_counter = 0
        self.passed_epoch_steps = False
        self.max_eval_steps = max_eval_steps  # Add max_eval_steps as an attribute

    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, ignore_keys=None, metric_key_prefix='eval'):
        eval_loss, perplexity = evaluate(self.model, dataloader, self.args.device, self.max_eval_steps)
    
        # Check if epoch_steps are surpassed
        if self.state.epoch >= 1:
            self.passed_epoch_steps = True
    
        # Check for improvements if the epoch_steps are surpassed
        if self.passed_epoch_steps:
            if perplexity < self.best_perplexity:
                self.best_perplexity = perplexity
                self.best_model_state_dict = {k: v.clone().to('cpu') for k, v in self.model.state_dict().items()}
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1
    
        # Stop training, load the best state_dict in the model, and return the best_model if the perplexity did not improve 3 times consecutively
        if self.no_improvement_counter == 3:
            if self.best_model_state_dict:
                self.model.load_state_dict(self.best_model_state_dict)
            self.model.to(self.args.device)
            self.control.should_training_stop = True
            print("Training stopped, best model loaded with Perplexity:", self.best_perplexity)
    
        self.log({
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "epoch": self.state.epoch,
        })
    
        # Define num_samples as the total number of samples in the dataloader
        #num_samples = len(dataloader.dataset)
    
        # Initialize an instance of EvalPrediction without the 'metrics' keyword argument 
        #eval_prediction = EvalPrediction(predictions=None, label_ids=None, num_samples=num_samples)
        eval_prediction = EvalPrediction(predictions=None, label_ids=None)
        
        # Define num_samples as the total number of samples in the dataloader
        num_samples = len(dataloader.dataset)
    
        # Add the num_samples attribute to the eval_prediction instance
        eval_prediction.num_samples = num_samples
    
        # Set the metrics dictionary
        eval_prediction.metrics = {"eval_loss": eval_loss}
    
        return eval_prediction
    
    def get_completed_steps(self):
        return self.state.global_step



