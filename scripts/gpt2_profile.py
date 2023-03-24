from datasets import Dataset
import numpy as np
import torch
from torch.profiler import ProfilerActivity
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    logging,
)

logging.set_verbosity_error()


seq_len = 512
dataset_size = 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size, seq_len)),
}
trn_ds = Dataset.from_dict(dummy_data)
trn_ds.set_format("pt")

model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    output_dir="tmp",
    evaluation_strategy="steps",
    num_train_epochs=1,
    log_level="error",
    report_to="none",
)
trainer = Trainer(model=model, args=training_args, train_dataset=trn_ds)

with torch.profiler.profile(
    activities=[ProfilerActivity.CUDA],
    record_shapes=True,
    with_modules=True,
    with_stack=True,
    with_flops=True,
    use_cuda=True,
) as prof:
    result = trainer.train()
print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
