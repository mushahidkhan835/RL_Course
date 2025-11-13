import os
from datasets import load_dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
import torch
import random


MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
DATA_DIR = "./data_splits"
LOAD_DIR = "./checkpoints/stage1_qformer_align"
OUTPUT_DIR = "./checkpoints/stage2_finetune_lm_qformer"
MAX_LENGTH = 256
LR = 1e-4
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8

# Make sure checkpoint directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# Load model and processor
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(device)

# Load stage 1 checkpoints
qformer_path = os.path.join(LOAD_DIR, "qformer.pt")
proj_path = os.path.join(LOAD_DIR, "proj.pt")
model.qformer.load_state_dict(torch.load(qformer_path, map_location=device))
model.language_projection.load_state_dict(torch.load(proj_path, map_location=device))

# Stage 2: Freeze vision encoder, fine-tune language model and q-former cross-attention + mlp
for name, param in model.named_parameters():
    param.requires_grad = False
    if (
        "language_model" in name or
        "qformer.crossattention" in name or
        "qformer.mlp" in name or
        "language_projection" in name
    ):
        param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters in stage 1: {trainable_params/1e6:.2f} M")

# Loading dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "val": os.path.join(DATA_DIR, "val.jsonl"),
    },
)


### NOT YET FINISHED. TODO: Consider setting up for LoRA fine-tuning instead to save memory ###


PROMPTS = ["Generate the Pennylane code for this image:"] # for now just one prompt, maybe extend later? 

# Defining preprocessing function
def preprocess(batch):

    # choose prompt 
    prompts = [random.choice(PROMPTS) for _ in batch["image"]]

    # encode inputs (images + prompts)
    inputs = processor(
        images=batch["image"],
        text=prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # encode targets (code)
    labels = processor.tokenizer(
        batch["code"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).input_ids

    inputs["labels"] = labels 
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=["image", "code"])

# Configure training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=torch.cuda.is_available(),
    bf16=False, 
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    remove_unused_columns=False,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
)

# Train the model
trainer.train()

# Save the final model
# trainer.save_model(OUTPUT_DIR) # saving full model, takes too much space

torch.save(model.qformer.state_dict(), os.path.join(save_dir, "qformer.pt"))
torch.save(model.language_projection.state_dict(), os.path.join(save_dir, "proj.pt"))

print(f"Stage 1 fine-tuning now completed, checkpoints saved to {OUTPUT_DIR}")
