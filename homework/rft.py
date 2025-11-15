import torch
import random
import math
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from .base_llm import BaseLLM
from .data import Dataset
from .sft import TokenizedDataset, test_model # Re-use components from SFT


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example_rft(prompt: str, correct_answer: float, completion: str):
    completion_text = completion.strip()
    
    if "<answer>" not in completion_text:
        completion_text = f"{completion_text} <answer>{correct_answer:g}</answer>"
        
    return {"question": prompt, "answer": completion_text}


def train_model(
    output_dir: str,
    **kwargs,
):
    base_model = BaseLLM()

    lora_rank = kwargs.pop("lora_rank", 8) # Default to 8
    lora_alpha = kwargs.pop("lora_alpha", lora_rank * 4)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model.model, config)
    # model.enable_input_require_grads()
    # model.print_trainable_parameters()
    # model.print_trainable_parameters()

    # if torch.cuda.is_available():
    model.enable_input_require_grads()

    print("Trainable parameters:")
    model.print_trainable_parameters()

    tokenizer = base_model.tokenizer

    all_data = Dataset("rft") #the dataset just generated from datagen.py
    data_list = all_data.data
    random.shuffle(data_list) # Shuffle before splitting

    # Use 10% for evaluation
    split_idx = math.floor(0.9 * len(data_list))
    
    train_list = data_list[:split_idx]
    eval_list = data_list[split_idx:]

    # Wrap the lists back into Dataset objects
    train_data = Dataset("rft"); train_data.data = train_list
    eval_data = Dataset("rft"); eval_data.data = eval_list
    
    print(f"RFT data split: {len(train_data)} training, {len(eval_data)} evaluation.")

    tokenized_train_data = TokenizedDataset(tokenizer, train_data, format_example_rft)
    tokenized_eval_data = TokenizedDataset(tokenizer, eval_data, format_example_rft)


    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=kwargs.pop("num_train_epochs", 10),
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=20,
        gradient_checkpointing=True,
        logging_steps=10,
        
        save_strategy="epoch",
        evaluation_strategy="epoch", 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss",
        greater_is_better=False,          
        save_total_limit=2,          
        
        **kwargs, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data, # Pass in the eval dataset
    )

    trainer.train(resume_from_checkpoint=kwargs.get("resume_from_checkpoint", None))
    
    model.save_pretrained(output_dir)
    print(f"Best RFT model saved to {output_dir}")
    
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})