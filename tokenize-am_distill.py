import os
import datasets
from datasets import load_dataset, Dataset, Features, Value, Sequence
from transformers import AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import numpy as np

def apply_chat_template(example, tokenizer):
    """Apply chat template to convert messages to text format"""
    messages = example["conversations"]
    converted_messages = []
    for msg in messages:
        if msg["from"] == "human":
            role = "user"
        elif msg["from"] == "assistant":
            role = "assistant"
        else:
            raise ValueError(f"Unknown role: {msg['from']}")
        converted_messages.append({"role": role, "content": msg["value"]})
    
    # Keep the converted messages in the dataset
    example["messages"] = converted_messages
    example["text"] = tokenizer.apply_chat_template(
        converted_messages, tokenize=False, add_generation_prompt=False)
    return example

def tokenize_function(examples, tokenizer, max_seq_length):
    """Tokenize the text data"""
    tokenized = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
        return_overflowing_tokens=False,
    )
    tokenized["num_tokens"] = [len(input_ids) for input_ids in tokenized["input_ids"]]
    
    return tokenized

def filter_by_length(example, max_seq_length):
    """Filter out examples that exceed maximum sequence length"""
    return example["num_tokens"] <= max_seq_length

def plot_token_distribution(all_token_counts, max_tokens, max_seq_length, output_dir):
    """Create and save token distribution plot"""
    print("Creating token distribution plot...")
    
    # Define bins with 256 intervals
    max_length = max(max_tokens, max_seq_length)
    bins = list(range(0, max_length + 256, 256))
    
    # Calculate interval percentages
    interval_percentages = []
    interval_labels = []
    
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        
        # Count tokens in this interval
        count = sum(1 for token_count in all_token_counts if bin_start < token_count <= bin_end)
        percentage = (count / len(all_token_counts)) * 100
        interval_percentages.append(percentage)
        interval_labels.append(f"{bin_start}-{bin_end}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    plt.bar(bin_centers, interval_percentages, width=200, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Token Count', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Token Length Distribution (Interval)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xlim(0, max_length)
    
    # Set x-axis ticks to show every 256
    plt.xticks(bins[::2])  # Show every other bin to avoid crowding
    
    # Save the plot
    plot_path = os.path.join(output_dir, "token_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Token distribution plot saved to: {plot_path}")
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Tokenize and save dataset")
    parser.add_argument("--model_path", type=str, default="/share/edc/home/xuan_luo/MTP/Ablation", 
                       help="Path to the model/tokenizer")
    parser.add_argument("--dataset_name", type=str, default="a-m-team/AM-Thinking-v1-Distilled",
                       help="Name of the dataset to load")
    parser.add_argument("--config_name", type=str, default="default",
                       help="Config name for the dataset")
    parser.add_argument("--output_dir", type=str, default="./am-distilled-8192",
                       help="Directory to save the processed dataset")
    parser.add_argument("--test_size", type=float, default=0.01,
                       help="Fraction of data to use for evaluation")
    parser.add_argument("--num_proc", type=int, default=32,
                       help="Number of processes for dataset processing")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                       help="Maximum sequence length for tokenization")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"Loading dataset: {args.dataset_name} with config: {args.config_name}...")
    raw_dataset = load_dataset(args.dataset_name, args.config_name, streaming=True)
    train_dataset = raw_dataset["train"]
    
    print("Applying chat template...")
    processed_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["conversations", "system"],
    )

    processed_dataset = Dataset.from_generator(
        lambda: processed_dataset,
        features=processed_dataset.features,
        num_proc=args.num_proc,
    )

    print("Tokenizing dataset...")
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["text"],  # Remove text column after tokenization, keep messages
        desc="Tokenizing dataset",
    )
    
    print(f"Filtering out sequences longer than {args.max_seq_length} tokens...")
    original_size = len(tokenized_dataset)
    filtered_dataset = tokenized_dataset.filter(
        filter_by_length,
        fn_kwargs={"max_seq_length": args.max_seq_length},
        num_proc=args.num_proc,
        desc="Filtering by sequence length",
    )
    filtered_size = len(filtered_dataset)
    discarded_count = original_size - filtered_size
    print(f"Original dataset size: {original_size}")
    print(f"Filtered dataset size: {filtered_size}")
    print(f"Discarded {discarded_count} samples ({discarded_count/original_size*100:.2f}%)")
    
    print(f"Splitting dataset (test_size={args.test_size})...")
    train_test_split = filtered_dataset.train_test_split(test_size=args.test_size, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    # Calculate token statistics
    train_token_counts = list(train_dataset["num_tokens"])
    eval_token_counts = list(eval_dataset["num_tokens"])
    all_token_counts = train_token_counts + eval_token_counts
    
    avg_tokens = sum(all_token_counts) / len(all_token_counts)
    min_tokens = min(all_token_counts)
    max_tokens = max(all_token_counts)
    
    print(f"Token statistics:")
    print(f"  Total tokens: {sum(all_token_counts)}")
    print(f"  Average tokens: {avg_tokens:.2f}")
    print(f"  Minimum tokens: {min_tokens}")
    print(f"  Maximum tokens: {max_tokens}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create token distribution plot
    plot_path = plot_token_distribution(all_token_counts, max_tokens, args.max_seq_length, args.output_dir)
    
    print(f"Saving processed dataset to {args.output_dir}...")
    # Save as a DatasetDict for easy loading
    dataset_dict = datasets.DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset
    })
    dataset_dict.save_to_disk(args.output_dir)
    
    print("Dataset processing and saving completed!")
    print(f"You can now load the dataset using: datasets.load_from_disk('{args.output_dir}')")
    print("The dataset is now fully tokenized and ready for training!")

if __name__ == "__main__":
    main() 
