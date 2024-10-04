import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Function to generate output with specified input length and max output tokens
def generate_text_with_length(model_name, input_text, max_input_length, max_output_length):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Check if input length exceeds the specified limit
    if input_ids.size(-1) > max_input_length:
        input_ids = input_ids[:, :max_input_length]

    # Generate text with the specified maximum output length
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=input_ids.size(-1) + max_output_length,
            do_sample=True,
            top_p=0.95,  # You can tweak sampling parameters like top_p or temperature
            top_k=50
        )

    # Decode the output tokens to text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate text using GPT-2 model")
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='Name of the GPT-2 model')
    parser.add_argument('--input_text', type=str, required=True, help='Input text to seed the generation')
    parser.add_argument('--input_length', type=int, required=True, help='Maximum length of the input tokens')
    parser.add_argument('--output_length', type=int, required=True, help='Maximum number of tokens to generate')

    args = parser.parse_args()

    # Generate text based on the provided arguments
    generated_text = generate_text_with_length(
        args.model_name, 
        args.input_text, 
        args.input_length, 
        args.output_length
    )

    print("Generated Text: ", generated_text)

if __name__ == "__main__":
    main()
