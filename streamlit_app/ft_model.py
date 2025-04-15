import mlx.core as mx
from mlx_lm import load, convert
import os

def setup_model():
    """
    Load the LED model and tokenizer from Hugging Face using MLX
    """
    print("SETTING UP THE MODEL")
    
    # Define model parameters
    original_model_name = "Prathyusha101/led-large-16384-arxiv"
    # mlx_model_path = "led-large-16384-arxiv-mlx"
    
    # Convert the model to MLX format if it hasn't been converted yet
    if not os.path.exists(mlx_model_path):
        print(f"Converting {original_model_name} to MLX format...")
        convert(
            original_model_name,
            mlx_model_path,
            quantize=True,  # Set to True for 4-bit quantization for even faster inference
        )
    
    # Load the converted model
    model, tokenizer = load(mlx_model_path)
    
    print("model is setup!!", model)
    print("tokenizer setup!", tokenizer)
    
    return model, tokenizer

def generate_answers(text, num_outputs=5):
    """
    Generate multiple answers from the input text using the LED model with MLX
    
    Args:
        text (str): Input text to be processed
        num_outputs (int): Number of different outputs to generate
    
    Returns:
        list: The generated answers
    """
    model, tokenizer = setup_model()
    
    # Prepare for generation
    print("input text is being processed")
    
    # For long context models like LED, tokenize the input
    tokens = tokenizer.encode(text)
    print(f"Input encoded to {len(tokens)} tokens")
    
    # Generate multiple predictions
    outputs = []
    from mlx_lm import generate, stream_generate
    from mlx_lm.sample_utils import sample_top_k
    
    for i in range(num_outputs):
        print(f"Generating output {i+1}/{num_outputs}...")
        
        # Use MLX-LLM's generate function
        generation = generate(
            model, 
            tokenizer, 
            prompt=text,
            max_tokens=512,
            temp=0.7,  # temperature parameter for sampling
            top_k=50,
            verbose=False
        )
        
        print(f"Output {i+1} generated!")
        outputs.append(generation)
    
    print("All outputs generated!", len(outputs))
    return outputs

def main():
    """
    Main function to run the script
    """
    print("Loading model and tokenizer... (this may take a moment)")
    model, tokenizer = setup_model()
    print(f"Model loaded successfully and optimized for Apple Silicon")
    
    # Optional: If you have macOS 15.0+, you can optimize memory usage
    import os
    import platform
    
    macos_version = platform.mac_ver()[0]
    if macos_version >= "15.0":
        print("Detected macOS 15.0 or higher - model wiring is available for better performance")
        print("If generation is slow, consider running: sudo sysctl iogpu.wired_limit_mb=N")
        print("where N is larger than the model size in MB")
    
    while True:
        print("\nEnter your text (or type 'exit' to quit):")
        text = input()
        if text.lower() == 'exit':
            break
        
        print("\nGenerating answers...")
        answers = generate_answers(text)
        
        print("\n5 Generated Answers:")
        for i, answer in enumerate(answers, 1):
            print(f"\nAnswer {i}:")
            print(answer)
            print("-" * 50)

if __name__ == "__main__":
    # Enable MLX's memory optimization
    os.environ["MLX_USE_METALFFT"] = "1"  # Improve performance with Metal
    main()