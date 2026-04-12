import isla
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on: {device}")

model, tokenizer = isla.load_model("outputs/model_trained", device=device)
model.eval()

prompt = "How much is 1+1=?"
print(f"\nPrompt: {prompt}\n---")

with torch.no_grad():
    for word in isla.generate_stream(
        model, 
        tokenizer, 
        prompt,
        max_new_tokens=150, 
        temperature=0.8,
        top_p=0.9, # top_p not natively in the signature? ah generate_stream handles it in kwargs! wait, does it?
        device=device
    ):
        print(word, end="", flush=True)
print()
