from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the lightweight open-source model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_answer(question, context_chunks):
    # Join top 3 chunks for relevance
    context = "\n\n".join(context_chunks[:3])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
