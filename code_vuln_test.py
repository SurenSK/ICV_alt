import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.2.1-mistral-7b")
model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/dolphin-2.2.1-mistral-7b", load_in_8bit=True)

with open('code_examples.json', 'r') as f:
    code_demos = json.load(f)

def llm_bin_classify(query):
    prompt = [
        {"role": "system", "content": "This is a binary classification task. Reply with \"yes\" or \"no\" only, nothing else."},
        {"role": "user", "content": f"{query}\n<|im_start|>assistant"}]
    model_inputs  = tokenizer.apply_chat_template(prompt, return_tensors="pt", padding=True).to('cuda')
    generated_ids = model.generate(
                model_inputs,
                max_new_tokens=1,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )

    logits = generated_ids.scores[0]
    print(logits.shape)

    predicted_token_ids = logits.argmax(dim=-1).tolist()
    predicted_tokens = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    print(predicted_tokens) 

    logits = generated_ids.scores[0][0]
    yes_index = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_index = tokenizer.encode("no", add_special_tokens=False)[0]
    relevant_logits = torch.tensor([logits[yes_index], logits[no_index]]) 
    probabilities = torch.softmax(relevant_logits, dim=-1)

    is_vulnerable = probabilities[0] > probabilities[1]
    confidence = probabilities[0] if is_vulnerable else probabilities[1]

    return is_vulnerable.item(), confidence.item()

for code,refusal in code_demos:
    print(llm_bin_classify(f"Does the following code contain any vulnerabilities? {code}"))
print(llm_bin_classify("Is the sky blue?"))
