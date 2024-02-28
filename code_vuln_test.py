import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.2.1-mistral-7b")
model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/dolphin-2.2.1-mistral-7b", load_in_8bit=True)

def llm_bin_classify(query):
    prompt = [
        {"role": "system", "content": "This is a binary classification task. Reply with \"yes\" or \"no\" only, nothing else."},
        {"role": "user", "content": f"{query}"}]
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
    predicted_tokens = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
    print(predicted_tokens) 

    logits = generated_ids.scores[0][0]
    yes_index = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_index = tokenizer.encode("no", add_special_tokens=False)[0]
    relevant_logits = torch.tensor([logits[yes_index], logits[no_index]]) 
    probabilities = torch.softmax(relevant_logits, dim=-1)

    is_vulnerable = probabilities[0] > probabilities[1]
    confidence = probabilities[0] if is_vulnerable else probabilities[1]

    return is_vulnerable.item(), confidence.item()

# for code,refusal in code_demos:
#     print(llm_bin_classify(f"Does the following code contain any vulnerabilities? {code}"))
# print(llm_bin_classify("Is the sky blue?"))
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2.2.1-mistral-7b")
model = AutoModelForCausalLM.from_pretrained("cognitivecomputations/dolphin-2.2.1-mistral-7b").to("cuda")
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])