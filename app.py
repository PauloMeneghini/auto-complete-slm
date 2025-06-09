# app.py

# Importações necessárias
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# --- Configuração Inicial ---
load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")
TRANSFORMERS_VERBOSITY = os.getenv("TRANSFORMERS_VERBOSITY")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("ERRO: HF_TOKEN não encontrado. Por favor, defina a variável de ambiente HF_TOKEN.")
    print("Você pode obter um token em: https://huggingface.co/settings/tokens")
    exit(1)

# model_name = "google/gemma-2b"
model_name = "google/gemma-3-1b-it"

# --- Carregamento do Modelo e Tokenizador ---
try:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("Modelo Gemma carregado em 4-bit (quantizado).")
except Exception as e:
    print(f"Não foi possível carregar em 4-bit: {e}. Tentando carregar normalmente...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    print("Modelo Gemma carregado normalmente (sem quantização).")

model.eval()
print(f"Modelo '{model_name}' e tokenizador carregados com sucesso.")

def generate_autocomplete_suggestion(prompt: str, max_new_tokens: int = 10) -> str:
    """
    Gera uma sugestão de autocomplete determinística usando o modelo Gemma.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Prompt: '{prompt}'")
    print(f"Tokens de entrada: {input_ids['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    if outputs.shape[0] > 0:
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Texto completo: '{full_generated_text}'")
        
        if full_generated_text.startswith(prompt):
            completion = full_generated_text[len(prompt):]
        else:
            completion = full_generated_text.replace(prompt, "", 1)
        
        completion = process_completion_for_autocomplete(completion)
        print(f"Sugestão processada: '{completion}'")
        return completion
    
    return ""

def process_completion_for_autocomplete(completion: str) -> str:
    """
    Processa a continuação para ser adequada para autocomplete.
    """
    completion = completion.strip()
    
    if '\n' in completion:
        completion = completion.split('\n')[0].strip()
    
    if '.' in completion:
        dot_index = completion.find('.')
        if dot_index != -1:
            completion = completion[:dot_index + 1]
    
    if len(completion) > 50:
        words = completion.split()
        truncated = ""
        for word in words:
            if len(truncated + " " + word) <= 50:
                truncated += " " + word if truncated else word
            else:
                break
        completion = truncated
    
    return completion

def generate_word_autocomplete(prompt: str, max_words: int = 5) -> str:
    """
    Gera apenas algumas palavras para autocomplete determinístico.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Prompt para palavras: '{prompt}'")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids['input_ids'],
            attention_mask=input_ids['attention_mask'],
            max_new_tokens=max_words * 2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    if outputs.shape[0] > 0:
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Texto completo: '{full_text}'")
        
        if full_text.startswith(prompt):
            completion = full_text[len(prompt):].strip()
        else:
            completion = full_text.replace(prompt, "", 1).strip()
        
        words = completion.split()
        if len(words) > max_words:
            completion = " ".join(words[:max_words])
        
        completion = completion.replace('\n', ' ').strip()
        print(f"Sugestão de palavras: '{completion}'")
        return completion
    
    return ""

def test_autocomplete_scenarios():
    """
    Testa diferentes cenários de autocomplete.
    """
    print("\n=== TESTE DE CENÁRIOS DE AUTOCOMPLETE ===")
    
    test_cases = [
        ("Good afternoon, I hope", 8),
        ("The weather today is", 6),
        ("How are you", 5),
        ("I would like to", 7),
        ("Dear Sir or Madam,", 10),
    ]
    
    for prompt, max_tokens in test_cases:
        print(f"\n--- Teste: '{prompt}' ---")
        
        suggestion = generate_word_autocomplete(prompt, max_words=4)
        
        if suggestion:
            print(f"✅ Sugestão: '{suggestion}'")
            print(f"💬 Completo: '{prompt} {suggestion}'")
        else:
            print("❌ Nenhuma sugestão gerada")
        
        print("-" * 50)


print("\n--- Diagnóstico do Modelo ---")
print(f"Modelo: {model_name}")
print(f"Vocab size: {tokenizer.vocab_size}")

test_autocomplete_scenarios()

print("\n=== TESTE INTERATIVO ===")
while True:
    try:
        user_input = input("\nDigite um texto para autocomplete (ou 'quit' para sair): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            suggestion = generate_word_autocomplete(user_input, max_words=3)
            if suggestion:
                print(f"💡 Sugestão: '{user_input} {suggestion}'")
            else:
                print("❌ Nenhuma sugestão disponível")
        
    except KeyboardInterrupt:
        print("\nSaindo...")
        break
    except Exception as e:
        print(f"Erro: {e}")
        continue