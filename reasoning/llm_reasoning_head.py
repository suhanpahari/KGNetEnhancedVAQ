"""LLM Reasoning Head with LoRA."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMReasoningHead(nn.Module):
    """LLM-based reasoning with LoRA."""

    def __init__(self, llm_model='meta-llama/Llama-3-8B-Instruct', num_answers=3129, 
                 use_lora=True, lora_r=16, lora_alpha=32, load_in_8bit=True):
        super().__init__()
        
        logger.info(f"Loading LLM: {llm_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=quantization_config,
                device_map='auto',
                torch_dtype=torch.float16
            )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model,
                device_map='auto',
                torch_dtype=torch.float16
            )
        
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            logger.info("LoRA enabled")
        
        self.feature_projection = nn.Linear(1024, self.llm.config.hidden_size)
        self.answer_classifier = nn.Linear(self.llm.config.hidden_size, num_answers)

    def forward(self, fused_features, questions, mode='classify'):
        """
        Args:
            fused_features: (B, 1024)
            questions: List of question strings
            mode: 'classify' or 'generate'
        """
        feature_embeds = self.feature_projection(fused_features).unsqueeze(1)  # (B, 1, hidden)
        
        if mode == 'classify':
            prompt = "Answer the question: " + questions[0] if isinstance(questions, list) else questions
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(feature_embeds.device)
            prompt_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
            
            combined_embeds = torch.cat([feature_embeds, prompt_embeds], dim=1)
            
            outputs = self.llm(inputs_embeds=combined_embeds, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            
            logits = self.answer_classifier(last_hidden)
            return logits
        else:
            # Generation mode
            prompt = "Question: " + questions[0] + "\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors='pt').to(feature_embeds.device)
            prompt_embeds = self.llm.get_input_embeddings()(inputs.input_ids)
            
            combined_embeds = torch.cat([feature_embeds, prompt_embeds], dim=1)
            
            generated = self.llm.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True
            )
            
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
