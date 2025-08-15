"""
LLaVA utilities for JailGuard - Compatible interface with MiniGPT-4

This module provides LLaVA model loading and inference functions that are compatible
with the existing MiniGPT-4 interface used in JailGuard.
"""

import os
import sys
import argparse
import torch
import warnings
from typing import Tuple, Any, Optional
from PIL import Image

# Suppress common PyTorch warnings during model loading
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")

# Global variables to cache model components
_tokenizer = None
_model = None
_image_processor = None
_context_len = None
_conv_template = None

def initialize_model(model_path: Optional[str] = None,
                    model_base: Optional[str] = None,
                    conv_mode: Optional[str] = None,
                    load_8bit: bool = False,
                    load_4bit: bool = False,
                    device: str = "cuda") -> Tuple[Any, Any, Any]:
    """
    Initialize LLaVA model with MiniGPT-4 compatible interface

    Returns:
        Tuple[vis_processor, chat, model]: Compatible with MiniGPT-4 interface
    """
    global _tokenizer, _model, _image_processor, _context_len, _conv_template

    # Resolve model path - work from both main directory and JailGuard subdirectory
    if model_path is None:
        # Try different possible locations for the model
        possible_paths = [
            "./model/llava-v1.6-vicuna-7b",  # From JailGuard subdirectory
            "../model/llava-v1.6-vicuna-7b",  # From JailGuard subdirectory to parent
            "model/llava-v1.6-vicuna-7b",    # From main directory
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise RuntimeError(f"Could not find LLaVA model in any of these locations: {possible_paths}")

    print(f'Initializing LLaVA Chat from: {model_path}')
    
    # Add LLaVA to Python path - handle different working directories
    possible_llava_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'LLaVA'),  # From JailGuard/utils
        os.path.join(os.path.dirname(__file__), '..', '..', 'LLaVA'),  # From JailGuard/utils to main
        './LLaVA',  # From main directory
        '../LLaVA',  # From JailGuard subdirectory
    ]

    llava_path = None
    for path in possible_llava_paths:
        if os.path.exists(path):
            llava_path = os.path.abspath(path)
            break

    if llava_path is None:
        raise RuntimeError(f"LLaVA directory not found in any of these locations: {possible_llava_paths}")
    
    if llava_path not in sys.path:
        sys.path.insert(0, llava_path)
    
    try:
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path
        from llava.utils import disable_torch_init
        
        # Disable torch init for faster loading
        disable_torch_init()
        
        # Load the model
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit, device=device
        )
        
        # Determine conversation mode
        if conv_mode is None:
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"
        
        # Cache components globally
        _tokenizer = tokenizer
        _model = model
        _image_processor = image_processor
        _context_len = context_len
        _conv_template = conv_templates[conv_mode].copy()
        
        # Create a chat wrapper that's compatible with MiniGPT-4 interface
        chat_wrapper = LLaVAChatWrapper(tokenizer, model, image_processor, context_len, _conv_template)
        
        print('LLaVA Initialization Finished')
        return image_processor, chat_wrapper, model
        
    except ImportError as e:
        raise RuntimeError(f"Failed to import LLaVA modules: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLaVA model: {e}")


def model_inference(vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
    """
    Perform LLaVA inference with MiniGPT-4 compatible interface
    
    Args:
        vis_processor: Image processor (from initialize_model)
        chat: Chat wrapper (from initialize_model)
        model: LLaVA model (from initialize_model)
        prompts_eval: [question, image_path] format (same as MiniGPT-4)
    
    Returns:
        str: Model response
    """
    # Add LLaVA to Python path if not already there - handle different working directories
    possible_llava_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'LLaVA'),  # From JailGuard/utils
        os.path.join(os.path.dirname(__file__), '..', '..', 'LLaVA'),  # From JailGuard/utils to main
        './LLaVA',  # From main directory
        '../LLaVA',  # From JailGuard subdirectory
    ]

    llava_path = None
    for path in possible_llava_paths:
        if os.path.exists(path):
            llava_path = os.path.abspath(path)
            break

    if llava_path is None:
        raise RuntimeError(f"LLaVA directory not found in any of these locations: {possible_llava_paths}")

    if llava_path not in sys.path:
        sys.path.insert(0, llava_path)
    
    try:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import SeparatorStyle
        from llava.mm_utils import process_images, tokenizer_image_token
        
        # Extract question and image path from prompts_eval (MiniGPT-4 format)
        question = prompts_eval[0]
        image_path = prompts_eval[1]
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_size = image.size
        
        # Process image using LLaVA's image processor
        image_tensor = process_images([image], vis_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        # Get fresh conversation template
        conv = chat.conv_template.copy()
        
        # Format the input with image token
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        
        # Add to conversation
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, chat.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        # Generate response with memory management
        with torch.inference_mode():
            # Clear cache before generation to free memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True,  # Enable sampling for better responses
                temperature=0.2,  # Lower temperature for more focused responses
                top_p=None,
                num_beams=1,
                max_new_tokens=256,  # Reduce max tokens to save memory
                use_cache=True,
                pad_token_id=chat.tokenizer.eos_token_id,
                eos_token_id=chat.tokenizer.eos_token_id,
                stopping_criteria=None  # Don't use custom stopping criteria
            )

        # Decode response - extract only the generated tokens
        input_token_len = input_ids.shape[1]
        output_token_len = output_ids.shape[1]

        if output_token_len > input_token_len:
            # Normal case: model generated new tokens
            outputs = chat.tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=True).strip()
        else:
            # Edge case: decode full output and extract response
            outputs = chat.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            # Try to extract just the response part
            if "ASSISTANT:" in outputs:
                outputs = outputs.split("ASSISTANT:")[-1].strip()
            elif "Assistant:" in outputs:
                outputs = outputs.split("Assistant:")[-1].strip()

        # Clean up the response
        response = outputs.strip()

        # Remove any conversation artifacts
        if response.startswith("ASSISTANT:"):
            response = response[10:].strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()

        # Remove any trailing separators or special tokens
        for sep in ["</s>", "<|im_end|>", "<|endoftext|>"]:
            if response.endswith(sep):
                response = response[:-len(sep)].strip()

        return response
        
    except Exception as e:
        raise RuntimeError(f"LLaVA inference failed: {e}")


class LLaVAChatWrapper:
    """Wrapper class to make LLaVA components compatible with MiniGPT-4 chat interface"""
    
    def __init__(self, tokenizer, model, image_processor, context_len, conv_template):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_template = conv_template
    
    def upload_img(self, img):
        """Compatibility method - not used in current implementation"""
        pass
    
    def ask(self, user_message, chat_state):
        """Compatibility method - not used in current implementation"""
        pass
    
    def answer(self, conv, img_list, num_beams=1, temperature=1.0):
        """Compatibility method - not used in current implementation"""
        pass
    
    def encode_img(self, img_list):
        """Compatibility method - not used in current implementation"""
        pass
