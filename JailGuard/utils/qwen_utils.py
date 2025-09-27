"""
Qwen utilities for JailGuard - Compatible interface with MiniGPT-4 and LLaVA

This module provides Qwen model loading and inference functions that are compatible
with the existing MiniGPT-4/LLaVA interface used in JailGuard.
"""

import os
import sys
import argparse
import torch
import warnings
import tempfile
from typing import Tuple, Any, Optional
from PIL import Image

# Suppress common PyTorch warnings during model loading
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*")

# Global variables to cache model components
_tokenizer = None
_model = None
_processor = None

def initialize_model(model_path: Optional[str] = None,
                    device: str = "cuda",
                    torch_dtype: torch.dtype = torch.float16,
                    trust_remote_code: bool = True) -> Tuple[Any, Any, Any]:
    """
    Initialize Qwen model with MiniGPT-4/LLaVA compatible interface

    Returns:
        Tuple[vis_processor, chat, model]: Compatible with MiniGPT-4/LLaVA interface
    """
    global _tokenizer, _model, _processor

    # Resolve model path - work from both main directory and JailGuard subdirectory
    if model_path is None:
        # Get the current working directory to understand where we are
        current_cwd = os.getcwd()
        
        # Try different possible locations for the model based on current working directory
        possible_paths = []
        
        # If we're in the JailGuard subdirectory, try paths relative to parent
        if current_cwd.endswith('/JailGuard'):
            possible_paths.extend([
                '../model/qwen2.5-vl-7b-instruct',  # Specific model first
                './model/qwen2.5-vl-7b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-7b-instruct',  # Absolute path
                '../model/qwen2.5-vl-3b-instruct',  # Alternative model sizes
                './model/qwen2.5-vl-3b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-3b-instruct',
                '../model/qwen2.5-vl-14b-instruct',
                './model/qwen2.5-vl-14b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-14b-instruct',
                '../model/qwen2.5-vl-72b-instruct',
                './model/qwen2.5-vl-72b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-72b-instruct',
                '../model',  # Parent directory (check for any qwen subdirs)
                './model',   # Current directory (check for any qwen subdirs)
                '/workspace/JailGuard/model'  # Absolute path (check for any qwen subdirs)
            ])
        else:
            # If we're in the main directory
            possible_paths.extend([
                './model/qwen2.5-vl-7b-instruct',  # Specific model first
                '../model/qwen2.5-vl-7b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-7b-instruct',  # Absolute path
                './model/qwen2.5-vl-3b-instruct',  # Alternative model sizes
                '../model/qwen2.5-vl-3b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-3b-instruct',
                './model/qwen2.5-vl-14b-instruct',
                '../model/qwen2.5-vl-14b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-14b-instruct',
                './model/qwen2.5-vl-72b-instruct',
                '../model/qwen2.5-vl-72b-instruct',
                '/workspace/JailGuard/model/qwen2.5-vl-72b-instruct',
                './model',  # Current directory (check for any qwen subdirs)
                '../model',  # Parent directory (check for any qwen subdirs)
                '/workspace/JailGuard/model'  # Absolute path (check for any qwen subdirs)
            ])
        
        # Find the first valid model path that contains model files
        print(f"Searching for Qwen model in {len(possible_paths)} possible locations...")
        for i, path in enumerate(possible_paths):
            abs_path = os.path.abspath(path)
            print(f"  {i+1}. Checking: {abs_path}")
            if os.path.exists(abs_path):
                print(f"     ✓ Path exists")
                # Check if this directory contains Qwen model files
                model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
                has_model_files = any(os.path.exists(os.path.join(abs_path, f)) for f in model_files)
                
                # Additional check: verify this is actually a Qwen model by checking config.json
                is_qwen_model = False
                if has_model_files:
                    config_path = os.path.join(abs_path, 'config.json')
                    if os.path.exists(config_path):
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            # Check if this is a Qwen model
                            model_type = config.get('model_type', '').lower()
                            if 'qwen' in model_type or 'qwen2' in model_type:
                                is_qwen_model = True
                                print(f"     ✓ Found Qwen model files in: {abs_path} (type: {model_type})")
                            else:
                                print(f"     ✗ Found model files but not Qwen (type: {model_type})")
                        except Exception as e:
                            print(f"     ✗ Error reading config.json: {e}")
                    else:
                        print(f"     ✗ Found model files but no config.json")
                
                if has_model_files and is_qwen_model:
                    model_path = abs_path
                    break
                    
                # Also check subdirectories for Qwen models
                try:
                    for subdir in os.listdir(abs_path):
                        subdir_path = os.path.join(abs_path, subdir)
                        if os.path.isdir(subdir_path):
                            has_model_files = any(os.path.exists(os.path.join(subdir_path, f)) for f in model_files)
                            if has_model_files:
                                # Check if this is a Qwen model
                                config_path = os.path.join(subdir_path, 'config.json')
                                if os.path.exists(config_path):
                                    try:
                                        import json
                                        with open(config_path, 'r') as f:
                                            config = json.load(f)
                                        model_type = config.get('model_type', '').lower()
                                        if 'qwen' in model_type or 'qwen2' in model_type:
                                            print(f"     ✓ Found Qwen model files in subdirectory: {subdir_path} (type: {model_type})")
                                            model_path = subdir_path
                                            break
                                        else:
                                            print(f"     ✗ Found model files in {subdir} but not Qwen (type: {model_type})")
                                    except Exception as e:
                                        print(f"     ✗ Error reading config.json in {subdir}: {e}")
                                else:
                                    print(f"     ✗ Found model files in {subdir} but no config.json")
                    if model_path:
                        break
                except PermissionError:
                    print(f"     ✗ Permission denied accessing subdirectories")
            else:
                print(f"     ✗ Path does not exist")
        
        if model_path is None:
            raise FileNotFoundError(
                f"Could not find Qwen model with required files (pytorch_model.bin, model.safetensors, or config.json).\n"
                f"Tried paths: {possible_paths}\n"
                f"Please specify --model-path or place model in one of the expected locations."
            )

    print(f"Loading Qwen model from: {model_path}")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError(
            f"Qwen dependencies not available: {e}\n"
            f"Please install: pip install qwen-vl-utils transformers>=4.37.0"
        )

    # Load model
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code
    )

    # Load processor
    try:
        _processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    except Exception:
        _processor = AutoProcessor.from_pretrained(model_path)

    # Get tokenizer from processor
    _tokenizer = _processor.tokenizer

    # Set model to eval mode
    _model.eval()

    print(f"Qwen model loaded successfully on device: {_model.device}")

    # Create a chat wrapper that mimics the MiniGPT-4/LLaVA interface
    class QwenChatWrapper:
        def __init__(self, model, processor, tokenizer):
            self.model = model
            self.processor = processor
            self.tokenizer = tokenizer
            self.conv_template = None  # For compatibility

    chat_wrapper = QwenChatWrapper(_model, _processor, _tokenizer)

    # Return in the same format as MiniGPT-4/LLaVA: (vis_processor, chat, model)
    return _processor, chat_wrapper, _model

def model_inference(vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
    """
    Perform Qwen model inference with MiniGPT-4/LLaVA compatible interface
    
    Args:
        vis_processor: Image processor (from initialize_model)
        chat: Chat wrapper (from initialize_model)
        model: Qwen model (from initialize_model)
        prompts_eval: [question, image_path] format (image_path can be None for text-only)
    
    Returns:
        str: Model response
    """
    if len(prompts_eval) != 2:
        raise ValueError("prompts_eval must be [question, image_path] format")
    
    question, image_path = prompts_eval
    
    # Prepare messages for Qwen2.5-VL
    if image_path is not None:
        # Multimodal input with image
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]}]
    else:
        # Text-only input
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question}
        ]}]

    try:
        from qwen_vl_utils import process_vision_info
        
        # Process the messages using the processor
        text = vis_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if image_path is not None:
            # Multimodal processing
            vision_info = process_vision_info(messages)
            image_inputs = vision_info[0] if len(vision_info) > 0 else None
            video_inputs = vision_info[1] if len(vision_info) > 1 else None

            inputs = vis_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Text-only processing
            inputs = vis_processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,  # Reduced for faster generation in jailbreak detection
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                eos_token_id=vis_processor.tokenizer.eos_token_id,
                pad_token_id=vis_processor.tokenizer.pad_token_id
            )

        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = vis_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return response.strip()

    except Exception as e:
        print(f"Error during Qwen inference: {e}")
        return "Error: Could not generate response"

def cleanup_model():
    """Clean up model from memory"""
    global _tokenizer, _model, _processor
    
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test the Qwen utilities
    print("Testing Qwen utilities...")
    
    try:
        vis_processor, chat, model = initialize_model()
        print("✓ Qwen model initialized successfully")
        
        # Test with a simple example
        test_prompts = ["What do you see in this image?", "test_image.jpg"]
        print("Testing inference...")
        response = model_inference(vis_processor, chat, model, test_prompts)
        print(f"Response: {response}")
        
        cleanup_model()
        print("✓ Qwen utilities test completed")
        
    except Exception as e:
        print(f"✗ Qwen utilities test failed: {e}")
        sys.exit(1)
