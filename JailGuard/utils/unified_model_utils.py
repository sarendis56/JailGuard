"""
Unified Model Utilities for JailGuard

This module provides a unified interface that can dynamically switch between
MiniGPT-4 and LLaVA models while maintaining backward compatibility with
existing JailGuard code.
"""

import os
import sys
import torch
from typing import Tuple, Any, Optional, Dict

def get_config():
    """Get model configuration"""
    try:
        from config_loader import get_config
        return get_config()
    except ImportError:
        # Fallback to environment variables if config_loader not available
        return _get_fallback_config()

def _get_fallback_config():
    """Fallback configuration using environment variables"""
    class FallbackConfig:
        def get_default_model(self):
            return os.environ.get('JAILGUARD_MODEL', 'minigpt4').lower()

        def get_llava_config(self):
            return {
                'model_path': os.environ.get('LLAVA_MODEL_PATH', None),  # Will be resolved in llava_utils
                'model_base': os.environ.get('LLAVA_MODEL_BASE'),
                'conv_mode': os.environ.get('LLAVA_CONV_MODE'),
                'load_8bit': os.environ.get('LLAVA_LOAD_8BIT', 'false').lower() == 'true',
                'load_4bit': os.environ.get('LLAVA_LOAD_4BIT', 'false').lower() == 'true',
                'device': os.environ.get('LLAVA_DEVICE', 'cuda'),
                'temperature': float(os.environ.get('LLAVA_TEMPERATURE', '1.0')),
                'max_new_tokens': int(os.environ.get('LLAVA_MAX_NEW_TOKENS', '300')),
                'do_sample': os.environ.get('LLAVA_DO_SAMPLE', 'false').lower() == 'true'
            }

        def get_minigpt4_config(self):
            return {
                'config_path': os.environ.get('MINIGPT4_CONFIG_PATH', './utils/minigpt4_eval.yaml'),
                'gpu_id': os.environ.get('MINIGPT4_GPU_ID', '0')
            }

    return FallbackConfig()

def get_available_models() -> Dict[str, bool]:
    """Check which models are available"""
    available = {}
    
    # Check MiniGPT-4
    try:
        from minigpt_utils import initialize_model as minigpt_init
        available['minigpt4'] = True
    except ImportError:
        available['minigpt4'] = False
    
    # Check LLaVA
    try:
        from llava_utils import initialize_model as llava_init
        available['llava'] = True
    except ImportError:
        available['llava'] = False
    
    # Check Qwen
    try:
        from qwen_utils import initialize_model as qwen_init
        available['qwen'] = True
    except ImportError:
        available['qwen'] = False
    
    return available

def initialize_model(model_type: Optional[str] = None, **kwargs) -> Tuple[Any, Any, Any]:
    """
    Initialize a model using the unified interface

    Args:
        model_type: 'minigpt4' or 'llava'. If None, uses default from config
        **kwargs: Additional arguments for model initialization

    Returns:
        Tuple[vis_processor, chat, model]: Compatible with existing JailGuard interface
    """
    # Load configuration
    config = get_config()

    # Determine which model to use
    if model_type is None:
        model_type = config.get_default_model()

    if model_type is not None:
        model_type = model_type.lower()

    # Check availability
    available_models = get_available_models()

    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}")

    if not available_models[model_type]:
        # Try to fallback to available model
        if model_type == 'llava' and available_models['minigpt4']:
            print(f"⚠ LLaVA not available, falling back to MiniGPT-4")
            model_type = 'minigpt4'
        elif model_type == 'minigpt4' and available_models['llava']:
            print(f"⚠ MiniGPT-4 not available, falling back to LLaVA")
            model_type = 'llava'
        else:
            raise RuntimeError(f"Model {model_type} is not available and no fallback found")

    print(f"🤖 Initializing {model_type.upper()} model...")

    if model_type == 'minigpt4':
        return _initialize_minigpt4(config, **kwargs)
    elif model_type == 'llava':
        return _initialize_llava(config, **kwargs)
    elif model_type == 'qwen':
        return _initialize_qwen(config, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def model_inference(vis_processor: Any, chat: Any, model: Any, prompts_eval: list, 
                   model_type: Optional[str] = None) -> str:
    """
    Perform model inference using the unified interface
    
    Args:
        vis_processor: Visual processor from initialize_model
        chat: Chat object from initialize_model
        model: Model from initialize_model
        prompts_eval: [question, image_path] format
        model_type: 'minigpt4' or 'llava'. If None, auto-detects from chat object
    
    Returns:
        str: Model response
    """
    # Auto-detect model type if not specified
    if model_type is None:
        model_type = _detect_model_type(chat)
    
    model_type = model_type.lower()
    
    if model_type == 'minigpt4':
        return _inference_minigpt4(vis_processor, chat, model, prompts_eval)
    elif model_type == 'llava':
        return _inference_llava(vis_processor, chat, model, prompts_eval)
    elif model_type == 'qwen':
        return _inference_qwen(vis_processor, chat, model, prompts_eval)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def _detect_model_type(chat: Any) -> str:
    """Auto-detect model type from chat object"""
    # Check if it's Qwen wrapper (more specific check first)
    if hasattr(chat, 'processor') and hasattr(chat, 'tokenizer') and hasattr(chat, 'model'):
        # Additional check: if it has a processor attribute, it's likely Qwen
        if hasattr(chat.processor, 'apply_chat_template'):
            return 'qwen'
    # Check if it's LLaVA wrapper
    if hasattr(chat, 'tokenizer') and hasattr(chat, 'conv_template') and chat.conv_template is not None:
        return 'llava'
    # Otherwise assume MiniGPT-4
    else:
        return 'minigpt4'

def _initialize_minigpt4(config, **kwargs) -> Tuple[Any, Any, Any]:
    """Initialize MiniGPT-4 model"""
    try:
        # Try different import paths
        try:
            from minigpt_utils import initialize_model as minigpt_init
        except ImportError:
            from utils.minigpt_utils import initialize_model as minigpt_init
        return minigpt_init()
    except ImportError as e:
        raise RuntimeError(f"MiniGPT-4 not available: {e}")

def _initialize_llava(config, **kwargs) -> Tuple[Any, Any, Any]:
    """Initialize LLaVA model"""
    try:
        # Try different import paths
        try:
            from llava_utils import initialize_model as llava_init
        except ImportError:
            from utils.llava_utils import initialize_model as llava_init

        # Use configuration values, but allow kwargs to override
        llava_config = config.get_llava_config()
        llava_kwargs = {
            'model_path': llava_config.get('model_path', None),  # Will be resolved in llava_utils
            'model_base': llava_config.get('model_base'),
            'conv_mode': llava_config.get('conv_mode'),
            'load_8bit': llava_config.get('load_8bit', False),
            'load_4bit': llava_config.get('load_4bit', False),
            'device': llava_config.get('device', 'cuda')
        }
        llava_kwargs.update(kwargs)

        return llava_init(**llava_kwargs)
    except ImportError as e:
        raise RuntimeError(f"LLaVA not available: {e}")

def _inference_minigpt4(vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
    """Perform MiniGPT-4 inference"""
    try:
        try:
            from minigpt_utils import model_inference as minigpt_inference
        except ImportError:
            from utils.minigpt_utils import model_inference as minigpt_inference
        return minigpt_inference(vis_processor, chat, model, prompts_eval)
    except ImportError as e:
        raise RuntimeError(f"MiniGPT-4 not available: {e}")

def _inference_llava(vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
    """Perform LLaVA inference"""
    try:
        try:
            from llava_utils import model_inference as llava_inference
        except ImportError:
            from utils.llava_utils import model_inference as llava_inference
        return llava_inference(vis_processor, chat, model, prompts_eval)
    except ImportError as e:
        raise RuntimeError(f"LLaVA not available: {e}")

def _initialize_qwen(config, **kwargs) -> Tuple[Any, Any, Any]:
    """Initialize Qwen model"""
    try:
        # Try different import paths
        try:
            from qwen_utils import initialize_model as qwen_init
        except ImportError:
            from utils.qwen_utils import initialize_model as qwen_init

        # Use configuration values, but allow kwargs to override
        qwen_kwargs = {
            'device': os.environ.get('QWEN_DEVICE', 'cuda'),
            'torch_dtype': torch.float16,
            'trust_remote_code': True
        }
        qwen_kwargs.update(kwargs)

        return qwen_init(**qwen_kwargs)
    except ImportError as e:
        raise RuntimeError(f"Qwen not available: {e}")

def _inference_qwen(vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
    """Perform Qwen inference"""
    try:
        try:
            from qwen_utils import model_inference as qwen_inference
        except ImportError:
            from utils.qwen_utils import model_inference as qwen_inference
        return qwen_inference(vis_processor, chat, model, prompts_eval)
    except ImportError as e:
        raise RuntimeError(f"Qwen not available: {e}")

def print_model_info():
    """Print information about available models and current configuration"""
    config = get_config()
    available = get_available_models()

    print("🤖 JailGuard Model Configuration:")
    print(f"   Default model: {config.get_default_model().upper()}")
    print(f"   Available models:")
    for model, avail in available.items():
        status = "✓" if avail else "✗"
        print(f"     {status} {model.upper()}")

    if available.get('llava', False):
        llava_config = config.get_llava_config()
        print(f"   LLaVA configuration:")
        print(f"     Model path: {llava_config.get('model_path', 'N/A')}")
        print(f"     Model base: {llava_config.get('model_base', 'N/A')}")
        print(f"     Device: {llava_config.get('device', 'N/A')}")
        print(f"     8-bit: {llava_config.get('load_8bit', False)}")
        print(f"     4-bit: {llava_config.get('load_4bit', False)}")

    if hasattr(config, 'print_config'):
        print("\n" + "="*50)
        config.print_config()

# Backward compatibility aliases
def get_method(method_name):
    """Backward compatibility for augmentation methods"""
    try:
        from augmentations import img_aug_dict
        return img_aug_dict[method_name]
    except ImportError:
        raise RuntimeError("Augmentation methods not available")

if __name__ == "__main__":
    print_model_info()
