"""
Model Manager for JailGuard - Unified interface for MiniGPT-4 and LLaVA models

This module provides a unified interface to work with both MiniGPT-4 and LLaVA models
while maintaining backward compatibility with existing JailGuard code.
"""

import os
import sys
from typing import Tuple, Any, Optional, Dict
from abc import ABC, abstractmethod
from enum import Enum

class ModelType(Enum):
    MINIGPT4 = "minigpt4"
    LLAVA = "llava"

class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def initialize_model(self) -> Tuple[Any, Any, Any]:
        """Initialize the model and return (vis_processor, chat, model)"""
        pass
    
    @abstractmethod
    def model_inference(self, vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
        """Perform model inference with the given inputs"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available for use"""
        pass

class ModelManager:
    """Unified model manager that can handle both MiniGPT-4 and LLaVA"""
    
    def __init__(self, model_type: ModelType = ModelType.MINIGPT4, **kwargs):
        self.model_type = model_type
        self.model_interface = None
        self.kwargs = kwargs
        self._initialize_interface()
    
    def _initialize_interface(self):
        """Initialize the appropriate model interface"""
        if self.model_type == ModelType.MINIGPT4:
            self.model_interface = MiniGPT4Interface(**self.kwargs)
        elif self.model_type == ModelType.LLAVA:
            self.model_interface = LLaVAInterface(**self.kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def initialize_model(self) -> Tuple[Any, Any, Any]:
        """Initialize the model using the appropriate interface"""
        return self.model_interface.initialize_model()
    
    def model_inference(self, vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
        """Perform model inference using the appropriate interface"""
        return self.model_interface.model_inference(vis_processor, chat, model, prompts_eval)
    
    def is_available(self) -> bool:
        """Check if the current model is available"""
        return self.model_interface.is_available()
    
    @classmethod
    def get_available_models(cls) -> Dict[str, bool]:
        """Get a dictionary of available models"""
        return {
            ModelType.MINIGPT4.value: MiniGPT4Interface().is_available(),
            ModelType.LLAVA.value: LLaVAInterface().is_available()
        }

class MiniGPT4Interface(BaseModelInterface):
    """Interface for MiniGPT-4 model"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def is_available(self) -> bool:
        """Check if MiniGPT-4 is available"""
        try:
            from minigpt_utils import initialize_model, model_inference
            return True
        except ImportError:
            return False
    
    def initialize_model(self) -> Tuple[Any, Any, Any]:
        """Initialize MiniGPT-4 model"""
        try:
            from minigpt_utils import initialize_model
            return initialize_model()
        except ImportError as e:
            raise RuntimeError(f"MiniGPT-4 not available: {e}")
    
    def model_inference(self, vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
        """Perform MiniGPT-4 inference"""
        try:
            from minigpt_utils import model_inference
            return model_inference(vis_processor, chat, model, prompts_eval)
        except ImportError as e:
            raise RuntimeError(f"MiniGPT-4 not available: {e}")

class LLaVAInterface(BaseModelInterface):
    """Interface for LLaVA model"""
    
    def __init__(self, model_path: Optional[str] = None, model_base: Optional[str] = None, 
                 conv_mode: Optional[str] = None, load_8bit: bool = False, load_4bit: bool = False,
                 device: str = "cuda", **kwargs):
        self.model_path = model_path or "liuhaotian/llava-v1.5-7b"
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        self.kwargs = kwargs
        
        # Cache for loaded model components
        self._tokenizer = None
        self._model = None
        self._image_processor = None
        self._context_len = None
        self._conv_template = None
    
    def is_available(self) -> bool:
        """Check if LLaVA is available"""
        try:
            # Check if LLaVA directory exists and has the required modules
            llava_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'LLaVA')
            if not os.path.exists(llava_path):
                return False
            
            # Try importing LLaVA modules
            sys.path.insert(0, llava_path)
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
            from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
            return True
        except ImportError:
            return False
        finally:
            # Clean up sys.path
            if llava_path in sys.path:
                sys.path.remove(llava_path)
    
    def initialize_model(self) -> Tuple[Any, Any, Any]:
        """Initialize LLaVA model and return components compatible with MiniGPT-4 interface"""
        if not self.is_available():
            raise RuntimeError("LLaVA not available")
        
        try:
            # Add LLaVA to path
            llava_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'LLaVA')
            if llava_path not in sys.path:
                sys.path.insert(0, llava_path)
            
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path
            from llava.utils import disable_torch_init
            
            # Disable torch init for faster loading
            disable_torch_init()
            
            # Load the model
            model_name = get_model_name_from_path(self.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                self.model_path, self.model_base, model_name, 
                self.load_8bit, self.load_4bit, device=self.device
            )
            
            # Determine conversation mode
            if self.conv_mode is None:
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
            else:
                conv_mode = self.conv_mode
            
            # Cache components
            self._tokenizer = tokenizer
            self._model = model
            self._image_processor = image_processor
            self._context_len = context_len
            self._conv_template = conv_templates[conv_mode].copy()
            
            # Return in MiniGPT-4 compatible format: (vis_processor, chat, model)
            # We'll use a wrapper object that contains all LLaVA components
            llava_wrapper = LLaVAWrapper(
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                context_len=context_len,
                conv_template=self._conv_template
            )
            
            return image_processor, llava_wrapper, model

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLaVA model: {e}")

    def model_inference(self, vis_processor: Any, chat: Any, model: Any, prompts_eval: list) -> str:
        """Perform LLaVA inference with MiniGPT-4 compatible interface"""
        if not self.is_available():
            raise RuntimeError("LLaVA not available")

        try:
            # Add LLaVA to path
            llava_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'LLaVA')
            if llava_path not in sys.path:
                sys.path.insert(0, llava_path)

            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from llava.conversation import SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token
            from PIL import Image
            import torch

            # Extract question and image path from prompts_eval (MiniGPT-4 format)
            question = prompts_eval[0]
            image_path = prompts_eval[1]

            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image_size = image.size

            # Process image using LLaVA's image processor (vis_processor is the image_processor)
            image_tensor = process_images([image], vis_processor, model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # Get conversation template from chat wrapper
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

            # Generate response
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,  # Use deterministic generation for consistency
                    temperature=1.0,
                    max_new_tokens=150,  # Reduced for faster generation in jailbreak detection
                    use_cache=True
                )

            # Decode response
            outputs = chat.tokenizer.decode(output_ids[0]).strip()

            # Extract only the assistant's response (remove the prompt part)
            if conv.sep_style == SeparatorStyle.TWO:
                sep = conv.sep2
            else:
                sep = conv.sep

            # Find the assistant's response
            assistant_start = outputs.find(conv.roles[1] + ":")
            if assistant_start != -1:
                response_start = assistant_start + len(conv.roles[1]) + 1
                response = outputs[response_start:].strip()
                # Remove any trailing separator
                if sep and response.endswith(sep):
                    response = response[:-len(sep)].strip()
            else:
                # Fallback: try to extract response after the last occurrence of the separator
                parts = outputs.split(sep)
                response = parts[-1].strip() if parts else outputs.strip()

            return response

        except Exception as e:
            raise RuntimeError(f"LLaVA inference failed: {e}")


class LLaVAWrapper:
    """Wrapper class to make LLaVA components compatible with MiniGPT-4 interface"""

    def __init__(self, tokenizer, model, image_processor, context_len, conv_template):
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.conv_template = conv_template


# Convenience functions for backward compatibility
def initialize_model(model_type: str = "minigpt4", **kwargs) -> Tuple[Any, Any, Any]:
    """Initialize a model using the unified interface"""
    if model_type.lower() == "minigpt4":
        manager = ModelManager(ModelType.MINIGPT4, **kwargs)
    elif model_type.lower() == "llava":
        manager = ModelManager(ModelType.LLAVA, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return manager.initialize_model()


def model_inference(vis_processor: Any, chat: Any, model: Any, prompts_eval: list,
                   model_type: str = "minigpt4") -> str:
    """Perform model inference using the unified interface"""
    if model_type.lower() == "minigpt4":
        interface = MiniGPT4Interface()
    elif model_type.lower() == "llava":
        interface = LLaVAInterface()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return interface.model_inference(vis_processor, chat, model, prompts_eval)
