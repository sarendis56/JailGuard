"""
Configuration loader for JailGuard models

This module handles loading and managing configuration for different models.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ModelConfig:
    """Configuration manager for JailGuard models"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Look for config file relative to this module
        current_dir = Path(__file__).parent
        config_file = current_dir.parent / "config" / "model_config.yaml"
        
        if config_file.exists():
            return str(config_file)
        
        # Fallback: create default config
        return self._create_default_config()
    
    def _create_default_config(self) -> str:
        """Create a default configuration file"""
        config_dir = Path(__file__).parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "model_config.yaml"
        
        default_config = {
            'default_model': 'minigpt4',
            'minigpt4': {
                'config_path': './utils/minigpt4_eval.yaml',
                'gpu_id': '0'
            },
            'llava': {
                'model_path': 'liuhaotian/llava-v1.5-7b',
                'model_base': None,
                'conv_mode': None,
                'load_8bit': False,
                'load_4bit': False,
                'device': 'cuda',
                'temperature': 1.0,
                'max_new_tokens': 300,
                'do_sample': False
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return str(config_file)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            return self._get_default_config_dict()
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing config file {self.config_path}: {e}")
            return self._get_default_config_dict()
    
    def _get_default_config_dict(self) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        return {
            'default_model': 'minigpt4',
            'minigpt4': {
                'config_path': './utils/minigpt4_eval.yaml',
                'gpu_id': '0'
            },
            'llava': {
                'model_path': 'liuhaotian/llava-v1.5-7b',
                'model_base': None,
                'conv_mode': None,
                'load_8bit': False,
                'load_4bit': False,
                'device': 'cuda',
                'temperature': 1.0,
                'max_new_tokens': 300,
                'do_sample': False
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Override default model
        if 'JAILGUARD_MODEL' in os.environ:
            model_type = os.environ['JAILGUARD_MODEL'].lower()
            if model_type in ['minigpt4', 'llava']:
                self.config['default_model'] = model_type
        
        # Override LLaVA settings
        llava_config = self.config.setdefault('llava', {})
        
        env_mappings = {
            'LLAVA_MODEL_PATH': 'model_path',
            'LLAVA_MODEL_BASE': 'model_base',
            'LLAVA_CONV_MODE': 'conv_mode',
            'LLAVA_DEVICE': 'device'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                llava_config[config_key] = os.environ[env_var]
        
        # Boolean overrides
        bool_mappings = {
            'LLAVA_LOAD_8BIT': 'load_8bit',
            'LLAVA_LOAD_4BIT': 'load_4bit',
            'LLAVA_DO_SAMPLE': 'do_sample'
        }
        
        for env_var, config_key in bool_mappings.items():
            if env_var in os.environ:
                llava_config[config_key] = os.environ[env_var].lower() == 'true'
        
        # Numeric overrides
        if 'LLAVA_TEMPERATURE' in os.environ:
            try:
                llava_config['temperature'] = float(os.environ['LLAVA_TEMPERATURE'])
            except ValueError:
                print(f"Warning: Invalid LLAVA_TEMPERATURE value: {os.environ['LLAVA_TEMPERATURE']}")
        
        if 'LLAVA_MAX_NEW_TOKENS' in os.environ:
            try:
                llava_config['max_new_tokens'] = int(os.environ['LLAVA_MAX_NEW_TOKENS'])
            except ValueError:
                print(f"Warning: Invalid LLAVA_MAX_NEW_TOKENS value: {os.environ['LLAVA_MAX_NEW_TOKENS']}")
        
        # Override MiniGPT-4 settings
        minigpt4_config = self.config.setdefault('minigpt4', {})
        
        if 'MINIGPT4_CONFIG_PATH' in os.environ:
            minigpt4_config['config_path'] = os.environ['MINIGPT4_CONFIG_PATH']
        
        if 'MINIGPT4_GPU_ID' in os.environ:
            minigpt4_config['gpu_id'] = os.environ['MINIGPT4_GPU_ID']
    
    def get_default_model(self) -> str:
        """Get the default model type"""
        return self.config.get('default_model', 'minigpt4')
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.config.get(model_type, {})
    
    def get_llava_config(self) -> Dict[str, Any]:
        """Get LLaVA configuration"""
        return self.get_model_config('llava')
    
    def get_minigpt4_config(self) -> Dict[str, Any]:
        """Get MiniGPT-4 configuration"""
        return self.get_model_config('minigpt4')
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 JailGuard Model Configuration:")
        print(f"   Config file: {self.config_path}")
        print(f"   Default model: {self.get_default_model().upper()}")
        
        print("\n   MiniGPT-4 settings:")
        minigpt4_config = self.get_minigpt4_config()
        for key, value in minigpt4_config.items():
            print(f"     {key}: {value}")
        
        print("\n   LLaVA settings:")
        llava_config = self.get_llava_config()
        for key, value in llava_config.items():
            print(f"     {key}: {value}")
        
        print("\n   Environment overrides:")
        env_vars = [
            'JAILGUARD_MODEL', 'LLAVA_MODEL_PATH', 'LLAVA_MODEL_BASE',
            'LLAVA_CONV_MODE', 'LLAVA_LOAD_8BIT', 'LLAVA_LOAD_4BIT',
            'LLAVA_DEVICE', 'LLAVA_TEMPERATURE', 'LLAVA_MAX_NEW_TOKENS',
            'MINIGPT4_CONFIG_PATH', 'MINIGPT4_GPU_ID'
        ]
        
        active_overrides = {var: os.environ[var] for var in env_vars if var in os.environ}
        if active_overrides:
            for var, value in active_overrides.items():
                print(f"     {var}: {value}")
        else:
            print("     None")

# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> ModelConfig:
    """Get the global configuration instance"""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = ModelConfig(config_path)
    return _config_instance

def reload_config(config_path: Optional[str] = None):
    """Reload the configuration"""
    global _config_instance
    _config_instance = ModelConfig(config_path)
    return _config_instance

if __name__ == "__main__":
    config = get_config()
    config.print_config()
