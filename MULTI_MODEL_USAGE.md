# JailGuard Multi-Model Support

JailGuard now supports both MiniGPT-4 and LLaVA models! This guide explains how to use the new multi-model functionality.

## Quick Start

### 1. Check Available Models
```bash
python test_model_integration.py
```

### 2. Configure Default Model
Set environment variable:
```bash
export JAILGUARD_MODEL=llava  # or minigpt4
```

Or edit `JailGuard/config/model_config.yaml`:
```yaml
default_model: 'llava'  # or 'minigpt4'
```

### 3. Run with Specific Model
```bash
# Using main_img.py
cd JailGuard
python main_img.py --serial_num 287 --model llava

# Using systematic testing
python systematic_test_jailguard.py --dataset VQAv2 --max-samples 10 --model llava

# Using batch testing
python batch_test_runner.py --model llava --safe-only
```

## Model Configuration

### Environment Variables
- `JAILGUARD_MODEL`: Default model (`minigpt4` or `llava`)
- `LLAVA_MODEL_PATH`: LLaVA model path (default: `liuhaotian/llava-v1.5-7b`)
- `LLAVA_DEVICE`: Device for LLaVA (default: `cuda`)
- `LLAVA_LOAD_8BIT`: Enable 8-bit quantization (`true`/`false`)
- `LLAVA_LOAD_4BIT`: Enable 4-bit quantization (`true`/`false`)

### Configuration File
Edit `JailGuard/config/model_config.yaml`:

```yaml
# Default model to use
default_model: 'llava'

# LLaVA Configuration
llava:
  model_path: 'liuhaotian/llava-v1.5-7b'
  model_base: null
  conv_mode: null  # auto-detected
  load_8bit: false
  load_4bit: false
  device: 'cuda'
  temperature: 1.0
  max_new_tokens: 300
  do_sample: false

# MiniGPT-4 Configuration  
minigpt4:
  config_path: './utils/minigpt4_eval.yaml'
  gpu_id: '0'
```

## Usage Examples

### Basic Usage
```bash
# Use default model (from config)
python main_img.py --serial_num 287

# Use specific model
python main_img.py --serial_num 287 --model llava
python main_img.py --serial_num 287 --model minigpt4
```

### Systematic Testing
```bash
# Test with LLaVA
python systematic_test_jailguard.py \
  --dataset VQAv2 \
  --max-samples 50 \
  --model llava \
  --output-dir llava_results

# Test with MiniGPT-4
python systematic_test_jailguard.py \
  --dataset VQAv2 \
  --max-samples 50 \
  --model minigpt4 \
  --output-dir minigpt4_results
```

### Batch Testing
```bash
# Run all tests with LLaVA
python batch_test_runner.py --model llava

# Run only safe datasets with MiniGPT-4
python batch_test_runner.py --model minigpt4 --safe-only
```

## Model Comparison

| Feature | MiniGPT-4 | LLaVA |
|---------|-----------|-------|
| **Initialization** | Faster | Slower |
| **Memory Usage** | Lower | Higher |
| **Performance** | Established | More recent |
| **Quantization** | Limited | 8-bit/4-bit support |
| **Conversation** | Custom | Multiple templates |

## Environment Setup

### For LLaVA
```bash
# Activate LLaVA environment
conda activate llava

# Verify LLaVA installation
cd LLaVA
python -c "from llava.model.builder import load_pretrained_model; print('LLaVA OK')"
```

### For MiniGPT-4
```bash
# Activate MiniGPT-4 environment  
conda activate minigptv

# Verify MiniGPT-4 installation
cd JailGuard
python -c "from utils.minigpt_utils import initialize_model; print('MiniGPT-4 OK')"
```

## Troubleshooting

### Model Not Available
If you get "Model not available" errors:

1. **Check environment**: Make sure you're in the correct conda environment
2. **Check paths**: Verify model paths in configuration
3. **Check imports**: Run the test script to diagnose import issues

```bash
python test_model_integration.py
```

### Memory Issues
For memory-constrained systems:

```bash
# Use quantization for LLaVA
export LLAVA_LOAD_8BIT=true
# or
export LLAVA_LOAD_4BIT=true
```

### Performance Issues
- **LLaVA**: First run is slower due to model download
- **MiniGPT-4**: Faster initialization, established pipeline

## Backward Compatibility

The integration maintains full backward compatibility:
- Existing scripts work without modification
- Default behavior unchanged (uses MiniGPT-4)
- All existing command-line arguments preserved

## Advanced Usage

### Custom Model Paths
```bash
# Use local LLaVA model
export LLAVA_MODEL_PATH=/path/to/local/llava/model

# Use different conversation template
export LLAVA_CONV_MODE=llava_v1
```

### Programmatic Usage
```python
from JailGuard.utils.unified_model_utils import initialize_model, model_inference

# Initialize specific model
vis_processor, chat, model = initialize_model(model_type='llava')

# Perform inference
response = model_inference(vis_processor, chat, model, [question, image_path])
```

## Support

For issues or questions:
1. Run the integration test: `python test_model_integration.py`
2. Check model availability and configuration
3. Verify environment setup for the specific model
4. Check memory and GPU requirements
