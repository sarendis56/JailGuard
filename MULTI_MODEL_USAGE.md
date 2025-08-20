# JailGuard Multi-Model Support

JailGuard now supports both MiniGPT-4 and LLaVA models.

## Quick Start

### 1. Configure Default Model
Set environment variable:
```bash
export JAILGUARD_MODEL=llava  # or minigpt4
```

Or edit `JailGuard/config/model_config.yaml`:
```yaml
default_model: 'llava'  # or 'minigpt4'
```

### 2. Run with Specific Model
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
