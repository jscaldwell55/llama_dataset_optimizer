# 🦙 Llama Dataset Optimizer

A comprehensive tool for optimizing datasets for LLaMA model training using intelligent filtering, deduplication, and scoring techniques.

## 🚀 Features

### 🔍 Quality Filtering
- **Conversation validation**: Ensures proper user/assistant message structure
- **Token-based filtering**: Configurable min/max token counts per message and total
- **Refusal pattern detection**: Automatically removes samples with AI refusal patterns
- **Balanced conversation ratios**: Maintains healthy user/assistant token balance
- **Turn-based filtering**: Controls conversation length and complexity

### 🎯 Semantic Deduplication
- **GPU-accelerated similarity search**: Uses FAISS for efficient near-duplicate detection
- **Sentence transformer embeddings**: High-quality semantic embeddings for similarity matching
- **Configurable similarity thresholds**: Fine-tune deduplication sensitivity
- **Memory-efficient processing**: Batch processing for large datasets

### 📊 Learning Value Scoring
- **Model-based scoring**: Uses actual LLaMA models to evaluate learning potential
- **Multi-factor scoring**: Combines learning value, quality, and diversity metrics
- **Configurable weights**: Adjust importance of different scoring factors
- **Smart selection**: Top-K selection based on composite scores

### ⚙️ Hardware Optimization
- **Multi-device support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU
- **Memory optimization**: 4-bit quantization and efficient batching
- **Flash Attention support**: Optional flash attention for supported hardware
- **Batch processing**: Configurable batch sizes for different hardware capabilities

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 8GB+ RAM (16GB+ recommended for large datasets)
- GPU with 6GB+ VRAM (optional but recommended)

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd llama_dataset_optimizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Install Flash Attention** (for A100+ performance):
```bash
pip install flash-attn --no-build-isolation
```

## 🎮 Usage

### Basic Usage

```bash
python llama_dataset_optimizer.py \
    --dataset your_dataset.jsonl \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output optimized_output \
    --config llama_3_2_instruct \
    --top-k 10000
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to input dataset (JSONL format) | Required |
| `--model` | HuggingFace model name for scoring | Required |
| `--output` | Output directory for optimized dataset | Required |
| `--top-k` | Number of top samples to select | Required |
| `--config` | Configuration file (without .yaml extension) | `llama_3_2_instruct` |
| `--validate` | Run validation after optimization | `False` |
| `--test-model` | Model for validation testing | `None` |
| `--skip-deduplication` | Skip the deduplication step | `False` |

### Example Commands

**Basic optimization with default settings:**
```bash
python llama_dataset_optimizer.py \
    --dataset data/training_data.jsonl \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output results/optimized \
    --top-k 5000
```

**Skip deduplication for faster processing:**
```bash
python llama_dataset_optimizer.py \
    --dataset data/training_data.jsonl \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output results/optimized \
    --top-k 5000 \
    --skip-deduplication
```

**With validation:**
```bash
python llama_dataset_optimizer.py \
    --dataset data/training_data.jsonl \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output results/optimized \
    --top-k 5000 \
    --validate \
    --test-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## 📁 Data Format

The tool supports datasets in JSONL format with the following structures:

### LLaMA/ChatML Format (Recommended)
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Alpaca Format (Auto-converted)
```json
{
  "instruction": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

## ⚙️ Configuration

### Available Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `llama_3_2_instruct` | Optimized for LLaMA 3.2 Instruct models | General instruction following |
| `llama_3_2_instruct_optimized` | Aggressive optimization settings | Large datasets needing heavy filtering |
| `llama_3_1_base` | Settings for LLaMA 3.1 base models | Base model fine-tuning |
| `test_config` | Lenient settings for testing | Small datasets or testing |

### Custom Configuration

Create a new YAML file in the `configs/` directory:

```yaml
# configs/my_custom_config.yaml
model_family: "Custom"

# Quality filtering settings
quality_filters:
  min_turns: 1
  max_turns: 10
  min_tokens_per_turn: 10
  max_tokens_per_turn: 3072
  min_total_tokens: 50
  max_total_tokens: 4096
  must_have_roles: ["user", "assistant"]
  check_for_refusal: true
  refusal_patterns:
    - "i'm sorry"
    - "i cannot"
    - "i am unable"
  balanced_conversation_ratio: 0.3

# Deduplication settings
deduplication:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  similarity_threshold: 0.95
  method: "sentence_transformer"

# Scoring weights
scoring_weights:
  learning_value: 0.5
  quality: 0.3
  diversity: 0.2

# Batch sizes for different hardware
batch_sizes:
  quality_filtering: 512
  embedding: 256
  scoring: 64
```

## 🔧 Advanced Features

### Python API

```python
from llama_dataset_optimizer import LlamaDatasetOptimizer

# Initialize optimizer
optimizer = LlamaDatasetOptimizer(config_path="my_custom_config")

# Run optimization
optimized_dataset = optimizer.optimize(
    dataset_path="data/training_data.jsonl",
    output_dir="results/optimized",
    model="meta-llama/Llama-3.2-1B-Instruct",
    top_k=10000,
    skip_deduplication=False,
    validate=True
)
```

### Colab Integration

```python
from colab_wrapper import optimize_dataset_colab

# Optimize dataset in Google Colab
optimized_dataset = optimize_dataset_colab(
    dataset_path="sample_data.jsonl",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    config_name="test_config",
    top_k=100
)
```

## 📊 Output Files

The optimizer generates several output files:

| File | Description |
|------|-------------|
| `optimized_dataset.jsonl` | The final optimized dataset |
| `optimization_report.yaml` | Detailed optimization statistics |
| `optimizer_log_[timestamp].log` | Complete processing logs |

### Sample Report
```yaml
timestamp: "2025-01-15T10:30:00"
config_used: "llama_3_2_instruct"
original_size: 10000
size_after_quality_filter: 8500
size_after_deduplication: 7200
optimized_size: 5000
reduction_percentage: "50.00%"
```

## 🎯 Quality Filters Explained

### Token-Based Filtering
- **min_tokens_per_turn**: Minimum tokens in any single message
- **max_tokens_per_turn**: Maximum tokens in any single message
- **min_total_tokens**: Minimum total tokens in conversation
- **max_total_tokens**: Maximum total tokens in conversation

### Conversation Structure
- **min_turns**: Minimum number of message exchanges
- **max_turns**: Maximum number of message exchanges
- **must_have_roles**: Required roles in conversation
- **balanced_conversation_ratio**: Max allowed ratio difference between user/assistant

### Refusal Detection
Automatically removes samples containing AI refusal patterns:
- "I'm sorry, but I cannot..."
- "As an AI, I'm unable to..."
- "I can't help with that..."

## 🔄 Deduplication Methods

### Sentence Transformer (Default)
- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Fast and accurate semantic similarity
- Good for most use cases

### LLaMA Model Embeddings
- Uses the actual LLaMA model for embeddings
- More accurate but slower
- Better for domain-specific deduplication

## 🚨 Troubleshooting

### Common Issues

**Model Loading Timeout**
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
python llama_dataset_optimizer.py ...
```

**Out of Memory Errors**
```bash
# Reduce batch sizes in config
batch_sizes:
  quality_filtering: 256
  embedding: 128
  scoring: 32
```

**Flash Attention Issues**
```bash
# Disable flash attention
# Flash attention is disabled by default
```

**No Samples Pass Quality Filter**
```bash
# Use more lenient test config
python llama_dataset_optimizer.py --config test_config ...
```

### Hardware-Specific Notes

**Apple Silicon (M1/M2/M3)**
- MPS acceleration is automatically detected
- Use `torch_dtype=torch.float16` for best performance
- Flash attention is not supported

**CUDA GPUs**
- Automatic GPU detection and utilization
- 4-bit quantization available with bitsandbytes
- Flash attention supported on A100+

**CPU Only**
- Automatically falls back to CPU processing
- Uses `torch_dtype=torch.float32`
- Significantly slower but functional

## 📈 Performance Tips

### For Large Datasets (>100K samples)
1. Use GPU acceleration when available
2. Enable 4-bit quantization: `use_4bit=True`
3. Increase batch sizes for your hardware
4. Consider using sentence transformers for deduplication

### For Small Datasets (<10K samples)
1. Use the `test_config` configuration
2. Skip deduplication with `--skip-deduplication`
3. Use smaller models like TinyLlama for faster processing

### Memory Optimization
1. Reduce batch sizes in configuration
2. Use 4-bit quantization
3. Process in smaller chunks
4. Clear GPU cache between runs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📜 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of HuggingFace Transformers
- Uses FAISS for efficient similarity search
- Inspired by research on dataset quality and model performance