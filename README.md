# 🏆 Charting Student Math Misunderstandings

<p align="left">
  <b>English</b> | <a href="README_ZH.md">简体中文</a>
</p>

> **Gemma-2-9B Sequence Classification via LoRA Fine-Tuning**  
> High-performance implementation for identifying student mathematical misconceptions with a state-of-the-art evaluation result of **MAP@3: 0.93** on hold-out tests.

---

## 📌 Project Background & Educational Value

In mathematics education, identifying *why* a student makes an error is far more valuable than simply checking if their answer is correct. Students' mistakes are rarely random slips; instead, they typically stem from systematic, deeply rooted **cognitive misconceptions**.

Traditionally, diagnosing these misconceptions requires experienced teachers to manually analyze open-ended student justifications. This process is highly labor-intensive, time-consuming, and difficult to scale for large classrooms or online learning platforms.

**This project automates the diagnostic pipeline.** By analyzing a unified student response context—comprising the multi-choice question description, the chosen option, the correctness status, and the student's free-form natural language justification—our system maps student reasoning flaws to one of **65 distinct mathematical misconceptions** with expert-level precision. This serves as a vital plug-and-play cognitive diagnostic engine for Intelligent Tutoring Systems (ITS) and Adaptive Learning platforms.

---

## 💻 Model & Technology Stack

To achieve deep reasoning performance while maintaining training efficiency on consumer-grade and enterprise GPUs, the project leverages a highly optimized open-source ecosystem:

* **Foundation Model**: **Gemma-2-9B-IT** (Google). Built on Google's latest lightweight, high-performance architecture, the 9B-Instruct model shows exceptional semantic understanding, logical reasoning, and context-parsing capability. It is uniquely qualified to parse multi-step mathematical arguments and student justifications.
* **Fine-Tuning Paradigm**: **PEFT / LoRA (Low-Rank Adaptation)**. By freezing the massive base model parameters and updating low-rank adapter matrices, we restrict trainable parameters to **< 1%**. This dramatically lowers the VRAM footprint while mitigating the risk of *catastrophic forgetting* of general reasoning capabilities.
* **Full Technology Stack**:
  * **Deep Learning Framework**: `Python 3.10+`, `PyTorch 2.0+` (Ampere-native bfloat16 mixed-precision optimization)
  * **Model Interface & Adapters**: Hugging Face `transformers` (Gemma-2 structural interface), `peft` (LoRA adapter engine), `datasets` (dynamic batch streaming)
  * **Scientific Computing & ML**: `pandas`, `numpy`, `scikit-learn` (LabelEncoder & robust multi-class dataset division)
  * **Training Telemetry & Plotting**: `matplotlib` (for exponential moving average loss and accuracy visualization)

---

## 🔄 Data and Model Pipeline Flow

```mermaid
graph TD
    A[Student Response & Explanation] --> B[Unified Text Template]
    B --> C[Gemma-2-9B Frozen Base]
    C --> D[LoRA Adapter]
    D --> E[Custom Class Head score layer]
    E --> F[Top-3 Ranked Probabilities]
    F --> G[MAP@3 Metrics Evaluation]
```

---

## 🛠️ Key Technical Challenges & Solutions

During development, we resolved several critical bugs and structural bottlenecks:

### 1. Custom PyTorch PEFT Integration (Bypassing LLaMA-Factory)
Standard fine-tuning frameworks like *LLaMA-Factory* load models natively via `AutoModelForCausalLM` and lack built-in support for sequence classification. 
* **Our Solution**: Built a custom script inheriting `AutoModelForSequenceClassification` coupled with `peft` to initialize a 65-class output linear layer.

### 2. Classification Head Persistence Trick
Under PEFT/LoRA, the base model is frozen, and only LoRA adapters are updated. By default, newly added classification linear heads (the `score` layer) are discarded during adapter checkpoints.
* **Our Solution**: Explicitly configured `LoraConfig` with `modules_to_save=["score"]`. This ensures the linear head's weights are saved alongside adapter weights in the checkpoint directories.

### 3. Upstream Compatibility (Hugging Face `datasets` VideoReader Fix)
Hugging Face `datasets.set_format(type='torch')` causes an upstream `torchvision` import crash on Linux and Windows servers due to the removal of `VideoReader` from newer torchvision versions.
* **Our Solution**: Removed `set_format` entirely. Batch formatting is dynamic and handled directly by Hugging Face `DataCollatorWithPadding` in the PyTorch DataLoader, achieving peak stability.

### 4. VRAM & Speed Optimizations (32GB+ Setup)
* **Native BF16 Training**: Ampere-native bfloat16 mixed-precision is configured to guarantee gradient stability.
* **SDPA (Scaled Dot Product Attention)**: Native PyTorch flash-attention backend yields faster speeds and low GPU memory overhead.
* **Gradient Checkpointing**: Enabled to bypass OOM, allowing sequence limits to scale to `MAX_LEN = 512`.
* **Throughput Tuning**: Set `BATCH_SIZE = 4` with `GRAD_ACCUM = 4` (effective batch size of 16). Evaluation is performed every `500` steps to avoid performance lag from heavy disk writes.

---

## 📁 File Structure

```text
├── data/
│   ├── my_train.csv             # Cleaned training dataset (~33,027 samples)
│   ├── my_test.csv              # Test dataset
│   └── my_test_labels.csv       # Test dataset ground truth labels
├── saves/
│   └── Gemma-2-9B/
│       └── lora_seqcls/         # LoRA Checkpoints & Adapter weights
├── output/
│   ├── output.csv               # Top-3 decoded prediction classes (ready for Kaggle)
│   └── evaluation_results.png   # Matplotlib Accuracy bar plot on hold-out set
├── scripts/
│   ├── plot_loss.py             # Exponential Moving Average (EMA) loss plotter
│   └── ...                      # Other helper scripts (data splitting, conversion, etc.)
├── train_seqcls.py              # Sequence classification LoRA training pipeline
├── gemma2_inference.py          # Fast batched inference & MAP@3 evaluation on test set
├── requirements.txt             # Precise python environment requirements
└── README.md                    # Project documentation (this file)
```

---

## 📊 Evaluation & Metric Telemetry

* **Optimal Model**: `checkpoint-5500`
* **Primary Metric**: **MAP@3 (Mean Average Precision @ 3) = 0.93**
* **Top-N Accuracies**:
  * **Top-1 Accuracy**: ~83.24%
  * **Top-3 Accuracy**: ~95.80%

### Convergence Visualization
You can generate a comprehensive loss and learning rate diagram using the built-in `plot_loss.py` script:

![Loss and LR Curves](output/training_loss_curves.png)

> [!NOTE]
> The minimum validation loss is reached at **Step 3600** ($Loss = 0.3683$). However, the optimal classification decision boundary matured at **Step 5500** ($MAP@3 = 0.93$), demonstrating that SFT loss and evaluation metrics can have a slight alignment offset near convergence.

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Installation
Install core dependencies using the generated `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: If the `bitsandbytes` backend is too old (< 0.46.1) or missing, `train_seqcls.py` has an embedded runtime auto-updater that will safely upgrade it to the correct version.*

### 2. High-Performance Training
To initiate the Gemma-2-9B Sequence Classification training pipeline:
```bash
python train_seqcls.py
```
*Paths fallback automatically between your local machine (`models/`) and autodl servers (`/root/autodl-tmp/`).*

### 3. Batched Inference & Metric Plotting
Load the best LoRA adapter checkpoint, run test predictions, calculate MAP@3, and export results:
```bash
python gemma2_inference.py
```
This script yields two main files:
* `output/output.csv` (Predictions)
* `output/evaluation_results.png` (Performance visualization)

### 4. Telemetry Plotting
Plot the EMA training loss, validation loss, and cosine learning rate schedule curves:
```bash
python scripts/plot_loss.py
```
Saves the visualization output directly to `training_loss_curves.png`.

---

## 🏆 License & Acknowledgments
Optimized for the *Kaggle: Charting Student Math Misunderstandings* competition. Powered by Hugging Face PEFT and PyTorch.
