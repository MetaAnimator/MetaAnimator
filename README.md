# Meta-Animator

**Meta-Animator** is a modular and efficient human image animation framework based on diffusion models. It supports few-shot fine-tuning with lightweight LoRA modules and enables high-quality, identity-preserving video generation from reference images.

## 🔧 Environment Requirements

- Python >= 3.10  
- PyTorch >= 2.3.1  
- (Optional) CUDA-compatible GPU for accelerated training and inference

You can set up the environment using `conda` or `pip`. For example:

```bash
conda create -n meta_animator python=3.10
conda activate meta_animator
pip install torch==2.3.1 torchvision torchaudio
pip install -r requirements.txt
# plus any other dependencies...
````

## 🗂 Project Structure

* `Step0_train_BaseCtrl_LoRA.sh` — Stage 1: Train base ControlNet with condition-specific LoRA.
* `Step1_train_Motion_Module.sh` — Stage 2: Train the motion module.
* `Step2_finetune_BaseCtrl_LoRA.sh` — Stage 3: Fine-tune the model on a reference image set.
* `extract_LoRA_weights.sh` — Export trained LoRA weights for modular reuse or deployment.
* `Step3_sample.sh` — Run inference using the trained model to generate animation.

## 🚀 Training & Inference Pipeline

1. **Train Base ControlNet with LoRA:**

   ```bash
   bash Step0_train_BaseCtrl_LoRA.sh
   ```

2. **Train the Motion Module:**

   ```bash
   bash Step1_train_Motion_Module.sh
   ```

3. **Fine-tune on a Reference Set (Few-shot):**

   ```bash
   bash Step2_finetune_BaseCtrl_LoRA.sh
   ```

4. **(Optional) Extract LoRA Weights Separately:**

   ```bash
   bash extract_LoRA_weights.sh
   ```

5. **Inference (Generate Animation):**

   ```bash
   bash Step3_sample.sh
   ```

## 📌 Notes

* Ensure that all dataset paths and configuration files are correctly set before running the scripts.
* For best performance, fine-tune with at least 5 high-quality reference images.
* The output animations will be saved in the `outputs/` directory by default.
