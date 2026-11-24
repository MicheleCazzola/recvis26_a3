## Object recognition and computer vision 2026/2027
### Assignment 3: Synthetic CIFAR-10 image classification

#### Requirements

1. Install PyTorch from [pytorch.org](https://pytorch.org/).
2. Run the following command to install the additional dependencies:

   ```bash
   pip install -r requirements.txt
   ```

#### Dataset

We will be using the ten-class CIFAR-10 dataset. The training and validation images come from standard CIFAR-10, while the test images come from a synthetic, distribution-shifted target domain. Download `train.npz`, `validation.npz`, `test.npz`, and `sample_submission.csv` from the private Kaggle competition. The test image labels are not provided.

#### Rules

Open-source models, pretrained checkpoints, and external data are allowed if they are fully documented in the report. Proprietary or closed models and API-only services are prohibited. Do not obtain, reconstruct, manually annotate, or otherwise access the hidden test labels.

#### Approaches

Two approaches are provided:
- **Vision**: Fine-tune a pretrained vision model on CIFAR-10 (ViT in this case). For linear probe, the features are cached for efficiency.
- **CLIP**: Use the pretrained CLIP model to classify images with text prompts. The CLIP model is frozen, and only the linear head is trained on CIFAR-10. ***Variant***: use WiSE-FT to fine-tune the CLIP model on CIFAR-10 and ensemble the results with a mixing coefficient $\alpha$ to control the importance of the fine-tuned model over the original CLIP model.

#### Repository structure

```
<root>/
├── data/                     <- CIFAR-10 dataset (train, validation, test)
├── configs/
    ├── config.yaml           <- default configuration file (initialized with the best configuration here)
    └── <config_name>.yaml    <- configuration files for specific experiments
├── src/                      <- source code (models, datasets, training, evaluation, utilities)             
├── results/                  <- default folder for results (checkpoints, logs, predictions, etc.)  
├── main.py                   <- entry point
├── README.md                 <- this file
├── requirements.txt          <- dependencies
```

#### Usage

1. Run the following command to train a model with the default configuration:

   ```bash
   python main.py --config configs/config.yaml
   ```

2. To run a specific experiment, use the corresponding configuration file:

   ```bash
   python main.py --config configs/<config_name>.yaml
   ```

3. To describe the configuration options, run:

   ```bash
   python main.py --help
   ```

4. Further options can be specified by editing the configuration file [config.yaml](configs/config.yaml)

#### Acknowledgments

The source-domain data is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

This starter is adapted from the [2024 RecVis Assignment 3 repository](https://github.com/willowsierra/recvis24_a3).
