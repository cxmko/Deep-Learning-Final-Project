# Optimal Budget Rejection Sampling (OBRS) with GANs on MNIST

This project applies the **Optimal Budget Rejection Sampling (OBRS)** method to **Generative Adversarial Networks (GANs)** for generating high-quality and diverse images from the **MNIST dataset**. OBRS is integrated into the GAN training and generation process to enhance **precision**, **recall**, and overall model performance.

---

##  Repository Structure

```
.
├── checkpoints/                  # Saved model weights and checkpoints
├── generate.py                   # Script to generate 10,000 samples using OBRS
├── train.py                      # Script to train the GAN with OBRS
├── model.py                      # Contains the GAN architecture
├── utils.py                      # Utility functions for training and generation
├── requirements.txt              # Python dependencies for the project
├── Cameron_Mouangue_Rapport_OBRS.pdf  # Full project report 
├── Presentation_Cameron.pdf      # Project presentation slides
└── README.md                     # Project documentation
```

---

##  **Project Overview**

This project focuses on integrating **OBRS** into GANs for improving sample quality and diversity:

- **Precision**: Measures the proportion of generated samples that align with the true data distribution.
- **Recall**: Measures how well the generator captures the full diversity of the data.

### Key Implementations:
1. **Vanilla GAN** (Baseline)
2. **OBRS applied to generation only** 
3. **OBRS applied to training only**
4. **Full OBRS integration (training + generation)**

---

##  **Results Summary**

| **Model Version**         | **FID** | **Precision** | **Recall** |
|---------------------------|---------|---------------|------------|
| **Vanilla GAN (Baseline)** | 29      | 0.55          | 0.23       |
| **OBRS (Gen Only, K=2)**  | 45      | 0.69          | 0.51       |
| **OBRS (Train Only, K=5)**| 16      | 0.56          | 0.27       |
| **Full OBRS (K=2)**       | 31      | 0.79          | 0.51       |
| **Full OBRS (K=10, 300 Epochs)** | **13**  | **0.88**    | **0.66**   |

- **OBRS improves both precision and recall** significantly when integrated into the training and generation phases.
- Higher **\( K \)** values enhance precision but introduce computational overhead during generation.

---

##  **How to Use**

### 1. Clone the Repository

```bash
git clone https://github.com/cxmko/Deep-Learning-Final-Project.git
cd /Deep-Learning-Final-Project
```

### 2. Install Dependencies

Ensure you have Python 3.x installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

### 3. Train the Model

To train the GAN with OBRS, run:

```bash
python train.py
```

**Arguments in `train.py`:**
- `--epochs`: Number of training epochs (default: 100)
- `--rejection_budget`: OBRS rejection budget \( K \) (default: 10)


Example for full OBRS with \( K=5 \):

```bash
python train.py --epochs 200 --rejection_budget 5 
```

---

### 4. Generate Samples

To generate 10,000 samples using the trained model and OBRS:

```bash
python generate.py --num_samples 10000
```

**Arguments in `generate.py`:**
- `--K`: OBRS rejection budget \( K \) (default: 10)


---


##  **Performance Insights**

1. **Effect of OBRS on Training Stability**:
   - Incorporating OBRS in the training phase **improves convergence stability** and reduces local minima.
   - Training requires adjustments like **gradient clipping** and **learning rate tuning** to handle numerical instabilities.

2. **Impact of \( K \)**:
   - Higher values of \( K \) increase precision but also **introduce computational overhead** during generation.
   - Testing with \( K=5 \) and \( K=10 \) shows diminishing returns on performance beyond certain thresholds.

3. **Visual Results**:
   - OBRS significantly improves the **visual quality** of generated MNIST images, reducing artifacts and increasing diversity.
   - Higher \( K \) values produce more refined samples but at a higher computational cost.

---

##  **Report and Presentation**

- **[Written Report](./Cameron_Mouangue_Rapport_OBRS.pdf)**: Detailed report explaining the theoretical foundation of OBRS, its implementation, and experimental results.
- **[Presentation](./Presentation_Cameron.pdf)**: Slide presentation summarizing key findings and methodology.

---

##  **Future Work**

- **Test on Complex Datasets**: Extend OBRS to more complex datasets like **CIFAR-10** and **ImageNet** to evaluate its generalizability.
- **Adaptive Rejection Sampling**: Explore dynamic adjustment of the rejection budget \( K \) during training for better performance balance.
- **Optimization**: Further optimize OBRS to reduce computational load, particularly in the **generation phase**.

---

##  **References**

1. Verine, A., Pydi, M. S., Negrevergne, B., Chevaleyre, Y. (2024). *Optimal Budget Rejection Sampling for Generative Models*.
2. Nowozin, S., Cseke, B., Tomioka, R. (2016). *f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*. NeurIPS.
3. Goodfellow, I., et al. (2014). *Generative Adversarial Networks*. NeurIPS.

---

##  **Acknowledgments**

This project was conducted as part of my **Deep Learning class** at **Université Paris-Dauphine**, under the supervision of **Alex Verine**.

---

Feel free to fork, modify, and explore the potential of OBRS in your own projects!
