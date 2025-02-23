Observations:

1) **Variation-1**-

Model Architecture:

5 Convolutional Blocks
Batch Normalization after each convolutional layer
SeparableConv2D in deeper layers for computational efficiency
Dropout Regularization (0.3)
Adam Optimizer with Learning Rate Decay

Results & Observations:

1. Training vs Validation Accuracy
The training accuracy showed a steady increase, reaching ~95%.
The validation accuracy fluctuates initially but stabilizes around ~92-94%.
The fluctuation in validation accuracy suggests variance in generalization across epochs.
2. Possible Issues
Batch Normalization helped stabilize learning, but initial fluctuations suggest that the model was adjusting to normalization.
SeparableConv2D improved computational efficiency without sacrificing accuracy.
Overfitting is still present as the training accuracy remains consistently higher than validation accuracy.
Dropout (0.3) might be insufficient in deeper layers to prevent overfitting.
Validation accuracy fluctuation suggests the need for learning rate adjustments or smoother regularization.

2) **Variation-2**-

Model Architecture & Changes:

The following modifications were made to improve performance:
L2 Regularization (l2(0.001)) applied to Conv and Dense layers to reduce overfitting.
Batch Normalization added after each Conv layer to stabilize activations.
Increased Dropout (0.5 in Fully Connected layer) for better regularization.
Learning Rate Scheduling using ExponentialDecay for adaptive learning.

Accuracy & Overfitting Analysis:

Training Accuracy: ~88.1%, Validation Accuracy: ~87.1%.
Slight Overfitting → Training accuracy is slightly higher than validation accuracy, but the gap is small.
Validation accuracy is more stable compared to previous models, reducing fluctuations.
Loss values (~0.8542 for training and ~0.8885 for validation) indicate potential room for further optimization.

Issues Identified:

Slight Overfitting – Training accuracy is higher than validation accuracy.
Validation Accuracy Can Be Improved – Needs better generalization.
High Loss Values – Both training and validation losses are relatively high.
Potential Room for Optimizer Improvements – Adam might not be the best choice.