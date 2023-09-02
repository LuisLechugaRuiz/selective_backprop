# Selective BackPropagation

### Objective

We aim to extend the concept of backward propagation in neural networks by incorporating a "confidence" measure and a counter for each weight. The idea is to train a more "general" network that can hold onto "invariant representations" learned during different tasks without easily forgetting them when new tasks are trained.

### Why It's Useful

1. **Selective Updating**: This approach can prioritize the modification of neurons that have lower confidence, making them more "plastic" or adaptable.
2. **Resource Efficiency**: By adjusting the learning rate based on confidence, the model can potentially train faster and require fewer computational resources.
3. **Task Generalization**: By preserving high-confidence neurons, the network can better retain useful features across different tasks, potentially making it more robust and adaptable.

### How It Works

1. **Confidence Mechanism**: For each weight, we maintain a confidence variable that's updated via a running average or Bayesian scheme. Confidence is influenced by the magnitude of the gradients for that weight.
    - **Alpha**: Rate at which old and new confidence are blended.
    - **Gamma**: Decay term influenced by the counter.

2. **Counter Mechanism**: Each weight also has a counter that is incremented based on the number of correct predictions made during training. The counter serves as a form of memory that makes high-confidence neurons more resistant to changes.
    - Counter is normalized by the total number of training samples (or epochs) to keep it balanced.

3. **Gradient Adjustment**: During backpropagation, the gradients are modified based on the confidence and counter for each weight. This ensures that weights with high confidence and counters are updated less compared to those with low confidence.

### Implementation

We implemented a `CustomLinearLayer` class in PyTorch that integrates these mechanisms. The layer's `forward()` method performs the standard linear operation, while a separate `update_confidence_and_counter()` method updates the confidence and counter based on gradients and performance statistics.

The main neural network (`CustomNet`) uses these custom layers and adjusts its weights by taking into account both the gradients and the confidence measures.

### Limitations

This is the first iteration of all the experiments to create a selective back propagation.

The main limitation ist that the update is done after the gradients calculations, so the confidence is not taken into account. When we perform gradient descent we are still updating all the weights without taking into account the confidence. Studying the existing literature on the topic I found a promising paper that proposes an update on the Loss function, giving a penalty in case that the important weights need to be updated.
