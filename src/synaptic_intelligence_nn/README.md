### Continual Learning Through Synaptic Intelligence - Synaptic Intelligence (SI)

Paper: <https://arxiv.org/pdf/1703.04200.pdf>

In the "Continual Learning Through Synaptic Intelligence" paper, the authors propose an algorithm called Synaptic Intelligence (SI) that computes an importance measure for each synaptic weight in the network as learning proceeds. This importance measure is then used to protect the synaptic weight from drastic changes when the network learns new tasks, effectively mitigating catastrophic forgetting.

The algorithm can be broken down into several key steps:

1. **Calculating Importance Measure (`Ω`) for Each Weight**: The algorithm keeps track of a parameter `Ω` for each weight, which measures its importance. This is computed during the learning of each task.

    - The computation involves the rate of change of the loss with respect to the weight and the rate of change of the weight itself. Essentially, `Ω` accumulates the contributions of each weight to the loss over time.

    Mathematically, it is calculated as:

    Ωi=Ωi+Δθi⋅∂L∂θiΩi​=Ωi​+Δθi​⋅∂θi​∂L​

    Here, `Ω_i` is the importance measure for weight `i`, `Δθ_i` is the change in weight `i` during an update, and `\(\frac{\partial L}{\partial \theta_i}\)` is the gradient of the loss `L` with respect to the weight `i`.

2. **Updating Weights with Regularization Term**: During the learning of new tasks, a regularization term is added to the loss function. This term penalizes changes to weights that are deemed important for previous tasks.

    The regularized loss `L'` is:

    L′=L+12∑iΩi(θi−θi,prev)2L′=L+21​i∑​Ωi​(θi​−θi,prev​)2

    Here, `θ_i` is the current value of weight `i` and `θ_{i,prev}` is its value when the previous task was learned.

By adding this regularization term, SI makes it more costly to update weights that are important for previous tasks, allowing the network to preserve old knowledge while still being able to learn new tasks.

This concept shares similarities with your "confidence" mechanism, which also aims to determine the importance of each weight and modulate its update accordingly. Integrating a similar regularization term might be an interesting avenue for you to explore.

## TODO

Update the description with our findings.

### First results

##### Custom model

- Task A (1st time): 0.08147133886814117
- Task B: 0.2572690546512604
- Task A (2nd time): 0.08146069943904877

##### Standard model

- Task A (1st time): 0.36357301473617554
- Task B: 0.6636621356010437
- Task A (2nd time): 0.503075361251831

Looks like SI is getting better results than a classic NN on the small dataset that we are using.
