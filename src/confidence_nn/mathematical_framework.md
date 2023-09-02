## Mathematical Framework

#### Initialization

1. **Initialize Confidence CiCi​, Frequency FiFi​, and Moving Average MAiMAi​ for each Weight**
    - Ci=0.5
    - Fi=0
    - MAi=0

#### Post-Epoch Update

1. **Update Frequency for each weight**
    - Fi,new=Fi+1

2. **Normalize Frequency (you can also use a decay rate δδ to control this)**
    - NormalizedFrequencyi=FiFi+δ

3. **Update Moving Average of Gradient Directions**
    - MAi,new=α×MAi+(1−α)×sign(∂L∂wi)

4. **Calculate Stability**
    - Stabilityi=∣MAi,new∣

5. **Update Confidence**
    - Ci,new=α×Ci+(1−α)×(Stabilityi×∣∂L∂wi∣×(1−γ×NormalizedFrequencyi))
    - Ci=sigmoid(Ci,new)
    - Clamp(Ci,0,1)

#### Backward Propagation

1. **Modify Gradient using Confidence and Frequency**
    - f(Ci,NormalizedFrequencyi)=(1−Ci)×(1−β×NormalizedFrequencyi)
    - Gradientmodified,i=∂L∂wi×f(Ci,NormalizedFrequencyi)

2. **Update Weights**
    - wi,new=wi−learning rate×Gradientmodified,i

#### Threshold Condition _(Optional)_

- **Only update the weight if Confidence is below a certain threshold:**
  - If Ci<ThresholdCi​<Threshold, then update wiwi
