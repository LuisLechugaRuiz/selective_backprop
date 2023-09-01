## Mathematical Framework

#### Initialization

1. **Initialize Confidence CiCi​, Frequency FiFi​, and Moving Average MAiMAi​ for each Weight**
    - Ci=0.5Ci​=0.5
    - Fi=0Fi​=0
    - MAi=0MAi​=0

#### Post-Epoch Update

1. **Update Frequency for each weight**
    - Fi,new=Fi+1Fi,new​=Fi​+1

2. **Normalize Frequency (you can also use a decay rate δδ to control this)**
    - NormalizedFrequencyi=FiFi+δNormalizedFrequencyi​=Fi​+δFi​​

3. **Update Moving Average of Gradient Directions**
    - MAi,new=α×MAi+(1−α)×sign(∂L∂wi)MAi,new​=α×MAi​+(1−α)×sign(∂wi​∂L​)

4. **Calculate Stability**
    - Stabilityi=∣MAi,new∣Stabilityi​=∣MAi,new​∣

5. **Update Confidence**
    - Ci,new=α×Ci+(1−α)×(Stabilityi×∣∂L∂wi∣×(1−γ×NormalizedFrequencyi))Ci,new​=α×Ci​+(1−α)×(Stabilityi​×∣∂wi​∂L​∣×(1−γ×NormalizedFrequencyi​))
    - Ci=sigmoid(Ci,new)Ci​=sigmoid(Ci,new​)
    - Clamp(Ci,0,1)Clamp(Ci​,0,1) _(Optional)_

#### Backward Propagation

1. **Modify Gradient using Confidence and Frequency**
    - f(Ci,NormalizedFrequencyi)=(1−Ci)×(1−β×NormalizedFrequencyi)f(Ci​,NormalizedFrequencyi​)=(1−Ci​)×(1−β×NormalizedFrequencyi​)
    - Gradientmodified,i=∂L∂wi×f(Ci,NormalizedFrequencyi)Gradientmodified,i​=∂wi​∂L​×f(Ci​,NormalizedFrequencyi​)

2. **Update Weights**
    - wi,new=wi−learning rate×Gradientmodified,iwi,new​=wi​−learning rate×Gradientmodified,i​

#### Threshold Condition _(Optional)_

- **Only update the weight if Confidence is below a certain threshold:**
  - If Ci<ThresholdCi​<Threshold, then update wiwi
