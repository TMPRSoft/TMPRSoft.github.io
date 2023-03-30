---
layout: post
title:  "[리뷰] High-Resolution Image Synthesis with Latent Diffusion Models"
date:   2023-03-30 23:29:00 +0900
categories: [Diffusion Model]
tags: [Diffusion Model, Image Generation]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [High-Resolution Image Synthesis with Latent Diffusion Models]
<br/>
Authors: R. Rombach, T. Zhang, B. Zhang, H. Ouyang, D. Chen, Q. Chen, and F. Wen
<br/>
Submitted to [CVPR 2022]
<br/>
Project Page: [Github]

## Abstract

* 이미지 형성 프로세스를 디노이징 autoencoder의 순차적 응용으로 분해하여, 확산 모델(DM)은 이미지 데이터 및 그 외에서 SotA 합성 결과 성능을 달성함.
* 또한, 그것들의 공식화는 재트레이닝(retraining) 없이 이미지 생성 프로세스를 컨트롤하는 가이드 메커니즘을 허용함.
* 그러나, 이러한 모델은 일반적으로 픽셀 공간에서 직접 작동하기 때문에, 강력한 DM의 최적화는 종종 수백 일의 GPU 동작 시간을 소모하며 순차적 평가로 인해 추론 비용이 많이 듦.
* 품질과 유연성을 유지하면서 제한된 계산적 자원에 대한 DM 트레이닝을 가능하게 하기 위해, 위 모델들을 강력한 사전 트레이닝(pretrain)된 autoencoder의 잠재 공간(latent space)에 적용함.
* 이전 연구와 달리, 그러한 표현에 대한 확산 모델을 트레이닝하면 처음으로 복잡도 (complexity) 감소와 세부 보존 사이에서 거의 최적의 지점에 도달할 수 있어, 시각적 충실도가 크게 향상됨.
* 교차 어텐션 (cross-attention) 레이어를 모델 구조에 도입함으로써, 확산 모델을 텍스트나 bounding box와 같은 일반적인 조건 인풋을 위한 강력하고 유연한 생성기로 전환하고 고해상도 합성이 convolution 방식으로 가능해짐.
* 잠재 확산 모델(LDM)은 이미지 inpainting 및 클래스 조건부 이미지 합성에서 새로운 SotA 스코어를 달성하고 무조건부 이미지 생성, 텍스트-이미지 (text-to-image) 합성, SR(super-resolution)를 포함한 다양한 작업에서 매우 경쟁력 있는 성능을 달성하는 동시에, 픽셀 기반의 확산 모델에 비해 계산 요구 사항을 크게 감소시킴.

<div id="Fig1"></div>
![Fig1](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/fig1.png)

그림 1: 덜 공격적인 downsampling으로 달성 가능한 품질의 상한 증폭. 확산 모델은 공간적 데이터를 위한 우수한 유도 편향을 제공하므로, 잠재 공간에서 관련 생성 모델의 강한 공간적 downsampling이 필요하지는 않으나, 적절한 autoencoding 모델을 통해 데이터의 차원(dimensionality)을 크게 줄일 수 있음. 이미지들은 $$512^2$$ 픽셀에서 평가된 DIV2K 유효성 검사 데이터셋에서 가져옴. 공간적 downsampling factor를 $$f$$로 표기함. 재구성 FID(R-FID)와 PSNR은 ImageNet-val에서 계산됨. 표 8도 참고할 것.

<div id="Fig2"></div>
![Fig2](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/fig2.png)

그림 2: 지각적 및 의미론적 압축 설명: 디지털 이미지의 대부분의 비트들은 지각할 수 없는 세부 요소에 해당함. DM이 주원인인 loss 항을 최소화함으로써 의미론적으로 의미 없는 정보들을 압축하면서도, 경사 (트레이닝 중) 및 신경망 backbone(트레이닝 및 추론)은 여전히 모든 픽셀에 대해 측정되고, 이는 과도한 계산 및 필요 이상으로 비용이 드는 최적화 및 추론을 야기함. 여기서 잠재 확산 모델(LDM)을 효과적인 생성 모델로서 제안하고 또한 오직 지각할 수 없는 세부 요소만을 제거하는 분리된 가벼운 압축 단계를 제안함.

## Introduction

* **Contributions**
    * 순전히 transformer 기반 접근 방식과는 달리, 본 기법은 더 높은 차원의 데이터로 능숙하게 확장함.
        * 이전 연구보다 더 충실하고 상세한 재구성(reconstruction)을 제공하는 압축 레벨에서 동작할 수 있음([그림 1](#Fig1) 참고).
        * 메가픽셀 이미지의 고해상도 합성에 적용할 수 있음.
    * 계산 비용(computational cost)을 크게 낮추면서 다수의 task(무조건부 이미지 합성, inpainting, 확률적 SR(stochastic super-resolution)) 및 데이터셋에서 뛰어난 성능을 달성함.
    * 인코더/디코더 구조와 스코어 기반 prior를 동시에 학습하는 이전 연구와 달리, 본 접근 방식은 재구성 및 생성 능력에 대한 섬세한 가중치(weight)를 요구하지 않음을 입증함. 이는 매우 충실한 재구성을 보장하고 잠재 공간의 정규화(regularization)를 거의 요구하지 않음을 의미함.
    * SR, inpainting, semantic 합성과 같은 조밀하게 조건화된 task에 대해, 본 모델이 convolution 방식으로 적용될 수 있고 최대 $$1024^2$$ 픽셀까지의 크고 일관된 이미지를 나타낼 수 있음을 보임.
    * 또한, 교차 어텐션(cross-attention)을 기반으로 범용 조건화 메커니즘을 설계하여, 멀티모달(multi-modal) 트레이닝을 가능하게 함. 클래스 조건부, 텍스트-이미지(text-to-image), 레이아웃-이미지(layout-to-image) 모델을 트레이닝하는 데 사용함.

## Method

* 고해상도 이미지 합성을 위한 트레이닝 DM의 계산 요구량을 낮추기 위해, DM이 해당 손실 항을 undersampling하여 지각적으로 관련 없는 세부 사항을 무시할 수 있지만, 픽셀 공간에서 여전히 높은 비용의 기능 검증이 필요하다는 것을 관찰하였으며, 이는 계산 시간과 에너지 자원에 엄청난 수요를 야기함.
* 우리는 생성 학습 단계에서 압축을 명시적으로 분리함으로써 이러한 단점을 피할 것을 제안함([그림 2](#Fig2) 참고). 이를 달성하기 위해 이미지 공간과 지각적으로 동일한 공간을 학습하지만, 계산 복잡성이 크게 감소하는 autoencoding 모델을 사용함.
* 이러한 접근 방식은 다음과 같은 몇 가지 이점을 제공함.
    * 고차원 이미지 공간을 떠남으로써, 샘플링이 저차원 공간에서 수행되기 때문에 계산적으로 훨씬 효율적인 DM을 얻을 수 있음.
    * UNet 구조로부터 상속된 DM의 유도 바이어스를 활용하는데, 이는 공간 구조를 가진 데이터에 특히 효과적이므로 이전 접근법에서 요구하는 공격적이고 품질 향상적인 압축 수준의 필요성을 완화함.
    * 마지막으로, 잠재 공간이 여러 생성 모델을 트레이닝하는 데 사용될 수 있고 단일 이미지 CLIP-guided 합성과 같은 다른 다운스트림 애플리케이션에도 활용될 수 있는 범용 압축 모델을 얻음.

### Perceptual Image Compression

* 본 지각적 압축 모델은 이전 연구[[23]]에 기반을 두고 지각적 loss 및 패치 기반 적대적 목적의 조합으로 트레이닝된 autoencoder로 구성됨. 이는 재구성(reconstruction)이 로컬 사실주의를 강제하는 이미지 manifold에 국한됨을 보장하고, $$L_2$$나 $$L_1$$ 목적 같은 픽셀-공간 loss에 단독으로 의존함으로써 도입된 흐림 현상을 회피함.
* 더 구체적으로, RGB 공간에서 한 이미지 $$x \in \mathbb{R}^{H \times W \times 3}$$가 주어졌을 때, 인코더 $$\mathcal{E}$$는 $$x$$를 잠재 표현 $$z = \mathcal{E}(x)$$로 인코딩하고, 디코더 $$\mathcal{D}$$는 그 잠재로부터 이미지를 재구성하여, $$\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x))$$를 얻으며, 여기서 $$z \in \mathbb{R}^{h \times w \times c}$$임. 중요한 것은, 인코더는 이미지를 factor $$f = H/h = W/w$$로 downsampling 한다는 것이고, 여러 가지 downsampling factor $$f = 2^m$$, $$m \in \mathbb{N}$$을 조사하고자 함.
* 임의의 고분산의 잠재 공간을 회피하기 위해, 두 가지 다른 종류의 정규화로 실험함.
    * 첫째 변형, KL-reg.는 VAE와 유사하게 학습된 잠재 상의 표준 normal 쪽으로 약간의 KL 페널티를 지우는 반면, VQ-reg.는 디코더 내 벡터 quantization 레이어를 사용함. 이 모델은 VQGAN으로 해석될 수 있지만 quantization 레이어는 디코더에 흡수됨.
* 차후 DM은 학습된 잠재 공간 $$z = \mathcal{E}(x)$$의 2차원 구조에서 동작하도록 설계되어 있으므로, 상대적으로 약한 압축률을 사용하고 매우 좋은 재구성 결과를 달성할 수 있음. 이는 이전 연구들[[23], [66]]이 분포를 자동 회귀적으로 모델링하기 위해 학습된 공간 $$z$$의 임의의 1차원 ordering에 의존했고 그리하여 $$z$$의 내재 구조의 대부분을 무시한 것과는 대조적임.
* 그래서, 본 압축 모델은 $$x$$의 세부 요소를 더 잘 보존함([표 8](#Table8) 참고).
* 전체 목적 및 트레이닝 상세 내용은 부록에서 찾을 수 있음.

### Latent Diffusion Models

* **DM(Diffusion Model)**은 데이터 분포 $$p(x)$$를 정규분포 변수를 점진적으로 디노이징하여 학습하기 위해 설계된 확률적 모델인데, 이는 길이 $$T$$의 고정된 Markov Chain의 역방향 프로세스를 학습하는 것에 해당함.
    * 이미지 합성에 대해, 가장 성공적인 모델들은 $$p(x)$$에 대한 변형 하한의 재가중 (reweighted) 변형에 의존하는데, 이는 디노이징 스코어 매칭을 반영함.
    * 이 모델들은 디노이징 autoencoder의 동등하게 가중치가 부여된 시퀀스 $$\epsilon_\theta (x_t, t)$$, $$t = 1, \cdots, T$$로 해석 가능하며, 인풋 $$x_t$$의 디노이징된 변형을 예측하도록 트레이닝되고, 여기서 $$x_t$$는 인풋 $$x$$의 노이즈 있는 버전임.
    * 대응하는 목적은 다음과 같이 간략화될 수 있음.
    <br/><br/>
    <div id="Eq1"></div>
    ![Eq1](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/eq1.png)
    <br/><br/>
    여기서 $$t$$는 $$\{1, \cdots, T\}$$로부터 균일하게 샘플링됨.
* **잠재 표현의 생성 모델링** 우리의 트레이닝된 지각적 압축 모델은 $$\mathcal{E}$$와 $$\mathcal{D}$$로 구성되어, 이제 고주파 및 지각이 불가능한 세부 요소는 무시된 효율적이고 저차원의 잠재 공간에 접근이 가능함.
    * 고차원의 픽셀 공간에 비해, 이 공간은 likelihood 기반의 생성 모델에 더 적합한데, 이는 모델이 이제 (i) 데이터에서 중요하고 의미론적인 비트에 집중하고 (ii) 더 낮은 차원의 계산적으로 훨씬 더 효율적인 공간에서 트레이닝될 수 있기 때문임.
    * 높은 압축률의 discrete 잠재 공간에서 자동회귀적이고, 어텐션 기반의 transformer 모델에 의존하는 기존 연구와는 달리, 본 모델이 제공하는 이미지 특정의 유도 바이어스를 이용할 수 있음. 여기에는 주로 2차원 convolutional 레이어에서 기본 UNet을 구출할 수 있는 기능이 포함되며, 재가중 경계를 사용하여 지각적으로 가장 관련성이 높은 비트에 목적을 집중함.
    <br/><br/>
    <div id="Eq2"></div>
    ![Eq2](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/eq2.png)
    <br/><br/>
    * 본 모델의 신경 backbone $$\epsilon_\theta (\cdot, t)$$은 시간 조건부 UNet으로 실현됨.
    * 정방향 프로세스가 고정되어 있으므로, $$z_t$$는 트레이닝 동안 $$\mathcal{E}$$로부터 효율적으로 얻을 수 있고, $$p(z)$$로부터의 샘플들은 $$\mathcal{D}$$를 통하는 단일 통과로 이미지 공간으로 디코딩 될 수 있음.

<div id="Fig3"></div>
![Fig3](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/fig3.png)

그림 3: LDM을 결합이나 더 일반적인 교차 어텐션 메커니즘을 통해 조건화함. [3.3절](#Sec3.3) 참고.

<div id="Sec3.3"></div>
### Conditioning Mechanisms

* 생성 모델의 다른 타입들과 유사하게, DM은 원리상 형태 $$p(z \vert y)$$의 조건부 분포 모델링이 가능함. 이는 조건부 디노이징 autoencoder $$\epsilon_\theta (z_t, t, y)$$로 구현될 수 있고 텍스트, 의미론 맵, 다른 이미지 간 변환 (image-to-image translation) task과 같이 인풋 $$y$$를 통한 합성 프로세스를 제어하는 길을 닦음.
* 그러나, 이미지 합성의 맥락에서, DM의 생성력을 클래스-레이블 또는 인풋 이미지의 흐려진 변형을 넘어선 다른 조건화 타입과 결합하는 것은 지금까지 잘 연구되지 않은 지역임.
* DM에 내재된 UNet backbone을 교차 어텐션 (cross-attention) 메커니즘으로 증강시켜 더 유연한 조건부의 이미지 생성기로 바꾸고자 하는데, 이는 다양한 인풋 모달리티(modality)의 어텐션 기반의 모델을 학습하는 데 효과적임. 다양한 모달리티(언어 프롬프트 등)로부터 $$y$$를 전처리하기 위해 도메인 특정의 인코더 $$\tau_\theta$$를 도입하는데, 이는 $$y$$를 중간 표현 $$\tau_\theta (y) \in \mathbb{R}^{M \times d_\tau}$$에 projection하고, 그 다음 $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d}}) \cdot V$$를 구현하는 교차 어텐션 레이어를 통해 UNet의 중간 레이어에 매핑됨.
<br/><br/>
![Eq2.5](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/eq2.5.png)
<br/><br/>
* 여기서 $$\varphi_i (z_t) \in \mathbb{R}^{N \times d_\epsilon^i}$$는 UNet 구현 $$\epsilon_\theta$$의 (평면화된) 중간 표현을 의미하고 $$W_V^{(i)} \in \mathbb{R}^{d \times d_\epsilon^i}$$, $$W_Q^{(i)} \in \mathbb{R}^{d \times d_\tau}$$, $$W_K^{(i)} \in \mathbb{R}^{d \times d_\tau}$$는 학습 가능한 projection 행렬들임. 시각적 묘사는 [그림 3](#Fig3) 참고.
* 이미지 조건화 짝에 기반을 두어, 다음을 통해 조건부 LDM을 학습함.
<br/><br/>
<div id="Eq3"></div>
![Eq3](/assets/high-resolution-img-synthesis-w-latent-diffusion-models/eq3.png)
<br/><br/>
여기서 $$\tau_\theta$$와 $$\epsilon_\theta$$는 둘 다 [Eq. 3](#Eq3)을 통해 합동으로 최적화됨.
* 이 조건화 메커니즘은 $$\tau_\theta$$가 도메인 특정의 전문가들, 즉 $$y$$가 텍스트 프롬프트일 때 (마스크 없는) transformer들로 파라미터화될 수 있음으로 해서 유연성이 생김.([4.3.1절] 참고)

## Conclusions

* 품질을 저하시키지 않고 디노이징 확산 모델의 트레이닝 및 샘플링 효율성을 크게 향상시키는 방법으로 잠재 확산 모델 (LDM) 제안
* 교차 어텐션 조건화 메커니즘 기반으로, task 특정화 구조 없이 광범위한 조건부 이미지 합성 task에서 SotA 기법에 비해 유리한 결과를 보여줄 수 있음

[High-Resolution Image Synthesis with Latent Diffusion Models]: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
[CVPR 2022]: https://openaccess.thecvf.com/CVPR2022?day=all
[Github]: https://github.com/CompVis/latent-diffusion
[23]: https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html
[66]: https://proceedings.mlr.press/v139/ramesh21a.html