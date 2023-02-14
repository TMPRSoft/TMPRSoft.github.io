---
layout: post
title:  "[리뷰] (PITI) Pretraining is All You Need for Image-to-Image Translation"
date:   2023-02-09 15:49:00 +0900
categories: [Diffusion Model]
tags: [Diffusion Model, Image Translation]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [Pretraining is All You Need for Image-to-Image Translation]
<br/>
Authors: T. Wang, T. Zhang, B. Zhang, H. Ouyang, D. Chen, Q. Chen, and F. Wen
<br/>
Submitted on 25 May 2022
<br/>
Project Page: [Github]

## Abstract

* Image-to-Image (I2I) 변환(translation)을 강화하기 위해 pretraining을 사용할 것을 제안함.
* 이전의 I2I 변환 기법은 보통 전용 구조 설계가 필요하고 개별 변환 모델을 처음부터 트레이닝하여 특히 paired 트레이닝 데이터가 충분하지 않을 시 복잡한 장면의 고품질 생성에 어려움이 있음.
* 본 논문에서는 각 I2I 변환 문제를 downstream task로 간주하고 다양한 종류의 I2I 변환을 수행하기 위해 사전 트레이닝 (pretrain) 된 확산 모델(diffusion model)을 적용하는 간단하고 일반적인 프레임워크를 소개함.
* 또한 생성 품질을 향상시키기 위해 정규화된 guidance 샘플링과 함께 확산 모델 트레이닝에서 텍스처 합성을 향상시키는 적대적 트레이닝을 제안함.
* ADE20K, COCO-Stuff, DIODE과 같은 난이도 있는 벤치마크에 대한 다양한 task에 걸쳐 광범위한 경험적 비교를 제시하여 제안된 사전 트레이닝 기반 I2I 변환(PITI)이 전례 없는 사실적이고 충실하게 이미지 합성을 할 수 있음을 보임.

![Fig1](/assets/pretraining-is-all-you-need-for-i2i-translation/fig1.png)

그림 1: 주어진 semantic 레이아웃 또는 스케치에서 본 기법으로 샘플링된 다양한 이미지들.

## Introduction

* PITI로 칭해지는 사전 트레이닝 기반의 I2I 변환은 마스크-이미지(mastk-to-image), 스케치-이미지(sketch-to-image), 기하학-이미지(geometry-to-image) 변환 같은 다양한 downstream task에서 전례 없는 퀄리티를 달성함.
* 그림 1은 뛰어난 품질과 큰 다양성을 나타내는 복잡한 장면의 일부 생성된 이미지 샘플을 보여 줌.
* ADE20K, COCO-Stuff, DIODE를 포함한 어려운 데이터셋에 대한 광범위한 실험은 사전 트레이닝 없이 모델뿐 아니라 정량적 metric과 주관적 평가 모두에 의해 측정된 본 접근 방식의 상당한 우수성을 보여 줌.
* 또한 제안된 방법은 few-shot I2I 변환에서 유망한 잠재력을 보여 줌.

## Approach

### Preliminary

* 확산 모델은 점진적인 노이징 프로세스를 거꾸로 진행하여 반복적으로 이미지를 생성함. 정방향 프로세스 $$q$$를 $$T$$ 스텝 동안 가우시안 노이즈를 점진적으로 추가하여 이미지 $$x_0 \sim q(x_0)$$를 손상시킴. 여기서 $$\beta_t$$는 노이즈의 분산을 결정함.
<br/><br/>
![Eq1](/assets/pretraining-is-all-you-need-for-i2i-translation/eq1.png)
<br/><br/>
* 그래서, 이 정방향 프로세스는 점진적으로 노이즈가 있는 잠재 변수들의 시퀀스 $$x_1, \cdots, x_T$$를 산출하고, 충분한 노이징 스텝 후 완전한 노이즈, 즉 $$x_T \sim \mathcal{N}(0, I)$$에 도달함.
* 중요한 것은, 중간 스텝들을 전부 무시하고 아래와 같이 $$x_0$$로부터 $$x_t$$를 직접 유도할 수 있다는 것임.
<br/><br/>
![Eq2](/assets/pretraining-is-all-you-need-for-i2i-translation/eq2.png)
<br/><br/>
여기서 $$\alpha_t := \prod_{i=1}^{t}(1-\beta_i)$$임. 또는 동등하게, $$x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\epsilon$$을 얻을 수 있고, 여기서 $$\epsilon$$은 표준 가우시안 노이즈임.

* 데이터 분포로부터 이미지를 생성하기 위해, 가우시안 노이즈 $$x_T \sim \mathcal{N}(0, I)$$로부터 시작하는 디노이징 모델을 트레이닝 할 수 있고 반복적으로 $$x_{T-1}, \cdots, x_1, x_0$$ 순으로 노이즈를 줄임.
* 디노이징 모델 $$\epsilon_\theta(x_t,t)$$는 노이즈 있는 인풋 $$x_t$$을 가지고 타임 스텝 $$t$$에서 다음과 같이 mean square error loss를 사용하여 추가된 노이즈 $$\epsilon$$를 예측함.
<br/><br/>
![Eq3](/assets/pretraining-is-all-you-need-for-i2i-translation/eq3.png)
<br/><br/>
* 디노이징에 대한 이미지 생성의 노이즈 감소는 $$\nabla_{x_t}\log p(x_t) \propto \epsilon_\theta (x_t)$$이므로 디노이징 스코어 매칭이나, 데이터 log-likelihood의 간략화된 variational lower bound를 최적화하는 것으로 정당화될 수 있음.
* 역방향 확산 프로세스를 용이하게 수행하기 위해, 클래스 레이블, 텍스트 프롬프트, 열화된 이미지와 같은 조건 $$\pmb{y}$$를 추가로 제공할 수 있음. 그러면 디노이징 모델은 $$\epsilon_\theta(x_t, \pmb{y}, t)$$가 되고 조건은 input concatenation이나 denormalization, cross-attention을 통해 주입됨.
* 광범위한 이미지에서의 탁월한 생성 능력으로 인해, 확산 모델은 생성 선행(genrative prior) 역할로서 이상적인 선택이 되고 있음. 아래에서, 대량의 데이터를 사용하여 네트워크를 적절히 사전 트레이닝하고 학습된 지식을 그림 2와 같이 downstream task에 적용하는 방법을 설명함.

![Fig2](/assets/pretraining-is-all-you-need-for-i2i-translation/fig2.png)

그림 2: 전반적인 프레임워크. 여러 pretext task를 통해 거대한 데이터에서 사전 트레이닝을 수행하고 일반적이고 고품질의 이미지 통계를 모델링하는 고도의 semantic 잠재 공간을 학습함. Downstream task를 위해, 이 사전 트레이닝된 semantic 공간에 task 특정의 조건들을 매핑하기 위한 조건부 finetuning을 수행함. 사전 트레이닝한 지식을 활용하여, 본 모델은 여러 조건에 기반을 둔 그럴 듯한 이미지들을 표현함.

### Generative pretraining

* 판별 task에 대해서와 같이 같은 도메인으로부터 이미지를 취하는 것과 반대로, 생성 task에 대한 사전 트레이닝 된 모델은 별개의 downstream task에서 매우 다양한 종류의 이미지를 소모함. 그래서, 생성 사전 트레이닝 동안, 확산 모델이 나중에 모든 downstream task에 사용하기 위해 공유되는 잠재 공간으로부터 이미지를 생성할 것이 예상됨.
* 중요한 것은, 사전 트레이닝 된 모델은 고도의 semantic 공간, 즉 이웃한 점끼리는 대응되는 이미지가 semantic하게 유사한 공간을 가지는 것이 바람직함.
* 이런 식으로, downstream finetuning은 어려운 이미지 합성(그럴 듯한 레이아웃과 현실적인 텍스처 표현)이 사전 트레이닝 된 지식을 사용하여 수행되는 동안, 오직 task 특정의 인풋을 이해하는 것에만 관여함.

* 위를 수행하기 위해, 확산 모델을 사전 트레이닝 하여 semantic 인풋에 조건을 부여하는 것을 제안함.
* 시각-언어적 사전 트레이닝의 상당한 전환 가능성에 영감을 받아, 텍스트로 조건을 부여받고 거대하고 다양한 텍스트-이미지 쌍 데이터로 트레이닝 된 GLIDE 모델을 채택함.
* 특히, transformer 네트워크는 텍스트 인풋을 인코딩하고 확산 모델에 주입되는 텍스트 토큰을 생산함. 텍스트 임베딩 공간은 선천적으로 semantic임.
* 많은 최근 연구들과 유사하게, GLIDE는 계층적 생성 scheme을 활용하는데, 이것은 해상도 $$64 \times 64$$에서 기반 확산 모델로 시작하고, 그 다음 $$64 \times 64$$에서 $$256 \times 256$$ 해상도까지 확산 upsampling 모델이 이어짐. 본 실험은 공공 GLIDE 모델에서 수행되었고, 해당 모델은 사람과 폭력적인 객체는 제거된 약 6,700만 텍스트-이미지 쌍 데이터에서 트레이닝 됨.

### Downstream adaptation

* 모델이 사전 트레이닝 되고 나면, 그것을 여러 가지 전략으로 기반 모델과 upsampler 모델을 각각 finetuning 하여 다양한 downstream 이미지 합성 task에 적응시킬 수 있음.

* **기반 모델 finetuning**<br/>
    * 기반 모델을 사용하는 생성은 $$x_t = \tilde{\mathcal{D}}(\tilde{\mathcal{E}}(x_0, \pmb{y}))$$로 식이 세워질 수 있고, 여기서 $$\tilde{\mathcal{E}}$$와 $$\tilde{\mathcal{D}}$$는 각각 사전 트레이닝 된 인코더와 디코더를 나타내고 $$\pmb{y}$$는 사전 트레이닝에 사용되는 조건임.
    * 텍스트 이상의 새로운 modality 조건을 수용하기 위해, task 특정 head $$\mathcal{E}_i$$를 트레이닝 하여 조건부 인풋을 사전 트레이닝 된 임베딩 공간에 매핑함. 인풋이 충실하게 projection 될 수 있다면, 사전 트레이닝 된 디코더는 그럴 듯한 아웃풋을 생산할 것임.

* 2단계 finetuning scheme을 제안함. 첫 단계에서, 특히 task 특정 인코더를 트레이닝하고 사전 트레이닝 된 디코더는 그대로 둠. 이 단계에서 아웃풋은 인풋의 semantic과 대강 매치될 것이지만, 정확한 공간적 정렬은 되어 있지 않음.
* 그 다음 인코더와 디코더를 같이 finetuning 함. 이 다음, 공간적 semantic 정렬을 훨씬 개선하게 됨.
* 이러한 단계별 트레이닝은 사전 트레이닝 된 지식을 가능한 많이 구축하는 데 도움이 되고 품질을 훨씬 개선하는 데에 중요한 것으로 입증됨.

* **적대적 확산 upsampler**<br/>
    * 고해상도 생성을 위해 확산 upsampler를 더욱 finetuning 함. 트레이닝 이미지와 기반 모델의 샘플들 사이의 격차를 줄이기 위해 random degradation, 특히 real-world BSR degradation을 적용함.
    * 특히, oversmoothed 효과를 모방하기 위해 L0 필터 역시 도입함.
    * 그럼에도, 강력한 데이터 증강(data augmentation)을 적용하더라도 여전히 oversmoothed 결과를 보임. 이 이슈는 확산 디노이징 처리에서 가우시안 노이즈 가정에서 발생한 것으로 추측됨.
    * 따라서, 노이즈 예측을 위한 표준 MSE (mean square error) 손실을 계산하는 것 외에도, 로컬 이미지 구조의 지각적 현실성(perceptual realism)을 개선하기 위해 지각적 loss와 적대적 loss를 부과하는 것을 제안함. 이미지 예측 $$\hat{\pmb{x}}_0^t = (\pmb{x}_t - \sqrt{1-\alpha_t} \pmb{\epsilon}_\theta (\pmb{x}_t, \pmb{y}, t)) / \sqrt{\alpha_t}$$에서 계산되는 지각적 loss와 적대적 loss는 다음과 같음.
    <br/><br/>
    ![Eq4](/assets/pretraining-is-all-you-need-for-i2i-translation/eq4.png)
    ![Eq5](/assets/pretraining-is-all-you-need-for-i2i-translation/eq5.png)
    <br/><br/>
    여기서 $$D_{\theta}$$는 $$\mathcal{L}_\mathrm{adv}$$를 최대화하려는 적대적 판별기이고, $$\pmb{\psi}_m$$은 사전 트레이닝 된 VGG 네트워크로부터의 멀티레벨 특징점들을 의미함.

### Normalized classifer-free guidance

* 확산 모델은 조건부 인풋을 무시하여 관련 없는 결과를 만들 가능성도 있음. 이를 해결하는 한 가지 방법은 분류기 없는 (classifier-free) guidance로, 샘플링 동안 $$p(\pmb{y} \vert \pmb{x}_t)$$와 함께 $$p(\pmb{x}_t \vert \pmb{y})$$를 고려하는 것임. Log 확률 $$p(\pmb{y} \vert \pmb{x}_t)$$의 경사(gradient)는 다음과 같이 추정될 수 있음.
<br/><br/>
![Eq6](/assets/pretraining-is-all-you-need-for-i2i-translation/eq6.png)
<br/><br/>
* 샘플링 동안, 주어진 조건 $$\pmb{y}$$와 무조건 $$\phi$$로 각각 노이즈를 추정할 수 있고, $$\pmb{\epsilon}_\theta(\pmb{x}_t \vert \phi)$$로부터 훨씬 더 샘플들을 생성할 수 있음.
<br/><br/>
![Eq7](/assets/pretraining-is-all-you-need-for-i2i-translation/eq7.png)
<br/><br/>
여기서 $$w \ge 0$$은 guidance 강도를 컨트롤함. 이러한 분류기 없는 guidance는 각자 샘플들의 품질을 개선하기 위한 샘플링 다양성과 trade off 관계임.

* 그러나, 그런 샘플링 과정은 후속의 디노이징를 저해하는 평균 분산 이동(shift)을 일으킴을 발견함.
    * 식 (7)로부터 guide 된 노이즈 샘플 $$\hat{\pmb{\epsilon}}_\theta (\pmb{x}_t \vert \pmb{y})$$은 평균이 $$\hat{\mu} = \mu + w(\mu - \mu_\phi)$$인데, 이는 분류기 없는 guidance로부터 이끌린 평균 이동이 있음을 나타냄.
    * 노이즈 샘플의 분산은 $$\pmb{\epsilon}_\theta (\pmb{x}_t \vert \pmb{y})$$와 $$\pmb{\epsilon}_\theta (\pmb{x}_t \vert \phi)$$가 독립 변수라는 가정하에 $$\hat{\sigma}^2 = (1 + w)^2 \sigma^2 + w^2 \sigma_\phi^2$$과 같이 이동함.
* 이러한 통계적 이동은 전체 $$T$$개의 확산 디노이징 단계 내내 축적되어, oversmoothed 텍스처에 과포화된 이미지를 초래함.
* 이를 해결하기 위해, 원본 추정 $$\pmb{\epsilon}_\theta (\pmb{x}_t \vert \pmb{y})$$에 따라 guide 된 노이즈 샘플 $$\hat{\pmb{\epsilon}}_\theta (\pmb{x}_t \vert \pmb{y})$$ normalize 된 분류기 없는 guidance를 제안함.
<br/><br/>
![Eq8](/assets/pretraining-is-all-you-need-for-i2i-translation/eq8.png)
<br/><br/>
이 제안된 분류기 없는 normalized guidance가 특히 높은 guidance 강도 $$w$$에서 효과적으로 샘플링 품질을 개선할 수 있음을 보일 예정임.

## Experiments

### Implementation details

* 2단계 finetuning scheme을 채택함.
    * 단계 1: 디코더를 고정시키고 인코더를 learning rate $$3.5 \times 10^{-5}$$, 배치 사이즈 128로 트레이닝 함.
    * 단계 2: 전체 모델을 learning rate $$3 \times 10^{-5}$$로 함께 트레이닝 함.
<br/><br/>
![Fig3](/assets/pretraining-is-all-you-need-for-i2i-translation/fig3.png)
그림 3. COCO 및 ADE20K에서의 시각적 비교. 더 많은 결과는 **부록**에서 확인.
<br/><br/>
![Fig4](/assets/pretraining-is-all-you-need-for-i2i-translation/fig4.png)
그림 4. 다른 데이터셋에서의 시각적 비교. 더 많은 결과는 **부록**에서 확인.
<br/><br/>
* **질적 결과들**
    * 그림 3과 그림 4에서 여러 가지 task의 시각적 결과를 나타냄. 복잡한 장면에 대해 심각한 artifact가 발생하는 처음부터 트레이닝된 기법들에 비해, 사전 트레이닝된 모델은 생성된 이미지의 품질과 다양성을 상당히 개선함. COCO 데이터셋이 다양한 조합의 많은 범주를 포함하여, 모든 baseline 기법들은 강인한 구조로 시각적으로 만족스러운 결과를 생성하는 데 실패함. 그에 반해, 본 기법들은 어려운 케이스에서도 올바른 semantic으로 생생한 디테일을 생산할 수 있음. 그림 4는 다양한 인풋 모달리티에 대한 본 접근의 우수한 적용 가능성을 보여 줌.

## Conclusion and Limitation

* 다양한 I2I 변환 task에 대한 사전 트레이닝의 힘을 가져오는 간단하고 보편적인 프레임워크를 제시함.
* 적대적 확산 upsampler 및 분류기 없는 normalized guidance 같은 기술들로 향상된 전체 모델, PITI는 특히 어려운 시나이로에서 SotA 합성 품질을 크게 향상시킴.
* 본 기법의 한 가지 한계는 샘플링된 이미지가 주어진 입력과 충실히 일치하는 데 어려움이 있고, 작은 물체를 놓칠 수 있다는 것임.
* 한 가지 가능성 있는 이유는 사전 트레이닝된 모델의 중간 공간에 정확한 공간적 정보가 부족하다는 점임.

[Pretraining is All You Need for Image-to-Image Translation]: https://arxiv.org/abs/2205.12952
[Github]: https://tengfei-wang.github.io/PITI/index.html