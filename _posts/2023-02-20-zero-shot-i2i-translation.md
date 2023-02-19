---
layout: post
title:  "[리뷰] (pix2pix-zero) Zero-shot Image-to-Image Translation"
date:   2023-02-20 07:48:00 +0900
categories: [Diffusion Model, Few-Shot Learning]
tags: [Diffusion Model, Image Translation, Few-Shot Learning, Zero-Shot Learning]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [Zero-shot Image-to-Image Translation]
<br/>
Authors: G. Parmer, K. K. Singh, R. Zhang, Y. Li, J. Lu, and J.-Y. Zhu
<br/>
Submitted on 6 Feb 2023
<br/>
Project Page: [Github]

## Abstract

* 대규모의 텍스트-이미지 (text-to-image) 생성 모델은 다양하고 고품질의 이미지를 합성하는 놀라운 능력을 보여줬음.
* 그러나 실제 이미지 편집에 이러한 모델을 직접 적용하는 것은 여전히 두 가지 난관이 존재함.
    * 사용자가 인풋 이미지의 모든 시각적 세부 사항을 정확하게 설명하는 완벽한 텍스트 프롬프트를 생각해 내기 어려움.
    * 기존 모델은 특정 영역에서 바람직한 변화를 도입할 수 있지만 인풋 내용을 극적으로 변경하고 원하지 않는 영역에서 예상치 못한 변화를 적용하는 경우가 많음.
* 본 연구에서는, 수동 프롬프트 없이 원본 이미지의 내용을 보존할 수 있는 I2I 변환 (image-to-image translation) 기법인 pix2pix-zero를 제안함.
* 먼저 텍스트 임베딩 공간에서 원하는 편집을 반영하는 편집 방향(edit direction)을 자동으로 발견함.
* 편집 후 일반적인 콘텐츠 구조를 보존하기 위해, 우리는 확산 프로세스 (diffusion process) 전반에 걸쳐 인풋 이미지의 교차 어텐션 (cross-attention) 맵을 유지하는 것을 목표로 하는 교차 어텐션 guidance를 추가로 제안함.
* 또한, 본 기법은 이러한 편집에 대한 추가 트레이닝이 필요하지 않으며 기존의 사전 트레이닝(pretrain)된 텍스트-이미지 확산 모델(diffusion model)을 직접 사용할 수 있음.
* 광범위한 실험을 수행하고 본 기법이 실제 및 합성 이미지 편집 모두에서 기존이나 같은 시기의 연구들을 능가한다는 것을 보임.

![Fig1](/assets/zero-shot-i2i-translation/fig1.png)

그림 1: pix2pix-zero: 사용자가 편집 방향을 즉석에서 지정할 수 있는 (예: cat $$\rightarrow$$ dog) 확산 기반의 I2I 변환 기법. 실제(위쪽 두 열)와 합성(아래쪽 열) 이미지 둘 다에서 다양한 변환 task를 수행하면서, 인풋 이미지의 구조를 보존함. 본 기법은 각 인풋 이미지의 수동 텍스트 프롬프트도, 각 task의 비용 높은 fine-tuning도 필요하지 않음.

## Introduction

* 본 연구에는 두 가지 핵심 contribution이 있음.
    * 인풋 텍스트 프롬프트 없는 효율적이고 자동 편집 방향 발견 메커니즘. 광범위한 인풋 이미지에 대해 작동하는 일반적인 편집 방향을 자동으로 발견함. 원 단어(예: cat)와 편집된 단어(예: dog)이 있다면, 원본 단어와 편집된 단어를 포함하는 문장들의 두 그룹을 따로 생성함. 그 다음 이 두 그룹 사이의 CLIP 임베딩 방향을 계산함. 이 편집 방향이 다수의 문장들에 기반을 두므로, 오직 원본과 편집된 단어들 사이의 방향만을 찾는 것보다는 더 강인함.
    * 교차 어텐션 guidance를 통한 콘텐츠 보존. 교차 어텐션 맵이 생성된 객체의 구조에 대응함을 발견함. 원 구조를 보존하기 위해, 텍스트-이미지 교차 어텐션 맵이 변환 전후에 일관성을 가지도록 함. 확산 프로세스 내내 이 일관성을 강제하기 위해 교체 어텐션 guidance를 적용함.
* 그림 1은 본 기법을 사용하여 인풋 이미지의 구조를 보존하면서 다양한 편집 결과를 보임.
* 다음 일련의 기술들을 통해 결과와 추론 속도를 더욱 향상시킴.
    * Autocorrelation 정규화(regularization): DDIM을 통해 inversion을 적용할 때, DDIM inversion이 중간 예측 노이즈를 덜 가우시안하게 만드는 경향이 있으며, 이는 invert 된 이미지의 편집성을 감소시킨다는 것을 발견함. 따라서, inversion 중에 노이즈가 가우시안에 가깝도록 보장하기 위해 autocorrelation 정규화를 도입함.
    * 조건부 GAN distillation: 확산 모델은 비용이 많이 드는 확산 프로세스의 다단계 추론(inference)으로 인해 느림. 대화형 (interactive) 편집을 가능하게 하기 위해, 확산 모델에서 원본 이미지와 편집된 이미지의 쌍을 이룬 데이터를 제공하여, 실시간 추론을 가능하게 하는 빠른 조건부 GAN 모델로 확산 모델에 distillation을 수행함.
* 실제 이미지와 합성 이미지 모두에 대해 전경 객체 변경(cat $$\rightarrow$$ dog), 객체 수정(cat 이미지에 안경 추가), 입력 스타일 변경(스케치 $$\rightarrow$$ 오일 파스텔)과 같은 광범위한 I2I 변환 작업에 대한 방법을 설명함.
* 광범위한 실험에 따르면 pix2pix-zero는 포토리얼리즘 및 콘텐츠 보존과 관련하여 기존 및 같은 시기의 연구들을 능가함.

## Method

* 인풋 이미지를 편집 방향(예: cat $$\rightarrow$$ dog)에 따라 편집하는 방법을 제안함. 먼저 인풋 $$\tilde{x}$$을 **Inverting Real Images**에서의 대응하는 노이즈 맵으로 deterministic한 방식으로 invert함. **Discovering Edit Directions**에서는 텍스트 임베딩 공간에서 편집 방향을 자동으로 발견하고 사전 계산하는 기법을 제안함.
* 편집 방향을 단순하게 적용하면 이미지 컨텐츠에서 종종 원하지 않은 결과를 얻음. 이 이슈를 해결하기 위해, 확산 샘플링 프로세스를 가이드하고 인풋 이미지의 구조를 유지하는 데 도움을 주는 교차 어텐션 guidance를 제안함(**Editing via Cross-Attention Guidance**).
* 본 기법은 여러 텍스트-이미지 모델에 적용 가능하지만 본 논문에서는 인풋 이미지 $$\tilde{x} \in \mathbb{R}^{X \times X \times 3}$$를 잠재 코드(latent code) $$x_0 \in \mathbb{R}^{S \times S \times 4}$$에 인코딩하는 Stable Diffusion을 사용함.
* 실험에서, $$X = 512$$는 이미지 사이즈이고, $$S = 64$$는 downsampling 된 latent 사이즈임.
* 이 절에서 설명된 inversion과 편집은 잠재 공간(latent space)에서 이루어짐.
* 텍스트 조건부 모델을 invert 하기 위해, BLIP을 사용하여 인풋 이미지 $$\tilde{x}$$를 설명할 초기 텍스트 프롬프트 $$c$$를 생성함.

### Inverting Real Images

* **Deterministic inversion**
    * Inversion은 샘플링 시 인풋 잠재 코드 $$x_0$$를 재구성하는 노이즈 맵 $$x_\mathrm{inv}$$를 찾는 것을 수반함.
    * DDPM에서, 이것은 고정된 순방향 노이징 프로세스와 그 다음의 역방향 프로세스의 디노이징 프로세스에 해당함. 그러나 DDPM의 순방향 및 역방향 프로세스는 모두 확률적이며 재구성을 충실히 수행하지 않음. 대신, 아래와 같이 deterministic DDIM 역방향 프로세스를 채택함.
    <br/><br/>
    ![Eq1](/assets/zero-shot-i2i-translation/eq1.png)
    <br/><br/>
    여기서 $$x_t$$는 타임스텝 $$t$$에서의 노이즈 있는 잠재 코드, $$\epsilon_\theta (x_t, t, c)$$는 타임스텝 $$t$$와 인코딩된 텍스트 특징점 $$c$$의 조건으로 $$x_t$$에서 추가된 노이즈를 예측하는 UNet 기반의 디노이저, $$\bar{\alpha}_{t+1}$$는 DDIM에 정의된 대로 노이즈 스케일링 factor이고, $$f_{\theta}(x_t, t, c)$$는 최종 디노이징 된 잠재 코드 $$x_0$$를 예측함.
    <br/><br/>
    ![Eq2](/assets/zero-shot-i2i-translation/eq2.png)
    <br/><br/>
    * DDIM 프로세스를 사용하여 초기 잠재 코드 $$x_0$$에 노이즈를 점진적으로 추가하고 inversion의 끝에서, 최종 노이징 된 잠재 코드 $$x_T$$는 $$x_\mathrm{inv}$$에 할당됨.

* **노이즈 정규화**
    * DDIM inversion $$\epsilon_\theta (z_t, t, c) \in \mathbb{R}^{S \times S \times 4}$$로 생성된 invert 된 노이즈 맵들은 종종 uncorrelate 된 가우시안 white 노이즈의 통계적 특성을 따르지 않고, 저조한 편집성을 일으킴.
    * 가우시안 white 노이즈 맵은 (1) 랜덤 위치의 어떤 짝 사이에도 correlation이 없어야 하고 (2) 각 공간적 위치에서 평균 0, 분산 1(zero-mean unit-variance)이어야 하는데, 이는 Kronecker 델타 함수인 autocorrelation 함수에 반영될 것임.
    * 이에 따라, inversion 프로세스를 autocorrelation 목적으로 가이드하는데, 해당 목적은 개개의 픽셀 위치에서 짝별 (pairwise) 항 $$\mathcal{L}_\mathrm{pair}$$와 KL 발산(divergence) 항 $$\mathcal{L}_\mathrm{KL}$$로 구성됨.
<br/><br/>
    * 모든 위치에서의 짝들을 조밀하게 샘플링하는 것은 비용이 많이 드므로, [Karras et al. 2020]을 따라서 피라미드를 형성하고, 여기서 초기 노이즈 레벨 $$\eta^0 \in \mathbb{R}^{64 \times 64 \times 4}$$는 예측된 노이즈 맵 $$\epsilon_\theta$$이고, 각 후속의 노이즈 맵은 $$2 \times 2$$ 근방으로 (그리고 예상 분산을 보존하기 위해 2를 곱함) 평균적으로 풀링됨. 특징점 사이즈 $$8 \times 8$$에서 멈춰, 4개의 노이즈 맵을 생성하여 집합 $$\{\eta^0, \eta^1, \eta^2, \eta^3\}$$을 형성함.
<br/><br/>
    * 피라미드 레벨 $$p$$에서 짝별 정규화는 노이즈 맵 사이즈들 $$S_p$$에 걸쳐 normalize 된, 가능한 $$\delta$$개의 offset에서의 auto-correlation 계수(coefficient)의 제곱의 합임.
    <br/><br/>
    ![Eq3](/assets/zero-shot-i2i-translation/eq3.png)
    <br/><br/>
    여기서 $$\eta_{x,y,c}^p \in \mathbb{R}$$는 원형 인덱싱(circular indexing)을 사용하여 공간적 위치와 채널로 indexing함.
    * [Karras et al. xxxx]은 이전에 GAN inversion에 대한 autocorrelation regularizer를 사용하여 노이즈 맵에 탐사했음. 이 autocorrelation 아이디어를 약간 변경하여 확산 맥락에서 성능을 증폭시킴.
        * [Karras et al. 2020]에서처럼 $$\delta = 1$$만을 사용하기보단, 각 iteration에서 shift를 랜덤으로 샘플링하여, 장거리 정보를 더 효율적으로 전파시키도록 함.
    * 확산 맥락에서, 장거리 연결을 전파하기 위해 다수의 iteration에 의존하면 중간의 타임 스텝이 분포를 벗어나므로, 각 타임 스텝이 잘 정규화되는 것이 중요하다고 가정함.
<br/><br/>
    * 게다가, normalization을 통한 엄격한 평균 0, 분산 1 기준 강제([Karras et al. 2020])는 디노이징 프로세스 동안 발산을 초래함을 발견함. 대신, 가변성 autoencoder에서 사용한 것처럼, 이것을 loss $$\mathcal{L}_\mathrm{KL}$$로 유하게 공식화함. 이것은 두 loss 사이를 유하게 균형을 잡도록 함. 최종 autocorrelation 정규화는 $$\mathcal{L}_\mathrm{auto} = \mathcal{L}_\mathrm{pair} + \lambda\mathcal{L}_\mathrm{KL}$$이고, 여기서 $$\lambda$$는 이 두 항의 균형을 잡음.

![Fig2](/assets/zero-shot-i2i-translation/fig2.png)

그림 2: **편집 방향 발견.** 소스와 타깃 텍스트(예: cat, dog)가 주어졌을 때, GPT-3을 사용하여 다양한 문장들의 대형 bank를 생성함. CLIP 임베딩을 계산하고 평균 차이를 가지고 편집 방향 $$\Delta c_\mathrm{edit}$$을 얻음.

### Discovering Edit Directions

* 최근의 대규모 생성 모델은 사용자들이 아웃풋 이미지를 설명하는 문장을 지정하여 이미지 합성을 컨트롤할 수 있게 함.
* 대신, 여기서는 사용자들에게 소스 도메인에서 타깃 도메인으로 (예: cat $$\rightarrow$$ dog) 원하는 변경 사항만 제공하면 되는 인터페이스를 제공하고자 함.
<br/><br/>
* 그림 2와 같이, 소스에서 타깃으로 해당 텍스트 임베딩 방향 벡터 $$\Delta c_\mathrm{edit}$$를 자동으로 계산함.
* GPT-3와 같은 기성 문장 생성기를 사용하거나 소스와 타깃 주위에 미리 정의된 프롬프트를 사용하여 소스 $$s$$와 타깃 $$t$$ 모두에 대해 다양한 문장들의 대형 bank를 생성함. 그 다음 문장들의 CLIP 임베딩 사이의 평균 차이를 계산함.
* 텍스트 프롬프트 임베딩에 방향을 추가하여 편집된 이미지가 생성될 수 있음. 그림 4는 이 접근법을 사용하여 계산된 방향으로, 몇 가지 편집의 결과를 나타냄.
* 단일 단어보다 여러 문장을 사용한 텍스트 방향이 더 강인하다는 것을 발견하고 실험에서 입증함.
* 편집 방향을 계산하는 방식은 5초 정도밖에 걸리지 않고 한 번만 사전 계산하면 됨.
* 그 다음, 이 편집 방향들을 I2I 변환 기법에 통합함.

### Editing via Cross-Attention Guidance

* 최근의 대규모 확산 모델들은 디노이징 네트워크 $$\epsilon_\theta$$에 교체 어텐션 레이어를 증강시켜 조건부를 통합시킴.
* 여기서는 잠재 확산 모델(LDM)로 구축된 오픈소스 Stable 확산 모델을 사용함. 이 모델은 CLIP 텍스트 인코더로 텍스트 임베딩 $$c$$를 생산함.
* 다음, 텍스트에 조건을 주기 위해, 모델은 인코딩된 텍스트와 디노이저 $$\epsilon_\theta$$의 중간 특징점들 사이에 교차 어텐션을 계산함.
<br/><br/>
![Eq4](/assets/zero-shot-i2i-translation/eq4.png)
<br/><br/>
* 쿼리 $$Q = W_{Q\varphi}(x_t)$$, key $$K=W_{K}c$$, value $$V = W_{V}c$$는 디노이징 UNet $$\epsilon_\theta$$의 중간 공간적 특징점들 $$\varphi(x_t)$$와 텍스트 임베딩 $$c$$에 적용된 학습된 projection $$W_Q$$, $$W_K$$, $$W_V$$로 계산되고, $$d$$는 projection 된 key와 쿼리의 차원임.
* 특히 관심을 끄는 것은 이미지의 구조와 밀접한 관계가 있는 것으로 관찰되는 교체 어텐션 맵 $$M$$임. 마스크의 개개의 엔트리 $$M_{i,j}$$는 $$i$$번째 공간적 위치를 향한 $$j$$번째 텍스트 토큰의 contribution을 표현함. 또한, 교차 어텐션 마스크는 타임 스텝에 따라 다르고, 각 타임 스텝 $$t$$마다 다른 어텐션 마스크 $$M_t$$를 얻음.
<br/><br/>
* 편집을 적용하기 위해, 단순한 방법은 미리 계산된 편집 방향 $$\Delta c_\mathrm{edit}$$을 $$c$$에 적용하여, $$c_\mathrm{edit} = c + \Delta c_\mathrm{edit}$$을 $$x_\mathrm{edit}$$을 생성하기 위한 샘플링 프로세스에 사용함.
* 이 접근법은 편집에 따라 이미지를 변경하는 데 성공하지만 인풋 이미지의 구조를 보존하지는 못함. 그림 3의 아래쪽 행과 같이, 샘플링 과정에서 교차 어텐션 맵의 편차는 영상의 구조에 편차를 초래함.
* 따라서, 교차 어텐션 맵의 일관성을 유지하기 위한 새로운 교차 어텐션 guidance를 제안함.

![Fig3](/assets/zero-shot-i2i-translation/fig3.png)

그림 3: **pix2pix-zero 기법의 overview.** cat $$\rightarrow$$ dog 편집 예시로 설명함. 먼저, 정규화된 DDIM inversion을 적용하여 invert 된 노이즈 맵을 얻음. 이것은 텍스트 임베딩 $$c$$으로 가이드되고, 이미지 캡셔닝 네트워크 BLIP과 CLIP 텍스트 임베딩 모델을 사용하여 자동으로 계산됨. 그 다음, 원본 텍스트 임베딩을 디노이징하여 교차 어텐션 맵들을 얻는데, 이는 인풋 이미지 구조에 대한 레퍼런스로 활용됨(위쪽 행). 다음, 편집된 텍스트 임베딩, $$c + \Delta c_\mathrm{edit}$$으로 디노이징하고, loss를 교차 어텐션 맵이 레퍼런스 교차 어텐션 맵에 매치시키도록 하는 게 사용함(2번째 행). 이것은 편집된 이미지의 구조가 원본 이미지에 비해 극적으로 변하지 않도록 함. 교차 어텐션 guidance 없는 디노이징은 3번째 행에 나타나 있고, 구조적으로 큰 편차를 초래함.

* Algorithm 1과 그림 3에서 설명된 대로 두 단계 프로세스를 따름.
    * 먼저, 편집 방향 적용 없이, 인풋 텍스트 $$c$$를 사용하여 각 타임 스텝 $$t$$에 대한 레퍼런스 교차 어텐션 맵들 $$M_t^\mathrm{ref}$$을 얻어 이미지를 재구성함. 이 교차 어텐션 맵들은 원본 이미지의 구조 $$e$$에 대응하는데, 이 구조가 보존하고자 하는 것임.
    * 다음, 이 편집 방향을 $$c_\mathrm{edit}$$을 사용하여 교차 어텐션 맵들 $$M_t^\mathrm{edit}$$을 생성함으로써 적용함. 그리고 레퍼런스 $$M_t^\mathrm{ref}$$에 매칭하는 방향으로 $$x_t$$와 경사 스텝(gradient step)을 취하고, 아래와 같이 교차 어텐션 loss $$\mathcal{L}_\mathrm{xa}$$를 감소시킴.
    <br/><br/>
    ![Eq5](/assets/zero-shot-i2i-translation/eq5.png)
    <br/><br/>
    이 loss는 $$M_t^\mathrm{edit}$$이 $$M_t^\mathrm{ref}$$로부터 벗어나지 않도록 하여, 원본 구조를 유지하면서 편집을 적용함.

## Limitations and Discussion

* 구조를 보존하는 이미지 편집을 수행하기 위해 사전 트레이닝된 텍스트-이미지 확산 모델을 사용한 I2I 변환 기법을 제안함.
* 자동 방식을 도입하여 텍스트 임베딩 공간에서 편집 방향을 학습함.
* 또한 교차 어텐션 맵 guidance를 제안하여 학습된 편집 방향을 적용한 후 원본 이미지의 구조를 보존함.
* 세부적인 양적 및 질적 결과를 제공하여 본 접근법의 효과를 보임.
* 본 기법은 트레이닝과 프롬프팅이 없음.

![Fig8](/assets/zero-shot-i2i-translation/fig8.png)

그림 8: **한계.** 본 기법은 객체 자세가 특이할 때 (예: 좌측의 고양이) 여러 케이스에서 실패하고 교차 어텐션 맵의 저해상도 때문에 정교한 공간적 위치 세부 내용 보존에 대해 가끔 실패함(예: 우측의 다리 위치와 꼬리).

* **한계**
    * 본 연구의 한 가지 한계는 구조 guidance가 교차 어텐션 맵의 해상도에 제한되어 있다는 점임. Stable 확산에 대해, 교차 어텐션 맵에 대한 해상도는 $$64 \times 64$$인데 이는 매우 결이 고운 구조 컨트롤에 충분하지 않을 수 있음.
        * 그림 8에서처럼, 편집된 얼룩말은 다리와 꼬리의 결이 고운 세부 내용을 따르지 않음.
    * 본 접근법은 교차 어텐션 맵의 아무 해상도와 잘 동작하나, 기반 모델이 교차 어텐션 맵에 대해 더 높은 해상도를 가진다면, 본 접근법은 더욱 정교한 구조 guidance 컨트롤을 제공할 수 있음.
    * 또한, 본 기법은 특이한 자세를 취하는 객체의 여러 케이스에서는 실패할 수 있음(그림 8의 고양이).

[Zero-shot Image-to-Image Translation]: https://arxiv.org/abs/2302.03027
[Github]: https://pix2pixzero.github.io/
[Karras et al. 2020]: https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html