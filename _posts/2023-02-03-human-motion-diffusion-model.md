---
layout: post
title:  "[리뷰] Human Motion Diffusion Model"
date:   2023-02-02 15:00:00 +0900
categories: [Diffusion Model]
tags: [Diffusion Model, Generative Model]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [Human Motion Diffusion Model]
<br/>
Authors: Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H. Bermano
<br/>
Submitted on 29 Sep 2022
<br/>
Github: [https://guytevet.github.io/mdm-page/]

## Abstract

* 자연스럽고 표현적인 인간의 움직임 생성은 컴퓨터 애니메이션의 성배라고도 할 수 있음.
* 가능한 움직임의 다양성, 그것에 대한 인간의 지각적 민감성, 그리고 그것의 정확한 묘사의 어려움 때문에, 현재 생성 솔루션들은 품질이 낮거나 표현성이 제한적임.
* 이미 다른 영역에서 주목할 만한 생성 능력을 보여준 diffusion model은 다대다 특성으로 인해 인간 모션의 유망한 후보이지만, 리소스가 부족하고 컨트롤하기 어려운 경향이 있음.
* 본 논문에서, 인간 모션 영역에 대해 신중하게 적응된 분류기(classifier)가 없는 diffusion 기반 생성 모델인 Motion Diffusion Model(MDM)을 제안함. MDM은 transformer 기반으로, 모션 생성 문헌의 통찰력들이 결합되어 있음.
* 주목할 만한 설계는 각 diffusion 단계에서, 노이즈보다는 샘플의 예측이라는 점임. 이는 발 접촉 loss와 같은 움직임의 위치와 속도에 대해 확립된 geometric loss를 사용하는 것을 용이하게 함.
* MDM은 다양한 조건화 모드들과 다양한 생성 task들을 가능하게 하는 일반적인 접근 방식임. 본 모델은 가벼운 리소스로 트레이닝되었지만 text-to-motion 및 action-to-motion에서 모션에 대한 주요 벤치마크에서 SotA 결과를 달성한다는 것을 보임.

![Fig1](/assets/human-motion-diffusion-model/fig1.png)

그림 1: Motion Diffusion Model(MDM)은 다대다 텍스트 프롬프트가 주어진 다양한 모션을 생성하여 text-to-motion 매핑의 다대다 특성을 반영함. 커스텀 구조와 geometric loss는 고품질 모션을 산출하는 데 도움을 줌. 진한 색일수록 나중 프레임을 의미함.

## Introduction

* 이 논문에서, Motion Diffusion Model(MDM) - 인간 모션 도메인을 위해 신중하게 적응된 diffusion 기반의 생성 모델을 소개함. Diffusion 기반으로, MDM은 

![Fig2](/assets/human-motion-diffusion-model/fig2.png)

그림 2: **(좌) MDM overview.** 본 모델은 조건화 코드 $$c$$와 노이징 스텝 $$t$$에서, 길이 $$N$$의 모션 시퀀스 $$x_t^{1:N}$$을 입력받음. 이 케이스에서의 텍스트 임베딩 기반의 CLIP $$c$$는 먼저 분류기 없는 학습을 위해 랜덤으로 마스킹되고, $$t$$와 함께 인풋 토큰 $$z_{tk}$$으로 projection 됨. 각 샘플링 스텝에서, transformer-인코더는 최종 노이즈 없는 모션 $$\hat{x}_0^{1:N}$$을 예측함. **(우) 샘플링 MDM.** 조건 $$c$$가 주어졌을 때, desired 모션의 차원에서 랜덤 노이즈 $$x_T$$를 샘플링하고, $$T$$에서 $$1$$까지 반복함. 각 스텝 $$t$$에서, MDM은 노이즈 없는 샘플 $$\hat{x}_0$$을 예측하고, $$x_{t-1}$$로 역확산시킴.

## Motion Diffusion Model

* **목표**<br/>
    * 길이 $$N$$의 인간 모션 $$x^{1:N}$$을 주어진 임의의 조건 $$c$$에서 합성하는 것. 이 조건은 오디오, 자연어 (text-to-motion), discrete 클래스 (action-to-motion) 같은 합성을 지시할 아무 현실 신호가 될 수 있음. 게다가, 조건 없는 모션 생성도 가능한데, 이때는 무조건 (null condition) $$c=\phi$$로 표시함.
<br/><br/>
* 생성된 모션 $$x^{1:N}=\{x^i\}_{i=1}^N$$은 관절 회전 또는 위치 $$x^i \in \mathbb{R}^{J \times D}$$로 표현된 인간의 자세의 시퀀스임. 여기서 $$J$$는 관절의 개수이고 $$D$$는 관절 표현의 차원임. MDM은 위치, 회전, 또는 둘 다로 표현된 모션을 받아들일 수 있음.

* **프레임워크**<br/>
    * Diffusion은 Markov noising process $$\{x_t^{1:N}\}_{t=0}^{T}$$으로 모델링되는데, 여기서 $$x_0^{1:N}$$은 데이터 분포와 다음으로부터 얻어짐.
    <br/><br/>
    ![Eq1](/assets/human-motion-diffusion-model/eq1.png)
    <br/><br/>
    여기서 $$\alpha_t \in (0, 1)$$는 상수 하이퍼파라미터임. 만약 $$\alpha_t$$가 충분히 작으면, $$x_T^{1:N} \sim \mathcal{N}(0, I)$$으로 간략화할 수 있음. 여기서부터 $$x_t$$는 노이징 스텝 $$t$$에서의 전체 시퀀스를 의미함.
    * 조건 있는 모션 합성은 분포 $$p(x_0 \vert c)$$를 점진적으로 $$x_T$$의 노이즈를 제거하는 reversed diffusion process로 모델링함. [Ho et al. (2020)]에서 공식화된 것과 같이 $$\epsilon_t$$를 예측하는 대신, [Ramesh et al. (2022)]를 따라 신호 자체를 예측함. 즉, 다음과 같은 simple 목적([Ho et al. (2020)])과 함께 $$\hat{x}_0 = G(x_t, t, c)$$임.
    <br/><br/>
    ![Eq2](/assets/human-motion-diffusion-model/eq2.png)
    <br/><br/>

* **Geometric loss**<br/>
    * 모션 도메인에서, 생성 네트워크는 geometric loss([Petrovich et al. (2021)]; [Shi et al. (2020)])를 사용하여 표준정규화됨. 이 loss는 물리적 특성을 강화하고 artifact를 예방하여, 자연스럽고 일관된 모션을 이루게 함. 이 연구에서 (1) 위치(회전을 예측하는 경우), (2) 발 접촉, (3) 속도를 정규화하는 세 가지 일반적인 geometric loss를 실험함.
    <br/><br/>
    ![Eq3](/assets/human-motion-diffusion-model/eq3.png)
    ![Eq4](/assets/human-motion-diffusion-model/eq4.png)
    ![Eq5](/assets/human-motion-diffusion-model/eq5.png)
    <br/><br/>
    * 관절 회전을 예측하는 경우, $$FK(\cdot)$$는 관절 회전을 관절 위치로 변환하는 정방향 kinematic 함수임(그 외의 경우, 항등 함수를 나타냄).
    * $$f_i \in \{0, 1\}^J$$는 각 프레임 $$i$$에 대한 바이너리 발 접촉 마스크임. 발에만 관련하여, 발이 땅에 닿아 있는지를 나타내고, 바이너리 ground truth 데이터에 따라 설정됨([Shi et al., 2020)]). 본질적으로, 그것은 땅에 닿을 때 속도를 무효화하여 발 미끄러짐 효과를 완화함.
<br/><br/>
* 전반적으로, 트레이닝 loss는 다음과 같음.
<br/><br/>
![Eq6](/assets/human-motion-diffusion-model/eq6.png)
<br/><br/>
* **모델**<br/>
    * 모델은 그림 2에 묘사되어 있음. $$G$$를 직관적인 transformer ([Vaswani et al, 2017]) 인코더 전용 구조로 구현함. Transformer 구조는 시간적으로 인식되어, 임의의 길이의 모션을 학습하는 것이 가능하며, 모션 도메인에 대해 잘 입증됨([Petrovich et al., 2021]; [Duan et al., 2021]; [Aksan et al., 2021]).
    * 노이즈 시간 스텝 $$t$$과 조건 코드 $$c$$는 각각 분리된 feed-forward 네트워크에 의해 transformer 차원으로 projection 되고, 합해져 토큰 $$z_{tk}$$을 산출함.
    * 노이즈 인풋 $$x_t$$의 각 프레임은 transformer 차원으로 선형적으로 projection 되고 표준 위치적 임베딩으로 합해짐.
    * 그 다음 $$z_{tk}$$와 projection 된 프레임들은 인코더에 입력으로 들어감. ($$z_{tk}$$에 대응하는) 첫째 아웃풋 토큰을 제외하고, 인코더 결과는 원본 모션 차원으로 돌아가서 projection되고, 예측 $$\hat{x}_0$$로서 역할을 함.
    * Text-to-motion 구현은 텍스트 프롬프트를 CLIP 텍스트 인코더를 사용하여 $$c$$로 인코딩함으로써 이루어지고, action-to-motion은 클래스 당 학습된 임베딩으로 구현됨.
<br/><br/>
* **샘플링**<br/>
    * $$p(x_0 \vert c)$$으로부터의 샘플링은 [Ho et al., 2020]에 따라 반복 방식으로 이루어짐. 매 타임 스텝 $$t$$마다 노이즈 없는 샘플 $$\hat{x}_0 = G(x_t, t, c)$$을 예측하고, 노이즈를 가해 $$x_{t-1}$$로 되돌림. 이는 $$x_0$$에 도달할 때까지 $$t=T$$에서부터 반복됨(그림 2 (우)).
    * 본 모델 $$G$$를 분류기 없는 지도([Ho & Salimans, 2022])를 사용하여 트레이닝함. 실제로, $$G$$는 조건 유무의 분포 둘 다 샘플의 10%에 대해 랜덤으로 $$c = \phi$$를 설정하고 학습하여, $$G(x_t, t, \phi)$$가 $$p(x_0)$$를 간략화하도록 함. 그 다음, $$G$$를 샘플링할 시 다음과 같이 $$s$$를 사용하여 두 가지 변형을 interpolation이나 extrapolation으로 다양성과 정확도 사이를 trade-off 할 수 있음.
    <br/><br/>
    ![Eq7](/assets/human-motion-diffusion-model/eq7.png)
    <br/><br/>
* **편집**<br/>
    * Diffusion inpainting을 모션 데이터에 적용하여, 시간 도메인에서의 모션 in-betweening과 공간 도메인에서의 신체 부분 편집을 가능하게 함.
    * 편집은 트레이닝 관여 없이 오직 샘플링 중에서만 수행됨.
    * 모션 시퀀스 인풋의 부분집합이 주어져 있을 때, 모델을 샘플링 중이면(그림 2 (우)), 각 반복 시 $$\hat{x}_0$$를 모션의 인풋 부분으로 덮어씀. 이것은 생성이 누락된 부분을 완성하면서, 원본 인풋에 일관성을 유지하도록 함.
    * 시간 설정에서, 모션 시퀀스의 접두사 및 접미사 프레임들이 인풋이 되어, 모션 in-betweening 문제를 해결함 ([Harvey et al., 2020]). 편집은 조건적으로나 무조건적으로($$c = \phi$$로 설정 시) 수행될 수 있음.
    * 공간 설정에서, 동일한 완성 기술을 사용함으로써 나머지는 그대로 유지하면서 조건 $$c$$에 따라 신체 부위를 재합성할 수 있음을 보임.

## Discussion

* 다양한 인간 모션 생성 task에 적합한 기법 MDM을 제안함. MDM은 분류기가 없는 비전형적 diffusion 모델로, transformer-인코더 backbone을 특징으로 하며, 노이즈보다는 신호를 예측함. 이는 트레이닝에 부담이 없는 경량 모델과 적용 가능한 geometric loss로부터 많은 것을 얻는 정확한 모델을 산출함. 실험에서 조건부 생성에서 우월함을 보이지만, 이 접근법은 구조 선택에 매우 예민하지 않음.
* Diffusion 접근법의 주목할 한계는 단일 결과에 약 1,000회의 정방향 통과를 요구하는 긴 추론 시간이라는 점임. 아무튼 본 모델은 작으므로, 이미지보다 작은 크기의 차원 order를 사용하여 1초 이내부터 단 1분 정도까지로 추론 시간을 이동되는데, 이것은 받아들일 수 있는 타협임. Diffusion 모델은 더 나은 계산 외에도 계속해서 진화함에 따라, 미래에는 더 좋은 컨트롤을 생성 과정에 통합하고 적용을 위한 옵션 범위를 훨씬 더 넓히는 방법에 관심을 가지게 될 것임.

[Human Motion Diffusion Model]: https://arxiv.org/abs/2209.14916
[https://guytevet.github.io/mdm-page/]: https://guytevet.github.io/mdm-page/