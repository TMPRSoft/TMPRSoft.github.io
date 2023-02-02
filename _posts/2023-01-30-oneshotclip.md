---
layout: post
title:  "[리뷰] (OneShotCLIP) One-Shot Adaptation of GAN in Just One CLIP"
date:   2023-01-31 05:07:00 +0900
categories: GAN
tags: [GAN, CLIP, Few-Shot Learning]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [One-Shot Adaptation of GAN in Just One CLIP]
<br/>
Authors: Gihyun Kwon and Jong Chul Ye
<br/>
Submitted on 17 Mar 2022

![Fig1a](/assets/oneshotclip/fig1a.png)
![Fig1b](/assets/oneshotclip/fig1b.png)
![Fig1c](/assets/oneshotclip/fig1c.png)

그림 1: 본 모델에서의 다양한 domain adaptation 결과. 본 모델은 대량의 데이터에서 pre-train 된 모델을 단일 shot의 타깃 이미지로 타깃 도메인으로 fine-tuning을 성공시킴. Experiment 절에서 더 많은 생성 샘플들을 찾을 수 있음.

## Abstract

???

## Introduction

Contribution은 다음과 같이 요약될 수 있음.

* Model adaptation에 더 적합한 reference를 찾는 데 latent 검색 단계에 CLIP을 사용하는 것을 최초로 제안함. 이전 연구에서는 시도된 적이 없음.
<br/>
* CLIP 임베딩 공간에서 패치 및 샘플 별 consistency 정규화 제안. 역시 GAN adaptation task에서 최초로 시도됨.
<br/>
* 본 기법은 질적 및 양적 결과 모두에서 다른 제한된 shot의 GAN adaptation 기법들을 능가함.
<br/>
* 본 기법은 다른 task에 쉽게 적용될 수 있도록 유연한 프레임워크를 갖고 있음. 실험 결과에서 attribute editing이나 text-guided adaptation 등 다양한 적용 사례를 보임.

## Methods

* **목표**<br/>
  기존 접근들을 확장하여, 개선된 정규화 및 트레이닝 전략을 찾을 것<br/>
  타깃 이미지 타입이나 pre-train 된 소스 도메인에 상관 없이, 그리고 과적합(overfitting) 및 과소적합(underfitting) 없이 모델을 강인하게 적응시킬 것
  
* **단계별 구성**<br/>
1. 이전 연구([Zhu et al. 2021])와 유사하게, latent code $$w_{ref}$$를 소스 도메인 생성기(generator) $$G_s$$의 잠재 공간(latent space)에서 찾아, 주어진 단일 shot 타깃 이미지 $$I_{trg}$$와 가장 유사한 레퍼런스 이미지 $$I_{ref}=G(w_{ref})$$를 생성할 수 있도록 함.
<br/>
  픽셀별 유사도를 사용하는 Mind the GAP([Zhu et al. 2021])과는 달리, CLIP 공간 유사도가 검색의 질을 상당히 개선시킴. 결과 이미지 $$I_{ref}$$는 이제 다음 단계인 모델 fine-tuning에서 $$I_{trg}$$에 맞출 레퍼런스 지점으로 사용됨.
2. Pre-train 된 생성기를 fine-tuning 하여 $$I_{trg}$$의 도메인 정보를 따르는 타깃 생성기 $$G_t$$를 생성함. 여기서 목표는 $$G_t$$를 이전 단계에서 얻은 타깃 이미지 $$I_{trg}$$와 레퍼런스 이미지 $$I_{ref}$$에 맞춰 $$G_t$$의 다양한 content attribute를 유지하도록 하는 것. 다시 이것을 CLIP 공간 안에서 정규화로 강화하여 소스와 타깃 생성기 사이의 semantic consistency를 강화함.

#### 단계 1: Clip-guided Latent Search

* 타깃 이미지와 가장 일관성 있는 이미지를 찾기 위해, 이전 적응 모델 Mind the Gap([Zhu et al. 2021])은 기존 StyleGAN inversion 기법인 II2S([Zhu et al. 2020])를 활용했음. 그러나, 추상 스케치와 FFHQ의 관계처럼 $$I_{trg}$$의 도메인과 소스 도메인 사이의 거리가 멀면, 기존 inversion 모델은 $$I_{trg}$$의 attribute를 재구성하는 데 실패함.

![Fig2](/assets/oneshotclip/fig2.png)

그림 2: (좌) 레퍼런스 이미지 $$I_{ref}$$를 찾기 위해 제안된 CLIP-guided latent 최적화. (우) 다양한 baseline과의 비교. 본 기법이 기존 기법들에 비해 타깃 이미지의 desired attribute를 가장 잘 담고 있음.

* 이러한 성능 저하의 원인을 오직 픽셀별 유사도에만 집중했다는 점으로 추측하여, 이를 바로잡기 위해 pre-train 된 CLIP 모델을 사용하여 생성된 $$I_{ref}$$를 $$I_{trg}$$의 semantic attribute를 따르도록 가이드할 것을 제안함. 더 구체적으로는, 그림 2(좌)처럼, $$I_{ref}$$와 $$I_{trg}$$ 사이에 픽셀별 loss를 사용하면서 두 이미지의 CLIP 공간 임베딩 사이의 cosine 거리를 줄이는 추가 loss를 포함시킴.
<br/>
  여기에 $$I_{ref}$$에 대한 증강(augmentation)을 도입하여 artifact를 회피함.

* 따라서 최적화 문제 식은 다음과 같음.
<br/>
![Eq1](/assets/oneshotclip/eq1.png)
<br/>
여기서 $$I_{ref}=G_s(w)$$이고, $$D_{CLIP}(\cdot,\cdot)$$은 CLIP 임베딩 사이의 cosine 거리이며, $$lpips$$는 perceptual loss, $$\bar{w}$$는 소스 생성기 $$G_s$$의 평균 스타일 코드임.
<br/>
비현실적인 이미지를 재구성하는 것을 회피하기 위해 $$w$$와 $$\bar{w}$$ 사이에 $$l_2$$ 정규화를 추가로 사용함.
<br/>
그리고 검색의 효율성을 개선하기 위해 $$w$$의 시작점을 랜덤보다는 $$\bar{w}$$로 지정함.
<br/>
최적화 단계 이후, 최종 latent code $$w$$를 레퍼런스 latent $$w_{ref}$$로 사용함.

#### 단계 2: Generative Model Fine-tuning

![Fig3](/assets/oneshotclip/fig3.png)

그림 3: CLIP 공간에서의 교차 도메인 semantic consistency. 소스 모델 $$G_s$$와 $$G_t$$로부터 같은 latent $$w$$로 이미지 생성, 그 다음 pre-train 된 CLIP 모델로 임베딩된 특징점 벡터를 얻고 CLIP 공간에서 두 벡터 사이의 cosine 유사도 스코어 계산. 두 도메인 ($$c^t, c^s$$) 사이의 유사도 스코어가 $$l_2$$ 회귀를 통해 결합됨. $$G_t$$의 텍스처를 타깃 $$I_{trg}$$에 매치시키도록 가이드하기 위해, 패치 판별기를 추가로 사용함.

* **Cross-domain semantic consistency**
<br/>
    * Pre-train 된 생성기의 가중치(weight)들을 fine-tuning 하여 타깃 생성기 $$G_t$$를 생성함. [Ojha et al. 2021]은 패치 판별기(discriminator) $$D_{patch}$$를 사용하여 $$G_t$$로 생성된 이미지가 타깃 이미지 $$I_{trg}$$의 텍스처를 갖도록 했음. 그러나 판별기만을 사용하면 과적합이 발생하기 쉬우므로, $$G_t$$가 $$G_s$$의 생성 다양성을 상속할 수 있도록 하는 추가 정규화가 필요함. 이를 해결하기 위해 이전 연구들은 소스와 타깃 생성기 사이의 특징점(feature)들의 분포를 매치시키는 시도를 하였으나, 단일 shot 조건에서는, 그런 생성기 특징점 분포 기반의 정규화는 과적합을 해결하지 못함.
<br/>
    * 단계 1과 비슷하게 이 이슈도 CLIP 공간 정규화가 해결함을 발견함. 구체적으로, 생성된 이미지의 semantic 정보를 고려하기 때문에 그림 3처럼 CLIP 공간에서 유사도 분포를 유지하는 것은 과적합을 예방하는 데 더 효과적임을 발견함. 

        > CLIP 인코더를 통해 각 이미지의 레이블을 출력하여 비교하는 것이므로 과적합 가능성을 줄이는 것으로 추측됨.

    * 특히 임의의 latent 변수들 $$[w_i]_0^N$$가 샘플링되면, 먼저 다음과 같이 $$G_s$$와 $$G_t$$에 대한 샘플별 유사도 스코어를 계산함. 여기서 $$n$$은 $$(i, j)$$의 재배열된 index이고 $$D_{CLIP}$$은 pre-train 된 CLIP 임베딩 공간에서의 cosine 유사도를 나타냄.
    
        ![Eq2](/assets/oneshotclip/eq2.png)
    
    * 계산된 유사도 스코어로, $$G_s$$가 $$G_t$$ 사이의 유사도 loss를 계산할 수 있음. 분포 매칭을 위해 softmax와 KL 발산(divergence)를 사용하는 기존 연구([Ojha et al. 2021])와는 달리, 단순히 유사도 스코어 사이에 $$l_2$$ 거리를 사용했는데, 이는 CLIP 공간에서 KL 발산을 사용하면 트레이닝의 안정성을 떨어뜨렸기 때문임.
    * $$G_s$$와 $$G_t$$ 사이의 semantic consistency에 대한 loss 함수
    
        ![Eq3](/assets/oneshotclip/eq3.png)
        
![Fig4](/assets/oneshotclip/fig4.png)

그림 4: (a) CLIP 공간에서 패치 별 semantic consistency. 생성된 이미지에서 패치를 crop 하고 패치를 pre-train 된 CLIP 모델을 사용하여 임베딩함. 그 다음 임베딩된 벡터로 비교 학습을 진행함. (b) Reference target alignment. Reference latent $$w_{ref}$$를 사용하는 경우, 생성된 이미지는 $$I_{trg}$$에 매치되어야 함.

> 주어진 벡터 $$v$$에 가장 매치가 되는 곳은 벡터 $$v^+$$이므로 가까워지도록 학습되고, 나머지 벡터들 $$v_i^-$$은 멀어지도록 학습됨.

* **Patch-wise semantic consistency**
<br/>
    * 새로운 loss $$L_{con}$$를 사용하여 과적합 문제를 회피할 수 있으나, 이 loss는 생성된 이미지들의 지역 특징점들이 무시되도록 하는 이미지들의 샘플 별 semantic에 대한 정규화이다. 따라서, 소스와 타깃 생성기 사이의 세부 내용의 일관성 개선을 위해 새로운 패치 별 consistency loss를 제안함.
    * 두 도메인 사이의 지역 attribute을 보존하기 위해, CUT([Park et al. 2020])의 패치 별 contrastive loss(Patch-NCE)의 아이디어에서부터 시작함. CUT는 생성기의 임베딩된 특징점들에 비교 학습(constrastive learning)을 도입했음. 생성기 특징점들을 직접 사용하기보단, CUT에서 추가적인 MLP 헤더 네트워크가 특징점들을 또 다른 공간으로 임베딩하는 데 사용됨.
    * 그러나 본 프레임워크에서는 이것을 직접 적용하면, header와 pre-train 된 생성기 사이의 불균형으로 인해 트레이닝이 실패함. 이 문제에 대해 pre-train 된 CLIP 모델을 패치 별 임베딩 네트워크로 사용하는 것을 제안함.
    * 구체적으로 $$G_s$$와 $$G_t$$로 생성된 이미지로부터 랜덤 위치에서 패치들을 crop 한 후, 그림 4(a)와 같이 이미지 패치들을 CLIP 인코더로 임베딩함. 그 다음, 같은 위치에서 crop 한 양성 패치들 사이의 거리를 좁히고, 다른 위치에서 crop 한 음성 패치들을 멀리함. 그러니까 임의의 위치를 $$s_0$$이라 하고, 생성기 $$G_s$$와 $$G_t$$의 아웃풋에서 crop 한 패치를 각각 $$[G_s(w)]_{s_0}$$와 $$[G_t(w)]_{s_0}$$이라 하자. 그리고 다른 위치들을 $$N$$개 더 지정하면, 소스 도메인에서 $$N$$개의 패치 $$[G_s(w)]_{s_i}$$ $$(i \in {1, \cdots, N})$$를 얻을 수 있음.
    * 그럼 패치 별 loss는 다음과 같이 계산됨. 여기서 $$v=E([G_t(w)]_{s_0})$$, $$v^+=E([G_s(w)]_{s_0})$$는 양의 벡터, $$v^-=E([G_s(w)]_{s_i})$$는 음의 벡터임. $$E$$는 이미지 패치에 대해 pre-train 된 CLIP 인코터이고, $$\cdot$$은 cosine 유사도임.
    
        ![Eq4](/assets/oneshotclip/eq4.png)

* **Reference Target Alignment**
<br/>
    * 이전 파트에서 제안된 loss들은 임의의 latent를 샘플링할 때 과적합을 예방하는 역할을 함. 단계 1에서 찾을 수 있는 reference latent $$w_{ref}$$를 통해 이미지를 생성하기 위한 추가적인 loss와 번갈아 나와야 함.
    * 단계 1에서 얻은 $$I_{ref}$$는 $$I_{trg}$$에 가장 유사한 attribute를 가지고 있고 소스 도메인을 표현하는 이미지이므로, 도메인 적응 아웃풋 $$G_t(w_{ref})$$는 $$I_{trg}$$에 매치되어야 함.
    * 따라서 그림 4(b)와 같이, [Zhu et al. 2021]에서 제안된 대로 픽셀과 perceptual 관점 둘 다에서 이미지를 매치시킴. 더 나아가, $$G_t(w_{ref})$$가 $$I_{trg}$$에 더 가까워지도록 전역 판별기 $$D_{glob}$$로 가이드함.

* **전반적인 트레이닝**
<br/>
    * 본 네트워크는 두 가지 loss를 번갈아서 최소화하도록 트레이닝됨. 첫째, reference latent $$w_{ref}$$를 사용할 때, loss는 아래와 같이 정의됨.

        ![Eq5](/assets/oneshotclip/eq5.png)
        
    * 여기서 StyleGAN2 적대적 loss는 $$L_{adv}^g(G,D_{glob}) = D_{glob}(G(w_{ref})) - D_{glob}(I_{trg})$$로 정의됨. 전역 판별기에 대해, pre-train 된 StyleGAN2 판별기를 fine-tuning 함.
    * 둘째, 임의의 latent $$w$$가 샘플링되면, loss는 아래와 같음.
    
        ![Eq6](/assets/oneshotclip/eq6.png)

    * 이 경우, $$L_{adv}^p(G,D) = D_{patch}(G(w)) - D_{patch}(I_{trg})$$로 정의됨. $$D_{patch}$$에 대해서는, [Ojha et al. 2021]에서 제안된 네트워크를 사용했는데, 여기서 $$D_{patch}$$는 $$D_{glob}$$의 부분집합임. 구체적으로, $$D_{glob}$$으로부터 중간 특징점을 추출한 다음, 몇 개의 conv. 레이어를 통해 특징점을 매핑하여 $$D_{patch}$$의 logit을 얻음.

[One-Shot Adaptation of GAN in Just One CLIP]: https://arxiv.org/abs/2203.09301
[Zhu et al. 2021]: https://arxiv.org/abs/2110.08398
[Zhu et al. 2020]: https://arxiv.org/abs/2012.09036
[Ojha et al. 2021]: https://openaccess.thecvf.com/content/CVPR2021/html/Ojha_Few-Shot_Image_Generation_via_Cross-Domain_Correspondence_CVPR_2021_paper.html
[Park et al. 2020]: https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19