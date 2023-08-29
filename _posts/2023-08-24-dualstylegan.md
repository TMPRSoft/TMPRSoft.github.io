---
layout: post
title:  "[리뷰] (DualStyleGAN) Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer"
date:   2023-08-24 12:12:00 +0900
categories: [GAN]
tags: [GAN, StyleGAN, Image Generation]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer]
<br/>
Authors: S. Yang, L. Jiang, Z. Liu, and C. C. Loy
<br/>
Submitted in CVPR 2022
<br/>
Project Page: [Github]

<div id="Fig1"></div>
![Fig1](/assets/dualstylegan/fig1.png)

그림 1. 예시 기반 (exemplar-based) 고해상도 (high-resolution) ($$1024 \times 1024$$) 초상화 스타일 변환(style transfer)을 위한 새로운 DualStyleGAN을 제안함. 실제 얼굴 (a)로부터 생성된 그림 초상화 (b)-(d)는 성공적으로 각 좌하 코너의 예시의 색과 스타일을 따라함. DualStyleGAN은 두 가지 스타일 경로가 특징임: 각각 콘텐츠와 스타일에 걸친 유연한 제어를 위한 내부 스타일 경로와 외부 스타일 경로. (e) 임의의 내부와 외부 스타일 코드로 생성된 카툰 얼굴. (f) 고정된 내부 스타일의 다양한 외부 스타일, (g) 고정된 외부 스타일의 다양한 내부 스타일, (h) 다양한 양쪽 스타일로 생성된 샘플들.

## Abstract

* StyleGAN에 대한 최근의 연구는 제한된 데이터로 전송 학습을 통해 예술적 초상화 생성에 대한 높은 성능을 보여줌.
* 본 논문에서는 원본 얼굴 도메인과 확장된 초상화 영역의 이중 스타일을 유연하게 제어하는 새로운 **DualStyleGAN**을 도입하여, 보다 도전적인 예시 기반 고해상도 초상화 스타일 전이를 탐구함.
* StyleGAN과는 달리, DualStyleGAN은 **고유한 스타일 경로**와 새로운 **외부 스타일 경로**로 초상화의 내용과 스타일을 각각 특성화함으로써 자연스러운 스타일 전이 방법을 제공함.
* 섬세하게 설계된 외부 스타일 경로를 통해 모델은 스타일 예제를 정확하게 붙여넣기 위해 색상과 복잡한 구조 스타일을 모두 계층적으로 변조할 수 있음.
* 또한 네트워크 아키텍처에서 위와 같은 수정 사항에도 불구하고 모델의 생성 공간을 타깃 도메인으로 원활하게 변환하기 위해 새로운 점진적 미세 조정 체계가 도입됨.
* 실험은 고품질 초상화 스타일 전이와 유연한 스타일 제어에서 SotA 기법에 비해 DualStyleGAN의 우수성을 입증함.

<div id="Sec1"></div>
## Introduction

* 이미지 스타일 전환과 image-to-image (I2I) 변환 기반의 자동 초상화 스타일 전환은 광범위하게 연구되어 왔음.
* 최근에는, SotA 얼굴 생성기인 StyleGAN은, 전이 학습(transfer learning)을 통한 고해상도 초상화 생성에 대해 매우 유망함.
    * 특히, StyleGAN은 생성 공간을 얼굴 도메인에서 초상화 도메인으로 변환하는 데 효과적으로 fine-tuning 될 수 있고, 보통 단 수백 장의 초상화 이미지와 수 시간의 트레이닝 시간만을 요구함.
    * 이미지 스타일 전환 및 I2I 변환 모델들에 비해 퀄리티, 이미지 해상도, 데이터 요구량, 효율성 면에서 훌륭한 우월성을 보임.
<br/><br/>
* 위의 전략은, 효과적이지만, 예제 기반 스타일 전환을 수행할 수 없는 분포의 전반적인 변환만 학습함.
* 고정 캐리커처 스타일을 생성하기 위해 전환된 StyleGAN의 경우, 웃는 얼굴이 캐리커처 도메인에서 가장 가까운 것, 즉 입이 과장된 초상화에 크게 매핑됨. 사용자는 [그림 1](#Fig1)(c)와 같이 선호하는 예술 작품을 모방하기 위해 얼굴을 축소할 수단이 없음.
* StyleGAN이 latent swapping에 의해 고유한 예제 기반 단일 도메인 스타일 믹싱을 제공하지만, 이러한 단일 도메인 지향 작업은 소스 도메인과 타깃 도메인을 포함하는 스타일 전환에 대해 반직관적이고 능력이 부족함. 이는 이러한 두 도메인 간의 잘못된 정렬이 특히 도메인별 구조에 대해 스타일 혼합 중 원하지 않는 아티팩트를 초래할 수 있기 때문임.
* 그러나 중요한 것은 전문적인 모방 이미지가 만화의 추상화와 캐리커처의 변형과 같은 얼굴 구조를 아티스트가 다루는 방식을 모방해야 한다는 것임.
<br/><br/>
* 위의 새로운 모형으로 제안된 DualStyleGAN은 고품질의 고해상도 모방 이미지를 제공하며 [그림 1](#Fig1)과 같이 색상 스타일과 복잡한 구조 스타일 모두에 대해 유연하고 다양한 제어를 제공함.
* 요약하면, 본 기여는 세 가지임:
    * 고품질의 다양한 초상화 생성에서 SotA 기법보다 우수한 성능을 달성하는 수백 개의 스타일 예시만 필요한 예시 기반 고해상도 초상화 스타일 전환을 위해 내부 및 외부 스타일을 특성화하고 제어하기 위한 새로운 DualStyleGAN을 제안함.
    * Fine-tuning을 통해 외부 도메인에서 스타일 특징을 도입하고 색상과 구조 측면에서 계층적 스타일 조작을 제공하기 위해 원칙적인 외부 스타일 경로를 설계함.
    * 아키텍처 수정을 통해 네트워크를 통한 강력한 전이 학습을 위한 새로운 점진적 fine-tuning 계획을 제안함.

<div id="Sec3"></div>
## Portrait Style Transfer via DualStyleGAN

<div id="Fig2"></div>
![Fig2](/assets/dualstylegan/fig2.png)

그림 2. StyleGAN에 대한 무조건부 fine-tuning과 DualStyleGAN에 대한 조건부 fine-tuning의 비교.

* 목표는 사전 트레이닝된 StyleGAN을 기반으로 DualStyleGAN을 구축하는 것이며, 이는 새로운 도메인으로 전환되어 원래 도메인과 확장된 도메인의 스타일을 모두 특성화할 수 있음.
* 무조건적인 미세 조정은 [그림 2](#Fig2)와 같이 StyleGAN 생성 공간을 전체적으로 변환하여 캡처된 스타일의 다양성을 잃게 함.
* 본 핵심 아이디어는 다양한 스타일을 학습하기 위해 유효한 지도(supervision)를 모색하는 것과([Sec. 3.1](#Sec3.1)), 두 가지 개별 스타일 경로로 두 종류의 스타일을 명시적으로 모델링하는 것임([Sec. 3.2](#Sec3.2)).
* 그리고 강력한 조건부 fine-tuning을 위한 원칙 있는 점진적 전략으로 DualStyleGAN을 트레이닝함([Sec. 3.3](#Sec3.3)).

<div id="Sec3.1"></div>
### Facial Destylization

<div id="Fig3"></div>
![Fig3](/assets/dualstylegan/fig3.png)

그림 3. Facial destylization의 묘사. 각 스테이지에서 (a)의 destylize 된 결과들은 (b)-(d)에서 과장된 눈이 점차 현실적이 되면서 순차적으로 보여짐. (e)-(g): 정규화는 얼굴과 무관한 녹색 장난감에 대한 오버피팅을 예방함. (h)-(j): $$\mathbf{z}_e^+$$는 복잡한 만화 얼굴을 맞추는 데 좋은 초기값으로 동작함.

* Facial destylization은 지도(supervision)로서 고정된 얼굴-초상화 쌍을 형성하기 위해 초상화에서 현실적인 얼굴을 복구하는 것을 목표로 함.
    * 타깃 도메인의 초상화가 주어질 때, 얼굴 도메인에서 그것들의 합당한 대응물을 찾고자 함.
    * 두 도메인은 외관 차이가 클 수 있기 때문에, 얼굴 현실성과 초상화에 대한 충실성(fidelity) 사이의 균형을 맞추는 것은 사소한 과제가 아님.
    * 이 문제를 해결하기 위해, 초상화의 현실성을 점진적으로 높이기 위한 다단계 destylization 기법을 제안함.
<br/><br/>
* **Stage I: Latent initialization**
    * 초상화 $$S$$는 먼저 StyleGAN 잠재 공간(latent space)로 인코더 $$E$$에 의해 임베딩됨. 여기서, pSp 인코더를 사용하고 FFHQ 얼굴들을 $$\mathcal{Z}+$$ 공간으로 임베딩하도록 수정하는데, 이것은 [34]에 제안된 것처럼 원래 $$\mathcal{W}+$$ 공간보다 얼굴과 무관한 배경 디테일 및 왜곡된 모양에 더 강인함.
    * 재구축된 얼굴 $$g \left( \mathbf{z}_e^+ \right)$$의 예시가 [그림 3(b)](#Fig3)에 보이는데, $$g$$는 FFHQ에서 사전  트레이닝된 StyleGAN이고 $$\mathbf{z}_e^+ = E \left( S \right) \in \mathbb{R}^{18 \times 512}$$는 잠재 코드(latent code)임. $$E$$가 실제 얼굴로 트레이닝되었음에도, $$E \left( S \right) $$는 초상화 $$S$$의 색과 구조를 잘 캡처함.
<br/><br/>
* **Stage II: Latent optimization**
    * [29]에서, 얼굴 이미지는 이 이미지를 재구축하기 위해 $$g$$의 잠재 코드를 최적화하고 이 코드를 fine-tuning 된 모델 $$g'$$에 적용함으로써 스타일화됨.
    * 새로운 정규화 항으로 $$S$$를 재구축하기 위해 $$g'$$의 잠재 $$\mathbf{z}^+$$를 최적화하는 역스텝을 밟고, 그 결과 $$\hat{\mathbf{z}}_e^+$$를 $$g$$에 적용하여 destylize 된 버전을 얻음.
    <br/><br/>
    <div id="Eq1"></div>
    ![Eq1](/assets/dualstylegan/eq1.png)
    <br/><br/>
    * 여기서 $$\mathcal{L}_{\mathrm{perc}}$$는 perceptual loss이고, $$\mathcal{L}_{\mathrm{ID}}$$는 얼굴의 identity를 보존하기 위한 identity loss이며 $$\sigma \left( \mathbf{z}^+ \right)$$는 $$\mathbf{z}^+$$ 내 18개의 서로 다른 512차원 벡터의 표준오차(standard error)이며, $$\lambda_{\mathrm{ID}} = 0.1$$임.
    * [1]과는 달리, 정규화 항을 설계하여 [그림 3(f)(g)](#Fig3)에서처럼 오버피팅을 피하기 위해 잘 정의된 $$\mathcal{Z}$$ 공간으로 $$\hat{\mathbf{z}}_e^+$$를 끌어당기고, 평균 잠재 코드보다는 $$\mathbf{z}_e^+$$를 사용하여 최적화 전에 $$\mathbf{z}^+$$를 초기화하는데, 이는 [그림 3(i)(j)](#Fig3)에서처럼 얼굴 구조를 정확히 맞추는 데 도움을 줌.
<br/><br/>
* **Stage III: Image embedding**
    * 마지막으로, $$g \left( \hat{\mathbf{z}}_e^+ \right)$$를 $$\mathbf{z}_i^+ = E \left( g \left( \hat{\mathbf{z}}_e^+ \right) \right)$$로 임베딩하는데, 이는 비현실적인 얼굴 디테일을 더 제거함.
    * 결과 $$g \left( \mathbf{z}_i^+ \right)$$는 합당한 얼굴 구조를 가지고, $$S$$를 흉내내기 위한 얼굴 구조를 변형하고 추상화하는 방법에 대한 유효한 지도를 제공함.

<div id="Sec3.2"></div>
### DualStyleGAN

<div id="Fig4"></div>
![Fig4](/assets/dualstylegan/fig4.png)

그림 4. DualStyleGAN의 네트워크 세부. 간단화를 위해, StyleGAN의 학습되는 가중치, 바이어스, 노이즈는 제거됨.

<div id="Fig5"></div>
![Fig5](/assets/dualstylegan/fig5.png)

그림 5. ResBlock이 가장 Toonify를 잘 시뮬레이션함.

* [그림 4](#Fig4)는 DualStyleGAN $$G$$의 네트워크 세부를 보여 줌.
* 내부 스타일 경로와 생성 네트워크는 표준 StyleGAN를 형성하고 fine-tuning 동안 고정으로 유지됨.
    * 내부 스타일 경로는 단위 가우시안 노이즈 $$\mathbf{z} \in \mathbb{R}^{1 \times 512}$$나 초상화 $$\mathbf{z}_i^+$$, 또는 $$E$$로 임베딩된 실제 얼굴의 $$\mathbf{z}^+$$의 내부 스타일 코드를 받아들임.
* 외부 스타일 경로는 간단히 초상화의 $$\mathbf{z}_e^+$$를 외부 스타일 코드로 사용하는데, 이는 머리카락 색과 얼굴 모양 같은 의미 있는 의미론적 신호를 캡처함([그림 3(b)](#Fig3)).
    * 외부 스타일 코드는 단위 가우시안 노이즈를 외부 스타일 분포에 매핑함으로써 샘플링 네트워크 $$N$$을 통해서도 샘플링될 수 있음.
* 형식적으로, 얼굴 이미지 $$I$$와 초상화 이미지 $$S$$가 주어질 때, 예시 기반 스타일 전환은 $$G \left( E\left( I \right), E\left( S \right), \mathbf{w} \right)$$로 수행되고. 여기서 $$\mathbf{w} \in \mathbb{R}^{18}$$은 두 경로의 유연한 스타일 혼합에 대한 가중치 벡터이고, 디폴트로 1로 설정됨.
* 초상화 생성은 $$G \left( \mathbf{z}_1, N\left( \mathbf{z}_2 \right), \mathbf{w} \right)$$으로 이루어짐.
* $$\mathbf{w} = \mathbf{0}$$일 때, $$G$$는 얼굴 형성을 위한 표준 $$g$$로 저하시킴. 즉, $$G\left( \mathbf{z}, \cdot, \mathbf{0} \right) \sim g \left( \mathbf{z} \right)$$.
<br/><br/>
* StyleGAN은 계층적 스타일 컨트롤을 제공하는데, 미세한 해상도 (fine-resolution) 및 거친 해상도 (coarse-resolution) 레이어들이 각각 하위 레벨 색 스타일과 상위 레벨 모양 스타일을 모델링하고 외부 스타일 경로의 본 설계에 영감을 줌.
* **Color control**
    * 미세한 해상도 레이어들(8~18)에서, 외부 스타일 경로는 StyleGAN과 같은 전략을 사용함.
    * 특히, $$\mathbf{z}_e^+$$는 매핑 네트워크 $$f$$, 색 변형 블록들 $$T_c$$, 아핀 (affine) 변형 블록들 $$A$$를 통과함.
    * 결과 스타일 바이어스는 최종 AdaIN에 대한 가중치 $$\mathrm{w}$$의 내부 스타일 경로로부터의 스타일 바이어스와 결합됨.
    * $$g$$와는 달리, 완전 연결 (fully connected) 레이어로 구성된 트레이닝 가능한 $$T_c$$가 도메인 특성의 색을 특징화하기 위해 추가됨.
* **Structure control**
    * 거친 해상도 레이어들(1~7)에서, 구조 스타일을 조정하기 위해 변조 잔차 블록(modulative residual blocks, ModRes)을 제안하고 도메인 특성의 구조 스타일을 특징화하기 위해 구조 변형 블록 $$T_s$$를 추가함.
    * ModRes에는 fine-tuning 중에 convolution 레이어의 변화를 시뮬레이션하는 ResBlock과 스타일 조건에 대한 AdaIN 블록이 포함되어 있음.
    * 제안된 ModRes의 동기를 이해하기 위해, 아래에서 StyleGAN에 대한 fine-tuning 동작에 대한 몇 가지 실험적 분석을 제공함.
* **Simulating fine-tuning behavior**
    * Toonification의 성공은 fine-tuning 전후 모델의 의미론적 정렬에 달려 있음. 즉, 두 모델이 잠재 공간을 공유하고 밀접하게 관련된 convolution 기능을 가지고 있다는 것을 의미함.
    * 또한 이러한 기능의 차이가 원래 기능과 밀접하게 관련되어 있음. 게다가, StyleGAN의 모든 하위 모듈 중에서 convolution 레이어는 fine-tuning 중에 가장 많이 바뀜.
    * 따라서, 다른 하위 모듈은 모두 고정시키고 fine-tuning 중에 convolution 가중치 행렬의 변화를 시뮬레이션하기 위해 convolution 특징점들에 걸친 변화만 학습하는 것이 가능함.
    * StyleGAN에서, 심층 특징점들에 대한 공통 조정은 각각 AdaIN, Diagonal Attention (DAT) 및 ResBlock에 해당하는 채널별, 공간별 및 요소별 변조를 포함함.
    * Toy 실험을 수행하여 채널([그림 5(d)](#Fig5) 또는 공간([그림 5(e)](#Fig5) 차원만의 변조로는 fine-tuning 동작을 근사화하기에 충분하지 않다는 것을 발견함.
    * ResBlocks는 전체 StyleGAN([그림 5(b)](#Fig5))을 fine-tuning 하여 가장 유사한 결과([그림 5(c)](#Fig5))를 달성함.
    * 따라서 residual block을 선택하고 residual path의 convolution 레이어에 AdaIN을 적용하여 외부 스타일 조건을 제공함.
* **Summary**
    * DualStyleGAN은 매우 간단하면서도 효과적임.
        * **1) 복잡한 스타일의 계층적 모델링:** 색상과 복잡한 구조 스타일 모두에 대한 계층적 모델링을 제공함.
        * **2) 유연한 스타일 조작:** 가중치를 가진 두 도메인 간의 유연한 스타일 혼합을 지원함.
        * **3) Mode collapse 완화:** Fine-tuning은 mode collapse를 피하기 위해 원래의 다양한 얼굴 특징을 그대로 유지하면서 외부 스타일 경로만 트레이닝시킴.
        * **4) 구조 보존:** 변조 residual block의 추가 특성은 강인한 콘텐츠 loss로 이어지는데, 이는 [3.3절](#Sec3.3)에서 자세히 다룸.

<div id="Sec3.3"></div>
### Progressive Fine-Tuning

<div id="Fig6"></div>
![Fig6](/assets/dualstylegan/fig6.png)

그림 6. 점진적 fine-tuning의 묘사. (a) DualStyleGAN은 상승하는 난이도의 스타일 전환으로 동작함. (b) 각 단계 후 DualStyleGAN의 성능.

* DualStyleGAN의 생성 공간을 타깃 도메인으로 원활하게 변환하기 위한 점진적 fine-tuning 스킴을 제안함.
* 이 스킴은 [그림 6(a)](#Fig6)와 같이 세 단계로 점차 task 난이도를 증가시키기 위해 커리큘럼 학습의 아이디어를 차용함.
* **단계 I: 소스 도메인에서의 색상 전달**
    * DualStyleGAN은 이 단계에서 소스 도메인 내에서 색상 전달을 담당함.
    * 외부 스타일 경로의 설계 덕분에, 그것은 순수하게 특정 모델 초기화에 의해 달성될 수 있음.
    * 구체적으로, 변조 residual 블록의 convolution 필터는 무시될 수 있는 특징점들을 생산하기 위해 0에 가까운 값으로 설정되고 색 변형 블록들 내 완전 연결 레이어들이 identity 행렬로 초기화되는데, 이는 입력 잠재 코드에 대한 변경이 없음을 의미함.
    * 이를 위해, DualStyleGAN은 StyleGAN의 표준 스타일 혼합 연산을 실행하며, 여기서 미세한 해상도와 거친 해상도 레이어는 각각 내부 및 외부 스타일 경로의 잠재 코드를 사용함.
    * [그림 6(b)](#Fig6)와 같이, 초기화된 DualStyleGAN은 사전 트레이닝된 StyleGAN의 생성 공간에 여전히 있는 그럴듯한 인간 얼굴을 생성하여 다음 단계에서 원활한 fine-tuning이 가능하게 함.
* **단계 II: 소스 도메인에서의 구조 전송**
    * 이 단계는 소스 도메인에서 DualStyleGAN을 fine-tuning 하여 중간 레벨의 스타일을 캡처하고 전송하기 위해 외부 스타일 경로를 완전히 트레이닝시키는 것을 목표로 함.
    * 중간 레벨에서 StyleGAN의 스타일 혼합은 메이크업과 같은 소규모 스타일 전송을 포함하므로 DualStyleGAN에게 효과적인 지도를 제공함.
    * 단계 II에서 우리는 랜덤 잠재 코드 $$\mathbf{z}_1$$과 $$\mathbf{z}_2$$를 그리고 $$G \left( \mathbf{z}_1 , \tilde{\mathbf{z}}_2 , \mathbf{1} \right)$$를 perceptual loss가 있는 스타일 혼합 타깃 $$g \left( \mathbf{z}_l^+ \right)$$에 근사하고자 하며, 여기서 $$\tilde{\mathbf{z}}_2$$는 $$\{ \mathbf{z}_2 , E \left( g \left( \mathbf{z}_2 \right) \right) \}$$에서 샘플링되고, $$l$$은 스타일 혼합이 발생하는 레이어이고 $$\mathbf{z}_l^+ \in \mathcal{Z}+$$는 l 벡터 $$\mathbf{z}_1$$과 $$\left( 18 - l \right)$$ 벡터 $$\mathbf{z}_2$$의 결합임.
    * 다음과 같은 objective를 가진 fine-tuning 동안 $$l$$을 7에서 5로 점진적으로 감소시킴:
    <br/><br/>
    <div id="Eq2"></div>
    ![Eq2](/assets/dualstylegan/eq2.png)
    <br/><br/>
    여기서 $$\mathcal{L}_\mathrm{adv}$$는 StyleGAN 적대적 손실임.
    * $$l$$을 감소시킴으로써, $$g \left( \mathbf{z}_l^+ \right)$$는 $$\tilde{\mathbf{z}}_2$$로부터 더 많은 구조적 스타일을 가지게 될 것임.
    * 그래서, 외부 스타일 경로는 색에 비해 더 많은 구조적 스타일을 캡처하고 전송하기 위해 학습하게 될 것임.
* **단계 III: 타깃 도메인에서 스타일 전송**
    * 마지막으로, 타깃 도메인에서 DualStyleGAN을 fine-tuning 함. 예시 초상화 $$S$$의 스타일 코드 $$\mathbf{z}_i^+$$와 $$\mathbf{z}_e^+$$가 $$\mathcal{L}_{\mathrm{perc}}\left( G \left( \mathbf{z}_i^+ , \mathbf{z}_e^+ , \mathrm{1} \right) , S \right)$$로 $$S$$를 재구축하도록 함.
    * 표준 예시 기반 스타일 전송 패러다임에서처럼, 랜덤 내부 스타일 코드 $$\mathrm{z}$$에 대해, 다음 스타일 loss를 적용하는데,
    <br/><br/>
    <div id="Eq2.5"></div>
    ![Eq2.5](/assets/dualstylegan/eq2.5.png)
    <br/><br/>
    여기서 $$\mathcal{L}_\mathrm{CX}$$는 contextual loss이고 $$\mathcal{L}_\mathrm{FM}$$은 특징점 매칭 loss이며, 이렇게 $$G \left( \mathbf{z} , \mathbf{z}_e^+ , \mathrm{1} \right)$$를 $$S$$에 매치시킴.
    * 콘텐츠 loss에 대해, identity loss와 ModRes의 가중치 행렬의 $$L_2$$ 정규화를 사용함.
    <br/><br/>
    <div id="Eq3"></div>
    ![Eq3](/assets/dualstylegan/eq3.png)
    <br/><br/>
    * 단계 I에서의 초기화와 유사하게, 가중치 행렬의 정규화는 residual 특징점들을 0에 가깝게 만드는데, 이는 원래 내부 얼굴 구조를 보존하고 오버피팅을 예방함.
    * 전체 objective는 다음과 같은 형태를 가짐.
    <br/><br/>
    <div id="Eq4"></div>
    ![Eq4](/assets/dualstylegan/eq4.png)
    <br/><br/>

<div id="Sec3.4"></div>
### Latent Optimization and Sampling

<div id="Fig7"></div>
![Fig7](/assets/dualstylegan/fig7.png)

그림 7. 외부 스타일 코드를 최적화하여 색을 정제함.

* **Latent optimization**
    * 극도로 다양한 스타일을 완벽하게 포착하기는 어려움.
    * 이 문제를 해결하기 위해, DualStyleGAN을 고정하고 각 외부 스타일 코드를 ground truth $$S$$에 맞게 최적화함.
    * 최적화는 잠재 공간에 이미지를 임베딩하는 프로세스를 따르고 [Eq. (4)](#Eq4)에서 perceptual loss와 contextual loss를 최소화함.
    * [그림 7](#Fig7)과 같이, 잠재 최적화에 의해 색이 잘 정제됨.
* **Latent sampling**
    * 랜덤 외부 스타일을 샘플링하기 위해, maximum likelihood criterion을 사용하여 단위 가우시안 노이즈를 최적화된 외부 스타일 코드의 분포에 매핑하는 샘플링 네트워크 $$N$$을 트레이닝함. 자세한 내용은 [9] 참조.
    * 구조들($$\mathbf{z}_e^+$$의 처음 7개의 행)과 색들($$\mathbf{z}_e^+$$의 마지막 11개의 행)은 DualStyleGAN에서 잘 분리되었기 때문에, 이 두 부분을 개별적으로 처리하는데, 즉, 구조 코드와 색 코드는 $$N$$에서 독립적으로 샘플링되고 결합되어 완전한 외부 스타일 코드를 형성함.

<div id="Sec4"></div>
## Experiments

* **Datasets**
    * 목표는 사용자들에게 선호하는 미술 작품들을 수집하여 DualStyleGAN이 모방하도록 하는 것임.
    * 데이터셋을 쉬운 수집을 위해 수백 장의 이미지로 한정하고자 함. 따라서, 세 가지 데이터셋을 카툰, 캐리커처, 애니메이션의 인기 있는 스타일로 선택함.
    * 카툰 데이터셋은 317장의 이미지를 사용함. WebCaricature로부터 199장의 이미지와 Danbooru 초상화로부터 174장의 이미지를 사용하여 각각 캐리커처와 애니메이션 데이터셋을 구축함.
    * 외부 및 내부 스타일에 대해 각각 같은 데이터셋과 CelebA-HQ에서 테스트함.
<br/><br/>
* **Implementation details**
    * 점진적인 fine-tuning은 8개의 NVIDIA Tesla V100 GPU를 사용하고 GPU 당 배치 사이즈를 4로 설정함.
    * 단계 II는 $$\lambda_{\mathrm{adv}}=0.1$$, $$\lambda_{\mathrm{perc}}=0.5$$를 사용하고, $$l = 7, 6, 5$$에서 각각 300, 300, 3000회 반복 트레이닝하며, 약 0.5시간이 소요됨.
    * 단계 III는 $$\lambda_{\mathrm{adv}}=1$$, $$\lambda_{\mathrm{perc}}=1$$, $$\lambda_{\mathrm{CX}}=0.25$$, $$\lambda_{\mathrm{FM}}=0.25$$를 설정하고, 카툰, 캐리커처, 애니메이션에서 각각 $$\left( \lambda_{\mathrm{ID}} , \lambda_{\mathrm{reg}} \right)$$를 $$\left( 1, 0.015 \right) , \left( 4, 0.005 \right) , \left( 1, 0.02 \right)$$로 설정하고 1400, 1000, 2100회 반복 트레이닝함.
    * 트레이닝은 평균 약 0.75시간이 소요됨.
    * Destylization ([3.1절](#Sec3.1)), 잠재 최적화와 트레이닝 샘플링 네트워크([3.4절](#Sec.3.4))는 1개의 GPU를 사용하고 각각 약 5, 1, 0.13시간이 소요됨.
    * 테스트는 이미지 당 약 0.13초가 소요됨.
    * 간단화를 위해, $$\left[n_1 * v_1 , n_2 * v_2 , \cdots \right]$$를 사용하여 벡터 $$\mathbf{w}$$ 내 처음 $$n_1$$개의 가중치가 $$v_1$$의 값으로 설정되고, 다음 $$n_2$$개의 가중치가 $$v_2$$의 값으로 설정됨을 나타내기로 함.
    * $$\mathbf{w}_s$$와 $$\mathbf{w}_c$$는 구조 가중치 벡터($$\mathbf{w}$$의 처음 7개의 가중치)와 색 가중치 벡터(마지막 11개의 가중치)를 각각 나타냄.
    * 디폴트로, 트레이닝에서 $$\mathbf{w}$$를 $$\mathbf{1}$$로 설정하고 테스트에서 $$\mathbf{w}_c$$를 $$\mathbf{1}$$로, 카툰, 캐리커처, 애니메이션에 대해 $$\mathbf{w}_s$$를 각각 $$\mathbf{0.75}$$, $$\mathbf{1}$$, $$\left[ 4 * 0 , 3 * 0.75 \right]$$로 설정함.

<div id="Sec4.1"></div>
### Comparison with State-of-the-Art Methods

<div id="Fig8"></div>
![Fig8](/assets/dualstylegan/fig8.png)

그림 8. 예시 기반 초상화 스타일 전송에 대한 시각적 비교.

<div id="Table1"></div>
![Table1](/assets/dualstylegan/table1.png)

표 1. 사용자 선호 점수. 최고 점수는 볼드체로 표시함.

<div id="Fig9"></div>
![Fig9](/assets/dualstylegan/fig9.png)

그림 9. StyleCariGAN과 비교.

* [그림 8](#Fig8)은 여섯 가지의 SotA 기법인 image-to-image-translation-based StarGAN2, GNR, U-GAT-IT, StyleGAN-based UI2I-style, Toonify, Few-Shot Adaptation(FS-Ada)과 질적 비교를 제시함.
* Image-to-image translation과 FS-Ada는 $$256 \times 256$$ 이미지를 사용함. 다른 기법들은 $$1024 \times 1024$$를 지원함.
* Toonify, FS-Ada, U-GAT-IT은 이미지 수준 스타일보다는 도메인 수준을 학습함. 그래서 이것들의 결과는 스타일 예시들과 일관되지 않음.
* 심각한 데이터 불균형 문제는 유효한 순환 변환 트레이닝을 어렵게 함. 그래서, StarGAN2와 GNR은 스타일 이미지를 오버피팅하고 애니메이션 스타일에서 입력 얼굴을 무시함.
* UI2I-style은 레이어 교환을 통해 좋은 색 스타일을 캡처하지만, 모델 비정렬은 구조 특징점들을 섞이기 어렵게 하여, [2절](#Sec2)에서도 분석한 것처럼 구조적 스타일 변환 실패를 초래함.
* 비교적, DualStyleGAN은 색과 복잡한 구조에서 예시 스타일의 가장 좋은 스타일을 전송함.
<br/><br/>
* 성능 측정을 양적화하기 위해, 사용자 연구를 수행하였는데, 27명의 피실험자에게 네 가지의 예시 기반 스타일 전송 기법들 중 가장 좋은 결과로 생각되는 것을 선택하도록 함.
* 각 스타일 데이터셋은 측정을 위해 10개의 결과를 사용함. [표 1](#Table1)은 평균 선호 점수를 요약하며, 여기서 DualStyleGAN은 최고의 점수를 받음.
* **StyleCariGAN과의 비교**
    * 더 나아가 캐리커처에서 진보된 StyleCariGAN과 비교함.
    * StyleCariGAN은 StyleGAN과 순환 변환을 결합하여 색 전송을 위해 스타일 혼합을 도입하고 순환 변환과 구조 전송을 학습함. 입력에 대한 최적화를 통해 콘텐츠와 예시 이미지의 잠재 코드를 찾기 위해 이를 따르기로 함.
    * 잠재 코드가 공식 스타일 팔레트에서 랜덤으로 샘플링되었는지 또는 예시 이미지에서 샘플링되었는지에 따라 StyleCariGAN은 $$256 \times 256$$ 이미지에서 랜덤 또는 예시 기반 스타일을 전송할 수 있음.
    * [그림 9](#Fig9)와 같이 StyleCariGAN은 순환 변환이 전체 구조 스타일만 학습하므로 동일한 얼굴 구조를 생성함.
    * 비교해 보면, 본 기법은 예시를 기반으로 구조 스타일을 효과적으로 조정함.
    * 게다가, StyleCariGAN이 6K 트레이닝 이미지를 사용하더라도, 결과는 해상도와 시각적 품질 면에서 본 기법이 더 높음.

<div id="Sec4.2"></div>
### Ablation Study

<div id="Fig10"></div>
![Fig10](/assets/dualstylegan/fig10.png)

그림 10. Ablation Study.

<div id="Fig11"></div>
![Fig11](/assets/dualstylegan/fig11.png)

그림 11. 제안된 외부 스타일 경로는 의미론적으로 계층적 구조 변조를 학습함.

* **Paired data**
    * [그림 10(a)](#Fig10)은 [3.1절](#Sec3.1)에서 얼굴-초상화 지도 유무의 결과를 비교함.
    * 지도가 없을 때, 모델은 입력 얼굴 구조를 고려하지 않고 초상화를 오버피팅함.
    * .지도는 효과적으로 모델이 얼굴과 초상화 사이의 구조적 관계를 찾도록 가이드하고, 더 합당한 결과로 이끎.
* **Regularization**
    * 콘텐츠 loss ([Eq. (3)](#Eq3)) 내 정규화 항의 효과는 [그림 10(b)](#Fig10)에 나타남.
    * 정규화 항 없이, 모델은 예시의 머리카락 스타일을 오버피팅함. 정규화 항을 사용하면 이 이슈를 해결함.
    * 큰 $$\lambda_\mathrm{reg}$$은 입과 같이 입력 얼굴 모양을 과하게 보존함. 따라서, trade-off로 $$\lambda_\mathrm{reg} = 0.005$$를 사용함.
* **Progressive fine-tuning**
    * [그림 10(c)](#Fig10)에서 보듯이, 단계 I의 초기화 없이, 사전 트레이닝된 StyleGAN의 생성 공간은 심각하게 변하고 ([그림 6(b)](#Fig6)), 이는 전이 트레이닝을 완전히 실패하게 함.
    * 얼굴 의미론적 특징점들을 캡처하기 위한 실제 얼굴의 사전 트레이닝 없이, 외부 스타일 경로는 단계 III에서 복잡한 task를 수행할 수 없음.
    * 오직 완전 점진적인 fine-tuning을 통해, DualStyleGAN은 정확히 외부 스타일을 전송할 수 있음.
* **Effect of different layers**
    * 외부 스타일 경로의 각 레이어가 얼굴 특징점들에 주는 영향을 알아보기 위해, 매번 레이어의 부분집합을 활성화하고 (예를 들어, $$\mathrm{w} = \left[ 3 * 0 , 2 * 1 , 13 * 0 \right]$$는 오직 두 개의 $$16 \times 16$$ 레이어만 활성화함) [그림 11](#Fig11)에서 결과를 비교함.
    * AdaIN 기반 색 변조는 StyleGAN에서 잘 연구되었으므로, 거친 해상도 레이어에서 구조 변조에서만 집중함.
    * 초기 레이어들은 전반적 얼굴 모양을 조정하고, $$16 \times 16$$ 레이어들은 입과 같은 얼굴 성분들을 과장하고, $$32 \times 32$$ 레이어들은 주름과 같은 지역적인 모양에 집중함.

<div id="Sec4.3"></div>
### Further Analysis

<div id="Fig12"></div>
![Fig12](/assets/dualstylegan/fig12.png)

그림 12. 사진으로부터 색과 구조의 보존.

<div id="Fig13"></div>
![Fig13](/assets/dualstylegan/fig13.png)

그림 13. 내부 및 외부 스타일 혼합.

<div id="Fig14"></div>
![Fig14](/assets/dualstylegan/fig14.png)

그림 14. Pixar, Comic, Slam Dunk 스타일에서의 성능.

<div id="Fig15"></div>
![Fig15](/assets/dualstylegan/fig15.png)

그림 15. Unseen 스타일에서의 성능.

* **Color and structure preservation**
    * 사용자들은 Toonify처럼 원본 사진의 색을 유지하길 원할 수 있음. 색 보존에 대한 두 가지 방법을 제공함.
    * 첫 번째는 단순히 [그림 12(c)](#Fig12)와 같이 $$\mathrm{w}_c = \mathrm{0}$$을 설정하여 외부 스타일 경로 내 색 관련 레이어를 비활성화하는 것임.
    * 다른 방법은 마지막 11개 레이어에서 외부 스타일 코드를 내부 스타일 코드로 대체하는 것임. 첫 번째 방법에 비해, 내부 잠재 코드는 추가적으로 색 변형 블록을 통과하여, [그림 12(d)](#Fig12)에서처럼 최종 색이 타깃 도메인에 더 정렬되게 함.
    * 마지막으로, 구조 보존은 $$\mathrm{w}_s < \mathrm{1}$$로 설정하여 쉽게 달성될 수 있음.
    * [그림 12(e)](#Fig12)는 $$\mathrm{w}_s = \mathrm{0.5}$$에서 유한 스타일 전송의 예시를 제시함.
* **Style blending**
    * [그림 13](#Fig13)에서, 우리는 두 내부 및 외부 스타일 코드를 interpolation 함으로써 스타일을 혼합함.
    * 부드러운 전환은 스타일 조작의 합당한 범위를 암시함.
* **다른 스타일에서의 성능**
    * 더 나아가 인터넷에서 Pixar, Comic, Slam Dunk 스타일로 각각 122장, 101장, 120장의 이미지의 데이터셋을 수집함.
    * 본 기법은 이러한 스타일로 [그림 14](#Fig14)와 같이 좋은 성능을 달성함.
* **Unseen 스타일에서의 성능**
    * 트레이닝 데이터와 동떨어진 unseen 스타일이 주어졌을 때, 본 기법은 합당하지만 덜 일관적인 스타일을 전송함([그림 15(c)](#Fig15)).
    * [3.4절](#Sec3.4)과 같이 unseen 이미지를 destylize하여 고정된 내부 스타일 코드를 얻고 외부 스타일 코드를 최적화함으로써, 더 좋은 스타일이 학습됨([그림 15(d)](#Fig15)).
    * 그러나, 약간의 artifact가 나타남. 추후 연구에 강인한 unseen 스타일 확장으로 남겨 둠.

<div id="Sec4.4"></div>
### Limitations

<div id="Fig16"></div>
![Fig16](/assets/dualstylegan/fig16.png)

그림 16. Unseen 스타일에서의 성능.

* [그림 16](#Fig16)에서 우리는 DualStyleGAN의 세 가지 전형적인 실패 사례를 보여줌.
    * 첫째, 얼굴 특징은 잘 포착되지만 모자와 배경 텍스처와 같은 얼굴이 아닌 영역의 세부 사항은 결과에서 손실됨.
    * 둘째, 애니메이션 얼굴에는 종종 매우 추상적인 코가 있음. 사진의 색상을 유지하면 코는 분명해지지만 애니메이션 스타일에 비해 부자연스러워짐.
    * 셋째, 본 기법은 여전히 데이터 바이어스 문제를 겪고 있음. 애니메이션 데이터셋은 생머리와 앞머리에 대한 강한 바이어스를 가지고 있어 앞머리가 없는 곱슬머리를 처리하는 데 실패함. 한편, 극도로 큰 눈과 같은 흔하지 않은 스타일은 잘 모방될 수 없음.
* 결과적으로 데이터 불균형 문제가 심각한 task에 본 기법을 적용하면 표현되지 않은 데이터에 대한 불만족스러운 결과가 초래될 수 있음.

<div id="Sec5"></div>
## Conclusion and Future Work

* 본 논문에서, 우리는 StyleGAN의 스타일 제어를 원래 도메인에서 유지하면서 새로운 도메인에서 스타일 조건을 수용하도록 StyleGAN을 확장함.
* 이는 우호적인 데이터 요구 사항과 함께 고해상도 예제 기반 초상화 스타일 전송을 흥미롭게 적용하는 결과를 초래함.
* StyleGAN에 대한 추가적인 스타일 경로가 있는 DualStyleGAN은 유연하고 다양한 초상화 생성을 위해 내부 스타일과 외부 스타일을 효과적으로 모델링하고 변조할 수 있음.
* DualStyleGAN에 대한 유효한 전이 학습이 특별한 구조 설계와 점진적인 트레이닝 전략으로 달성될 수 있음을 보여줌.
* 구조와 데이터 측면에서 모델 확장에 대한 본 아이디어가 보다 일반적인 I2I 변환 및 지식 증류(knowledge distillation)와 같은 다른 task에 잠재적으로 적용될 수 있다고 기대할 수 있음.
* 향후 연구에서, 쉽게 사용할 수 있도록 입력 사진에 대한 적합한 스타일 이미지와 그 가중치 벡터 $$\mathrm{w}$$의 권장 사항을 살펴보고 데이터 증강(data augmentation)을 통해 데이터 편향 (data bias) 문제를 완화하고자 함.

[Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer]: https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Pastiche_Master_Exemplar-Based_High-Resolution_Portrait_Style_Transfer_CVPR_2022_paper.html
[Github]: https://github.com/williamyang1991/DualStyleGAN