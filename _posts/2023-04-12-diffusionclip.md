---
layout: post
title:  "[리뷰] DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation"
date:   2023-04-12 21:06:00 +0900
categories: [Diffusion Model]
tags: [Diffusion Model, Image Generation, Image Translation]
use_math: true
---

{: .note .warning}
공부하면서 정리한 내용이므로 부정확한 내용이 포함되어 있을 수 있음.

Title: [DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation]
<br/>
Authors: G. Kim, T. Kwon, and J. C. Ye
<br/>
Submitted to [CVPR 2022]
<br/>
Project Page: [Github]

## Abstract

* 최근 CLIP(Contrastive Language-Image Pretraining)과 결합된 GAN inversion 기법은 텍스트 프롬프트에 의해 가이드되는 zero-shot 이미지 조작을 가능하게 함.
* 그러나 제한적인 GAN inversion 기법이 능력으로 인해 다양한 실제 이미지에 적용하는 것은 여전히 어려움.
* 특히, 이러한 접근 방식은 트레이닝 데이터에 비해 새로운 포즈, 시점 및 매우 가변적인 콘텐츠를 가진 이미지를 재구성하거나, 객체 아이덴티티를 변경하거나, 원하지 않는 이미지 아티팩트를 생성하는 데 어려움을 겪는 경우가 많음.
* 이러한 문제를 완화하고 실제 이미지의 충실한 조작을 가능하게 하기 위해 확산 모델(diffusion model)을 사용하여 텍스트 유도 (text-driven) 이미지 조작을 수행하는 DiffusionCLIP이라는 새로운 기법을 제안함.
* 최근 확산 모델들의 완전 inversion 능력과 고품질 이미지 생성 능력을 기반으로, 본 기법은 unseen 도메인 간에도 zero-shot 이미지 조작을 성공적으로 수행하고 광범위한 ImageNet 데이터셋의 이미지를 조작함으로써 일반적인 적용을 향헤 한 걸음 더 나아감.
* 또한 간단한 다중 속성 (multi-attribute) 조작을 가능하게 하는 새로운 노이즈 조합 방법을 제안함.
* 광범위한 실험과 인간의 평가를 통해 기존의 baseline에 비해 본 기법의 강력하고 우수한 조작 성능을 확인함.

<div id="Fig1"></div>
![Fig1](/assets/diffusionclip/fig1.png)

그림 1. DiffusionCLIP은 실제 이미지의 충실한 텍스트 유도 조작을 (a) SotA GAN inversion 기반의 기법이 실패할 때 중요한 디테일을 보존하여 가능케 함. 다른 새로운 응용은 (b) 두 unseen 도메인 사이의 이미지 변환, (c) unseen 도메인으로의 밑그림 조건부 이미지 합성, (d) 다수 속성 변화를 포함함.

## Introduction

* 새로운 DiffusionCLIP - 확산 모델에 의한 CLIP-guided 강력한 이미지 조작 기법을 제안함.
    * 여기서, 인풋 이미지는 먼저 정방향 확산을 통해 잠재 노이즈(latent noise)로 변환함. DDIM의 경우, 역방향 확산에 대한 스코어 함수가 정방향 확산의 스코어 함수와 동일하게 유지될 경우 잠재 노이즈는 역방향 확산을 사용하여 원본 이미지로 거의 완벽하게 반전될 수 있음.
    * 따라서 DiffusionCLIP의 핵심 아이디어는 텍스트 프롬프트를 기반으로 생성된 이미지의 속성을 제어하는 CLIP loss를 사용하여 역방향 확산 프로세스에서 스코어 함수를 fine-tuning 하는 것임.
<br/><br/>
* 따라서 DiffusionCLIP은 트레이닝된 도메인과 unseen 도메인 모두에서 이미지 조작을 성공적으로 수행할 수 있음([그림 1(a)](#Fig1)).
* 심지어 unseen 도메인에서 다른 unseen 도메인으로 이미지를 변환하거나([그림 1(b)](#Fig1)) 밑그림에서 unseen 도메인으로 이미지를 생성할 수 있음([그림 1(c)](#Fig1)).
* 또한 여러 fine-tuning 된 모델에서 예측된 노이즈를 단순히 결합함으로써 하나의 샘플링 프로세스만을 통해 여러 속성을 동시에 변경할 수 있음([그림 1(d)](#Fig1)).
* 게다가, DiffsuionCLIP은 광범위한 변화가 있는 ImageNet 데이터셋([그림 6](#Fig6))의 이미지를 조작함으로써 일반적인 응용을 향한 또 한 걸음을 내딛는데, 이는 저조한 재구성으로 인해 GAN-inversion으로 거의 탐구되지 않았음.
<br/><br/>
* 또한, 고품질 및 빠른 이미지 조작으로 이끄는 최적의 샘플링 조건을 찾기 위한 체계적인 접근 방식을 제안함. 정성적 비교 및 인간 평가 결과는 본 방식이 SotA 기준을 능가하는 강력하고 정확한 이미지 조작을 제공할 수 있음을 보여줌.

## Related Works

<div id="Sec2.2"></div>
### CLIP Guidance for Image Manipulation

* CLIP은 시각적 개념들을 자연어 (natural language) 지도로 효과적으로 학습하기 위해 제안되었음.
    * CLIP에서, 텍스트 인코더와 이미지 인코더는 데이터셋에서 어떤 텍스트가 어떤 이미지에 매칭되는지를 식별하기 위해 사전 트레이닝(pretrain)됨.
    * 그래서, 본 텍스트 유도 이미지 조작을 위해 사전 트레이닝된 CLIP 모델을 사용함.
<br/><br/>
* CLIP으로부터 지식을 효과적으로 추출하기 위해, 두 가지 서로 다른 loss가 제안되었음: 전역 타깃 loss, 지역 방향성 loss.
    * 전역 CLIP loss는 CLIP 공간에서 생성된 이미지와 주어진 타깃 텍스트 사이의 코사인 거리(cosine distance)를 다음과 같이 최소화하도록 함.
    <br/><br/>
    <div id="Eq8"></div>
    ![Eq8](/assets/diffusionclip/eq8.png)
    <br/><br/>
    여기서 $$y_{\mathrm{tar}}$$는 타깃의 텍스트 설명, $$\pmb{x}_{\mathrm{gen}}$$는 생성된 이미지를 나타내며, 그리고 $$D_{\mathrm{CLIP}}$$은 CLIP 공간에서 인코딩된 벡터들 사이의 코사인 거리를 반환함.
    * 반면, 지역 방향성 loss는 낮은 다양성 및 적대적 공격에 대한 민감성 같은 전역 CLIP loss의 이슈를 완화하기 위해 설계됨. 지역 방향성 CLIP loss는 레퍼런스 이미지와 생성된 이미지의 임베딩 사이의 방향을 CLIP 공간에서 레퍼런스 텍스트와 타깃 텍스트 짝의 임베딩 사이의 방향에 정렬되도록 다음과 같이 유도함.
    <br/><br/>
    <div id="Eq9"></div>
    ![Eq9](/assets/diffusionclip/eq9.png)
    <br/><br/>
    그리고 $$\Delta{T}=E_T(y_{\mathrm{tar}})-E_T(y_{\mathrm{ref}})$$, $$\Delta{I}=E_I(\pmb{x}_{\mathrm{gen}})-E_I(\pmb{x}_{\mathrm{ref}})$$임. 여기에서, $$E_I$$와 $$E_T$$는 각각 이미지 인코더와 텍스트 인코더이고, $$y_{\mathrm{ref}}$$와 $$\pmb{x}_{\mathrm{ref}}$$는 각각 소스 도메인 텍스트와 이미지임.
    * 방향성 CLIP loss로 가이드된 조작된 이미지들은 이미지 표현들 사이의 방향을 레퍼런스 텍스트와 타깃 텍스트 사이의 방향에 정렬함으로써, 뚜렷한 이미지들이 생성될 것이므로 mode-collapse 이슈에 강인한 것으로 알려져 있음. 또한, 동요(perturbation)는 이미지에 따라 다를 것이므로 적대적 공격에 더 강인함. 더 관련된 연구들은 부록 Section A에 나타남.

<div id="Sec3"></div>
## DiffusionCLIP

<div id="Fig2"></div>
![Fig2](/assets/diffusionclip/fig2.png)

그림 2. DiffusionCLIP의 overview. 인풋 이미지는 먼저 확산 모델을 통해 잠재로 변환됨. 그 다음, 방향성 CLIP loss로 가이드되어, 확산 모델은 fine-tuning 되고, 업데이트된 샘플은 역방향 확산 동안 생성됨.

* 이미지 조작을 위한 제안된 DiffusionCLIP의 전반적인 흐름은 [그림 2](#Fig2)에 보임.
    * 여기서, 인풋 이미지 $$\pmb{x}_0$$은 먼저 사전 트레이닝된 확산 모델 $$\epsilon_\theta$$를 사용하여 잠재 $$\pmb{x}_{t_0} (\theta)$$로 변환됨.
    * 그 다음, CLIP loss로 가이드되어, 역방향 경로에서 확산 모델은 타깃 텍스트 $$y_{\mathrm{tar}}$$로 유도된 샘플들을 생성하도록 fine-tuning 됨.
    * 결정론적 (deterministic) 정방향-역방향 프로세스들은 DDIM에 기반을 둠.
    * Unseen 도메인 사이의 변환에 대해, 잠재 생성은 또한 후술될 것처럼 정방향 DDPM 프로세스로 행해짐.

<div id="Sec3.1"></div>
### DiffusionCLIP Fine-tuning

* Fine-tuning 면에서, 잠재나 확산 모델 자체를 변경할 수 있음. 부록 Section D에서 분석된 것과 같이, 직접 모델 fine-tuning이 더 효과적이라는 점이 발견됨. 특히, 역방향 확산 모델 $$\epsilon_\theta$$를 fine-tuning 하기 위해, 방향성 CLIP loss $$\mathcal{L}_\mathrm{direction}$$과 아이덴티티 loss $$\mathcal{L}_\mathrm{id}$$로 구성된 다음 목적(objective)을 사용함.
<br/><br/>
<div id="Eq10"></div>
![Eq10](/assets/diffusionclip/eq10.png)
<br/><br/>
여기서 $$\pmb{x}_{0}$$은 원본 이미지, $$\hat{\pmb{x}}_{0} (\hat{\theta})$$는 최적화된 파라미터 $$\hat{\theta}$$로 잠재 $$\pmb{x}_{t_0}$$로부터 생성된 이미지, $$y_\mathrm{ref}$$는 레퍼런스 텍스트, $$y_\mathrm{tar}$$는 이미지 조작을 위해 주어진 타깃 텍스트임.
<br/><br/>
* 여기서, CLIP loss는 최적화를 지도하는 데 핵심 성분임. 상술한 두 가지 유형의 CLIP loss 중, [Section 2.2](#Sec2.2)에서 언급한 매력적인 특성 덕분에 방향성 CLIP loss를 가이드로 사용함.
* 텍스트 프롬프트를 위해, 방향성 CLIP loss는 트레이닝 동안 레퍼런스 텍스트 $$y_\mathrm{ref}$$와 타깃 텍스트 $$y_\mathrm{tar}$$를 요구함. 예를 들어, 주어진 얼굴 이미지의 표정을 화난 표정으로 바꾸는 경우, 'face'를 레퍼런스 텍스트로 'angry face'를 타깃 텍스트로 사용할 수 있음.
* 이 논문에서, 각 텍스트 프롬프트를 언급하기 위해 종종 축약된 단어를 사용함(예: 'tanned face'를 'tanned'로).
<br/><br/>
* 아이덴티티 loss $$\mathcal{L}_\mathrm{id}$$는 원하지 않는 변화를 방지하고 객체의 아이덴티티를 보존하기 위해 도입됨.
* 일반적으로 $$l_1$$ loss를 아이덴티티 loss로 사용하고, 사람 얼굴 이미지 조작의 경우, [[13]]의 얼굴 아이덴티티 loss를 추가함.
* 아이덴티티 loss의 필요성은 컨트롤의 타입에 따라 다름.
    * 일부 컨트롤에 대해서는, 픽셀 유사도와 사람 아이덴티티의 보존이 중요한 반면 (예: 표정, 머리 색) 다른 컨트롤은 엄격한 모양과 색 변화를 중요시함(예: 미술 작품, 종의 변화).

<div id="Fig3"></div>
![Fig3](/assets/diffusionclip/fig3.png)

그림 3. $$t$$에 걸친 공유된 구조의 확산 모델을 fine-tuning 하는 동안의 경사 흐름.

* 기존의 확산 모델들은 Transformer 안에 사용된 것처럼 사인(sine)형 위치 임베딩을 사용한 $$t$$의 정보를 삽입함으로써, 모든 $$t$$에 대해 공유된 U-Net 구조를 채택함. 이 구조로, DiffusionCLIP fine-tuning 동안 경사 흐름(gradient flow)은 [그림 3](#Fig3)과 같이 표현될 수 있는데, 이는 RNN (resursive neural network) 트레이닝의 유사한 프로세스임.
<br/><br/>
* 일단 확산 모델이 fine-tuning 되면, 사전 트레이닝된 도메인으로부터의 이미지는 [그림 4](#Fig4)에 묘사된 것처럼 타깃 텍스트 $$y_\mathrm{tar}$$에 대응하는 이미지로 조작될 수 있음.
* Fine-tuning 과정과 모델 구조의 디테일은, 부록 Section B와 C 참고.

<div id="Sec3.2"></div>
### Forward Diffusion and Generative Process

* [Eq.3](#Eq3)의 DDPM 샘플링 프로세스는 확률적(stochastic)이기 때문에, 동일한 잠재에서 생성된 샘플은 매번 다를 것임.
* 샘플링 과정이 결정론적이라 하더라도, [Eq.1](#Eq1)과 같이 랜덤 가우시안 노이즈가 추가되는 DDPM의 정방향 프로세스 또한 확률적이므로, 원본 영상의 재구성이 보장되지 않음.
* 이미지 조작을 목적으로 하는 확산 모델의 이미지 합성 성능을 완전히 활용하기 위해서는, 성공적인 이미지 조작을 위해 사전 트레이닝된 확산 모델과 함께 정방향과 역방향 모두에서 결정론적 프로세스가 필요함.
* 한편, unseen 도메인 간의 이미지 변환의 경우 DDPM에 의한 확률적 샘플링이 종종 도움이 되며, 이는 나중에 더 자세히 설명될 예정임.
<br/><br/>
* 완전 inversion에 대해, 결정론적 역방향 DDIM 프로세스를 생성 프로세스로 채택하고 그 역(reversal)의 ODE 근사(approximation)를 정방향 확산 프로세스로 채택함. 특히, 잠재를 얻기 위한 결정론적 정방향 DDIM 프로세스는 다음과 같이 표현됨.
<br/><br/>
<div id="Eq12"></div>
![Eq12](/assets/diffusionclip/eq12.png)
<br/><br/>
그리고 얻어진 잠재로부터 샘플을 생성하기 위한 결정론적 역방향 DDIM 프로세스는 다음과 같이 됨.
<br/><br/>
<div id="Eq13"></div>
![Eq13](/assets/diffusionclip/eq13.png)
<br/><br/>
여기서 $$f_\theta$$는 [Eq. 6](#Eq6)에 정의되어 있음. ODE 근사 식의 유도에 대해서는, 부록의 Section A를 참고.
<br/><br/>
* DiffusionCLIP의 또 다른 중요한 contribution은 빠른 샘플링 전략임. 특히, 마지막 time step $$T$$까지 정방향 확산을 수행하는 대신, $$t_0 < T$$까지 수행함으로써 정방향 확산을 가속할 수 있는데, 이를 'return step'이라고 함.
* 더 나아가 트레이닝을 $$[1, t_0]$$ 사이의 더 적은 이산화 (discretization) 스텝들을 사용하여 트레이닝을 가속할 수 있는데, 이것들은 정방향 확산과 생성 프로세스에서 각각 $$S_\mathrm{for}$$와 $$S_\mathrm{gen}$$으로 나타내어짐.
    * 예를 들어, $$T$$가 일반적인 선택으로 1000으로 세팅될 때, $$t_0 \in [300, 600]$$과 $$(S_\mathrm{for}, S_\mathrm{gen}) = (40, 6)$$의 선택이 목표를 만족시킴. $$S_\mathrm{gen} = 6$$이 불완전한 재구성을 만들어내더라도, 트레이닝을 위해 요구되는 객체의 아이덴티티는 충분히 보존됨을 발견함.
* 나중에 $$S_\mathrm{for}$$, $$S_\mathrm{gen}$$, $$t_0$$에 대한 양적 및 질적 분석 결과를 실험과 부록 Section F에서 보일 예정임.
<br/><br/>
* 마지막으로, 몇 개의 잠재가 미리 계산되었다면([그림 2](#Fig2)의 회색 사각형 영역), 그 잠재를 다른 속성을 합성하는 데 재활용함으로써 fine-tuning 시간을 더욱 줄일 수 있음. 이런 세팅으로, NVIDIA Quardro RTX 6000에서 fine-tuning이 1~7분 이내로 완료됨.

<div id="Sec3.3"></div>
### Image Translation between Unseen Domains

<div id="Fig4"></div>
![Fig4](/assets/diffusionclip/fig4.png)

그림 4. DiffusionCLIP의 새로운 응용. (a) 사전 트레이닝된 도메인의 이미지를 CLIP-guided 도메인으로 조작. (b) Unseen 도메인 간의 이미지 변환. (c) Unseen 도메인에서 밑그림 조건부 이미지 생성. (d) 다수 속성 변화. $$\epsilon_\theta$$와 $$\epsilon_{\hat{\theta}}$$는 각각 원본 사전 트레이닝된 확산 모델과 fine-tuning 된 확산 모델을 가리킴.

* DiffusionCLIP을 통한 fine-tuning 모델은 [그림 4](#Fig4)와 같이 추가적인 새로운 이미지 조작 task를 수행하기 위해 활용될 수 있음.
<br/><br/>
* 먼저 [그림 4(b)와 (c)](#Fig4)에 각각 설명된 것처럼 unseen 도메인에서 unseen 도메인으로 이미지 변환을 수행하고, unseen 도메인에서 밑그림 조건부 이미지 합성을 수행할 수 있음.
* 이 어려운 문제를 해결하기 위한 핵심 아이디어는 비교적 수집하기 쉬운 데이터셋에서 트레이닝된 확산 모델을 삽입하여 두 도메인 사이를 연결하는 것임. 구체적으로 [[8], [25]]에서 사전 트레이닝된 확산 모델을 사용하면 unseen 도메인에서 트레이닝된 이미지가 트레이닝된 도메인의 이미지로 변환될 수 있음을 발견함.
* 이 방법을 DiffusionCLIP과 결합하면 이제 소스 도메인과 타깃 도메인 모두에 대한 zero-shot 설정으로 이미지를 변환할 수 있음. 구체적으로, 소스 unseen 도메인 $$\pmb{x}_0$$의 이미지는 도메인 관련 구성 요소가 모호하지만 객체의 아이덴티티 또는 의미가 보존되는 충분한 time step $$t_0$$까지 Eq.1의 정방향 DDPM 프로세스를 통해 먼저 동요됨. 이것은 보통 500으로 설정됨.
* 다음, 사전 트레이닝된 도메인의 이미지들 $$\pmb{x}_0^\prime$$은 [Eq. 13](#Eq13)의 역방향 DDIM 프로세스를 사용하여 원본 사전 트레이닝된 모델 $$\epsilon_\theta$$로 샘플링됨. 그 다음, $$\pmb{x}_0^\prime$$은 [그림 4(a)](#Fig4)에서 fine-tuning 된 모델 $$\epsilon_{\hat{\theta}}$$로 하듯이 CLIP-guided unseen 도메인에서 이미지 $$\hat{\pmb{x}_0}$$로 조작함.

## Discussion and Conclusion

* 본 논문에서, 사전 트레이닝된 확산 모델과 CLIP loss를 사용하는 text-guided 이미지 조작 방법인 DiffusionCLIP을 제안함.
* 거의 완벽한 inversion 특성 덕분에 DiffusionCLIP은 확산 모델을 fine-tuning 하여 도메인 내 (in-domain) 및 도메인 외부 (out-of-domain) 조작 모두에서 우수한 성능을 보여줌. 또한 다양한 샘플링 전략을 결합하여 fine-tuning 된 모델을 사용하는 몇 가지 새로운 응용을 제시함.
* DiffusionCLIP에는 한계와 사회적 위험이 있음. 따라서, 사용자들에게 적절한 목적을 위해 본 기법을 신중하게 사용할 것을 권고함. 제한점과 부정적인 사회적 영향에 대한 자세한 내용은 부록 G와 H에 나와 있음.

[DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation]: https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html
[CVPR 2022]: https://openaccess.thecvf.com/CVPR2022?day=all
[Github]: https://github.com/gwang-kim/DiffusionCLIP.git
[8]: https://arxiv.org/abs/2108.02938
[13]: https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
[25]: https://arxiv.org/abs/2108.01073