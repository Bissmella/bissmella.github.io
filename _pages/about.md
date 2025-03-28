---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am an M2 MVA (Mathematics, Vision, Apprentisage/learning) student at École Normale Supérieure Paris-Saclay. Previously I both intered and worked as research engineer at L2TI lab of Univeristy of Sorbonne Paris-Nord where I mainly worked on multimodal object detection and few-shot learning for object detection.
More specifically, during my Master's internship at L2TI laboratory I worked on fusion of multi-modal (RGB & IR) data in aerial images for the purpose of object detection and I leveraged cross-channel attention mechanism for using the benefits of multi-modality in remote-sensing images. The output of this internship has been published as an oral paper at ICIP 2024.


Research interests
======
I am interested in computer vision, generative modelling, multimodal foundational models and theoretical understanding of deep learning. I am actively searching for internship and PhD program in these domains.


Publications
======
1. **A Comparative Attention Framework for Better Few-shot Object Detection on Aerial Images** [paper](https://www.sciencedirect.com/science/article/pii/S0031320324009944) [code](https://github.com/pierlj/aaf_framework) (Pattern Recognition 2024) Pierre Le Jeune, Bissmella Bahaduri, Anissa Mokraoui
2. **Multimodal Transformer using Cross-Channel Attention for Object Detection in Remote Sensing Images** [paper](https://arxiv.org/abs/2310.13876)   [code](https://github.com/Bissmella/Small-object-detection-transformers)
(ICIP-24, <span style="color:red">oral presentation</span>) Bissmella Bahaduri, Zuheng Ming, Fangchen Feng, Anissa Mokraoui

Talks
======
Vers un apprentissage pragmatique dans un contexte de données visuelles labellisées limitées:
**Indirect-attention: IA-DETR for one-shot object detection**, June 2024

Selected projects
======
1. **A Bayesian Approach for Preference Alignment for Language Models:** In this project/report we take a Bayesian approach to provide a better solution to the problem of noisy labels for LLM alignment with human preferences. First, we link the preference alignment setting to the more general Bayesian framework for noisy labels. Second, we experiment with loss functions, namely the reverse KL divergence that has more theoretical guarantees, and the Jensen-Shannon entropy. [code](https://github.com/Bissmella/llm_bayesian_preference.git) [Report](/files/bayesian_preference_LLM.pdf)


2. **A note on lazy training in differentiable programming:**
We explore how scaling factors influence the transition to lazy training, where models behave linearly around their initialization. Through theoretical analysis and extensive experiments on two-layer neural networks, we extend the demonstration of how large scaling factors induce minimal parameter updates, leading to faster convergence but potentially limiting the model’s ability to capture complex nonlinear relationships. Additionally, we examine the correlation between weight initialization scales and output scaling factors and visualize the loss landscape under different scaling conditions. Our visualizations, and empirical and theoretical analysis provide deeper insights into the training dynamics of neural networks and extend on top of lazy training dynamics understanding.
[code](https://github.com/RichardGou/MVA_GDA_PROJECT) [Report](/files/note_on_lazy_training.pdf)


3. **Fine-tune LLAMA-v2 on personal chat data:**
Llama v2 7b is a large language model (LLM) with 7 billion parameters that can be used for a variety of tasks, including text generation, translation, and question answering. However it can be fine-tuned on any other specific use such as on personal chat data for personal purposes using freely available colab GPU.
[code](https://github.com/Bissmella/FineTune_llama_on_chat_data)

4. **Blind navigation in 2d:**
This is a 2d implementation of the paper [Emergence of maps in the memories of blind navigation agents](https://arxiv.org/pdf/2301.13261). The main objective is to train a model for navigating an environment without any visual from the surroundings. The full project concerns an evolving environment where some part of the environment becomes unreachable as time evovles (for example a fire is spreading in the environment) and the goal of the agent is to navigate through the environment and reach the designated target. [code](https://github.com/Bissmella/Blind_navigation_2d.git)

<!-- For more info
------
More info about configuring academicpages can be found in [the guide](https://academicpages.github.io/markdown/). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
 -->
