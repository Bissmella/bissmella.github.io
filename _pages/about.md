---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a Machine Learning Engineer with a Master’s degree (M2) in **Mathematics, Vision & Learning (MVA)** from **École Normale Supérieure Paris-Saclay**.  
My work spans **deep learning, multimodal modeling, reinforcement learning, and LLM/VLM research**, with hands-on experience both in academic labs and applied ML roles.

I have previously worked as:

- **Research Engineer**, L2TI Lab, Université Sorbonne Paris-Nord, wroked on multimodal object detection, few-shot learning, transformer-based models, and dataset creation.  
- **ML Engineer & Data Analyst**, United Nations programs: applied ML, image analysis.  
- **Research Intern**, ISIR Lab (Sorbonne University): supervised and RL fine-tuning for LLMs/VLMs, exploration dynamics in RL for LLMs, and value modeling for planning for robots.

My research includes a **peer-reviewed journal publication**, an **ICIP 2024 oral presentation**, and an **arXiv preprint**.

---

# **Open Source**

## **1. Unified sequence parallelism implementation for HF diffusers**
[PR link](https://github.com/huggingface/diffusers/pull/12693)

## **2. Vectorized IoU computation for PerceptionMetrics library**
[PR link](https://github.com/JdeRobot/PerceptionMetrics/pull/398)

# **Selected Projects**

## **1. Real-time Meeting Copilot**
A production-style meeting assistant combining **speech-to-text inference**, **LLM inference**, and **vector search** for live transcription and interactive Q&A.  
[code](https://github.com/Bissmella/meeting_assistant)

<div style="text-align: center;">
  <video width="480" controls>
    <source src="/files/meeting_copilot_demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

---

## **2. Minimalist voice-assistant implementation:**
A minimialist implementation for a voice AI chatbot with low latency, and some simple tool calling implementaiton for LLM.

## **3. MVA Research Internship: RL + LLM/VLMs for robotics planning**
Explored:  
- LLM/VLM bias effects on exploration  
- LLM/VLM priors as exploration drivers  
- Enhanced value approximation via local utility functions for VLMs
[Report](/files/BAHADURI%20BISSMELLA%20RAPPORT.pdf)
[LLM finetuning](https://github.com/Bissmella/LLM-curiosity-RL)
[VLM finetuning framework](https://github.com/Bissmella/VLM_finetune)

---



## **4. Curiosity-Driven RL for LLMs**
Designed a curiosity-based exploration framework using action-level novelty and sequence-level novelty via a T5-based temporal predictor.  
[code](https://github.com/Bissmella/LLM-curiosity-RL)

---

## **5. PnP-Flow: Plug-and-Play Image Restoration with Flow Matching**
Trained a 2D Flow Matching model and integrated it into a plug-and-play restoration algorithm.  
[code](https://github.com/Bissmella/PnP-FM) · [report](/files/PnP_FM-report.pdf)

---

## **6. Fine-tuning LLaMA-2 on Personal Chats**
End-to-end pipeline for fine-tuning LLaMA-2 on user-specific chat data using free Colab GPU.  
[code](https://github.com/Bissmella/FineTune_llama_on_chat_data)

---

# **Publications**

1. **Indirect Attention: Turning Context Misalignment into a Feature**  
   *B. Bahaduri, H. Talaoubrid, F. Feng, Z. Ming, A. Mokraoui*  
   [paper](https://www.arxiv.org/abs/2509.26015)  

2. **A Comparative Attention Framework for Better Few-shot Object Detection on Aerial Images**  
   *P. Le Jeune, B. Bahaduri, A. Mokraoui*  
   *Pattern Recognition 2024*  
   [paper](https://www.sciencedirect.com/science/article/pii/S0031320324009944) · [code](https://github.com/pierlj/aaf_framework)

3. **Multimodal Transformer using Cross-Channel Attention for Object Detection in Remote Sensing Images**  
   *B. Bahaduri, Z. Ming, F. Feng, A. Mokraoui*  
   *ICIP 2024 (oral)*  
   [paper](https://arxiv.org/abs/2310.13876) · [code](https://github.com/Bissmella/Small-object-detection-transformers)



<!-- For more info
------
More info about configuring academicpages can be found in [the guide](https://academicpages.github.io/markdown/). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
 -->
