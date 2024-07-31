# Basic-Information-Consolidation
# Surveys On Memes
A collection of papers on **Memes**.

**Unimodal Memes** are those that use a single mode of communication, typically text or image alone. In contrast, **Multimodal Memes** combine text and images to convey their message, requiring the interaction of both to be understood fully.

Relevant papers and links are introduced in the second section. The third section categorizes the above papers by tasks into: **Memes Classification, Toxic Memes, Hate Speech Detection, Sentiment Analysis and Memes Generation**.

## Datasets
For **toxic memes detection**:

![Example Image](figure/dataset.png "datasets for toxic memes detection")

## Surveys
- (*arXiv 2024.06*) Toxic Memes: A Survey of Computational Perspectives on the Detection and Explanation of MemeToxicities [[paper](https://arxiv.org/abs/2406.07353)]
- (*Springer'23*) A literature survey on multimodal and multilingual automatic hate speech identification [[paper](https://link.springer.com/article/10.1007/s00530-023-01051-8)]
- (*IEEE WIECON-ECE*) Predicting Social Emotions based on Textual Relevance for News Documents [[paper](https://arxiv.org/pdf/2009.08395)]
- (*IEEE WIECON-ECE*) A Survey on Multimodal Disinformation Detection [[paper](https://aclanthology.org/2022.coling-1.576.pdf)]
- (*ACL'22*) An integrated explicit and implicit offensive language taxonomy [[paper](https://www.researchgate.net/publication/372449006_An_integrated_explicit_and_implicit_offensive_language_taxonomy)]
- (*SSRN'21*) Automatic Hate Speech Detection: A Literature Review [[paper](https://www.researchgate.net/publication/351788021_Automatic_Hate_Speech_Detection_A_Literature_Review)]
- (*IJCAI'22*) Detecting and Understanding Harmful Memes: A Survey [[paper](https://arxiv.org/pdf/2205.04274)]
- (*Springer'23*)  Detecting hate speech in memes: a review [[paper](https://link.springer.com/article/10.1007/s10462-023-10459-7)]
- (*arXiv 2018.12*) Handling Bias in Toxic Speech Detection: A Survey [[paper](https://arxiv.org/pdf/2202.00126)]
- (*Springer'20*) A Multimodal Memes Classication: A Survey and Open Research Issues [[paper](https://link.springer.com/chapter/10.1007/978-3-030-66840-2_109)]
- (*ACL'20*) Detecting the Role of an Entity in Harmful Memes: Techniques and Their Limitations [[paper](https://arxiv.org/pdf/2205.04402)]
- (*WWW'24*) Decoding Memes: A Comprehensive Analysis of Late and Early Fusion Models for Explainable Meme Analysis [[paper](https://dl.acm.org/doi/pdf/10.1145/3589335.3652504)]
- (*Springer'23*) Meme-Text Analysis: Identifying Sentiment of Memes [[paper](https://link.springer.com/chapter/10.1007/978-981-99-3656-4_62)]
- (*AAAI'22*) Memotion 3: Dataset on Sentiment and Emotion Analysis of codemixed Hindi-English Memes[[paper](https://arxiv.org/pdf/2303.09892)]
- (*Information Fusion'22*) Multimodal sentiment analysis: A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001634)]
- (*arXiv 2024.06*) Large Language Models Meet Text-Centric Multimodal Sentiment Analysis: A Survey [[paper](https://arxiv.org/pdf/2406.08068)]
- (*ACL'22*) Findings of the Shared Task on Multimodal Sentiment Analysis and Troll Meme Classification in Dravidian Languages [[paper](https://aclanthology.org/2022.dravidianlangtech-1.39.pdf)]
- (*Journal*) Sentiment Analysis of Internet Memes on Social Platforms [[paper](https://libstore.ugent.be/fulltxt/RUG01/003/006/973/RUG01-003006973_2021_0001_AC.pdf)]
- (*arXiv 2023.05*) A Review of Vision-Language Models and their Performance on the Hateful Memes Challenge [[paper](https://arxiv.org/pdf/2305.06159)]
 
## A Brief Summary of the above reviews
Combining the above different review topics, tasks can be roughly summarized as follows :
### 1. Memes Classification [10 11 12]
![Example Image](figure/classification.png "Pipeline of Memes Classification")
#### V-L Multimodal Classification:
- **Image Captioning**: Generate the best textual explanations related to the content of the image.
- **Visual Question Answering (VQA)**: Answer questions by combining image and text information, requiring precise modeling of the correlation between image and text representations.
- **Hateful Meme Classification**: Detect and classify hate speech in memes, requiring precise association and alignment between image and text content.

#### Mainstream Architectures:
- **Two-Stream Architecture**: For instance, VilBERT and LXMERT process visual and textual information separately and then fuse them through Transformer layers.
- **Single-Stream Architecture**: Models like VisualBERT and UNITER directly fuse the two modalities at an early stage and process them with a single Transformer.
  
 ![Example Image](figure/fusion.png "Cross Model Fusion")

### 2. Toxic Memes/Hate Speech Detection [1 2 5 7 8 9 10]
#### Unimodal detection-Textual Toxicity:
- **Two surveys on textual toxicity are worth mentioning**: [5] categorizes offensive language types into a hierarchical taxonomy, differentiating between explicit and implicit language, while [9] provides insights into the subjective nature of toxicity detection, biases in existing datasets, the influence of content source and topic on dataset characteristics, and challenges in collecting toxic comments.

#### Multimodal detection:
- **Only three published works have surveyed on recent toxic meme analysis**: [8] were the first to survey efforts in automatic meme understanding, highlighting key challenges for future research. [8] surveyed methodologies for detecting hateful memes and introduced a taxonomy of machine learning architectures specifically for this purpose.
[10] surveyed methodologies for detecting hateful memes and introduced a taxonomy of machine learning architectures specifically.
- **[1] represents the latest review outcome in the field**: The paper present an updated analysis of datasets comprising toxic memes, detailing labels, task definitions, origins, and research applications, emphasizing the diversity of task definitions and labeling schemas; conduct a comprehensive review of computationally examined meme toxicities' concepts and definitions, addressing the definitional consensus gap and introducing a standardized taxonomy for classifying toxicity types; identify specific toxicity dimensions in memes (e.g., targets and conveyance tactics), outlining a framework that illustrates the interplay among these dimensions and varied meme toxicities.

![Example Image](figure/trend.png "Trend")

### 3. Sentiment Analysis/Memes Generation [1 13 14 16 17 18]

- **There are few recent review papers in this field, and none provide a comprehensive summary of current achievements**: Here, we introduce three representative papers: [13] briefly introduces related work since the first paper in this field appeared in 2014 and presents a very simple pipeline, but it is limited to unimodal pure text analysis. [14] and [17] summarize the results of competitions on specific datasets held for many years by AAAI and ACL, but do not provide results or paper achievements on other datasets in the current field. [16] discusses the application of large models in this field, but its scope is broader and analysis of memes should include background knowledge (such as whether it is theme-based), thus it is also not comprehensive.
- Although there have been many recent papers in this area, the only related review is [1], which provides an overview of recent papers on trends part.

## Ppaers
### Memes Classification
- (*ACL'21*) Findings of the Shared Task on Troll Meme Classification in Tamil [[paper](https://aclanthology.org/2021.dravidianlangtech-1.16.pdf)]
- (*ACL'23*) Multimodal Offensive Meme Classification with Natural Language Inference [[paper](https://aclanthology.org/2023.ldk-1.12.pdf)]
- (*EMNLP'22*) Prompting for Multimodal Hateful Meme Classification [[paper](https://aclanthology.org/2022.emnlp-main.22/)]
- (*EMNLP'22*) Hate-CLIPper: Multimodal Hateful Meme Classification based on Cross-modal Interaction of CLIP Features [[paper](https://arxiv.org/pdf/2210.05916)]
- (*SSCI'21*) Hateful Memes Classification using Machine Learning [[paper](https://ieeexplore.ieee.org/abstract/document/9659896)]
- (*springer'22*) Combining Knowledge and Multi-modal Fusion for Meme Classification [[paper](https://aclanthology.org/2023.ldk-1.12.pdf)]
- (*ICMR'23*) Multi-channel Convolutional Neural Network for Precise Meme Classification [[paper](https://dl.acm.org/doi/abs/10.1145/3591106.3592275)]
- (*arXiv 2021.08*) Do Images really do the Talking? Analysing the significance of Images in Tamil Troll meme classification [[paper](https://arxiv.org/pdf/2108.03886)]

### Memes Detection
- (*ACL'21*) Detecting Harmful Memes and Their Targets [[paper](https://aclanthology.org/2021.findings-acl.246.pdf)]
- (*ACL'21*) MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets [[paper](https://aclanthology.org/2021.findings-emnlp.379.pdf)]
- (*ACL'22*) Are you a hero or a villain? A semantic role labelling approach for detecting harmful memes [[paper](https://aclanthology.org/2022.constraint-1.3.pdf)]
- (*ACL'24*) A Multimodal Framework to Detect Target Aware Aggression in Memes [[paper](https://aclanthology.org/2024.eacl-long.153.pdf)]
- (*NIPS'21*) An Interpretable Approach to Hateful Meme Detection [[paper](https://arxiv.org/pdf/2108.10069)]
- (*WWW'24*) Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models [[paper](https://arxiv.org/abs/2401.13298)]
- (*arXiv 2022.04*) On Explaining Multimodal Hateful Meme Detection Models [[paper](https://arxiv.org/pdf/2204.01734)]
- (*SSRN'21*) Multimodal Computation or Interpretation? Automatic vs. Critical Understanding of Text-Image Relations in Racist Memes [[paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4578752)]
- (*Transactions on Asian and Low-Resource Language Information Processing*) A multimodal deep framework for derogatory social media post identification of a recognized person [[paper](https://dl.acm.org/doi/10.1145/3447651)]
- (*IEEE SSP*) On the evolution of (hateful) memes by means of multimodal contrastive learning [[paper](https://arxiv.org/pdf/2212.06573)]
- (*arXiv 2021.06*) Aomd: An analogy-aware approach to offensive meme detection on social media [[paper](https://arxiv.org/pdf/2106.11229)]
- (*Springer'23*) Multimodal visual-textual object graph attention network for propaganda detection in memes [[paper](https://link.springer.com/article/10.1007/s11042-023-15272-6)]
- (*arXiv 2021.03*) Learning Transferable Visual Models From Natural Language Supervision [[paper](https://arxiv.org/pdf/2103.00020)]
- (*ACL'24*) Meme-ingful Analysis: Enhanced Understanding of Cyberbullying in Memes Through Multimodal Explanations [[paper](https://aclanthology.org/2024.eacl-long.56.pdf)]
- (*AAAI'23*) Tot: Topology-aware optimal transport for multimodal hate detection [[paper](https://arxiv.org/pdf/2303.09314)]
- (*IJCNN'23*) Generative Models vs Discriminative Models: Which Performs Better in Detecting Cyberbullying in Memes? [[paper](https://ieeexplore.ieee.org/document/10191363)]

### Sentiment Analysis
- (*ICML'20*) MemeSem:A Multi-modal Framework for Sentimental Analysis of Meme via Transfer Learning [[paper](https://openreview.net/pdf?id=Okmqu6xqXK)]
- (*IEEE ICIIP*) Multi Modal Analysis of memes for Sentiment extraction [[paper](https://ieeexplore.ieee.org/abstract/document/9702696)]
- (*ACL'20*) Only text? only image? or both? Predicting sentiment of internet memes [[paper](https://aclanthology.org/2020.icon-main.60.pdf)]
- (*ACL'20*) Generative Models vs Discriminative Models: Which Performs Better in Detecting Cyberbullying in Memes? [[paper](https://aclanthology.org/2020.aacl-main.31.pdf)]
- (*ACL'22*) Findings of the Shared Task on Multimodal Sentiment Analysis and Troll Meme Classification in Dravidian Languages [[paper](https://aclanthology.org/2022.dravidianlangtech-1.39/)]
- (*IEEE ICCIT*) Explainable Multimodal Sentiment Analysis on Bengali Memes [[paper](https://ieeexplore.ieee.org/document/10441342)]
- (*IEEE UPCON*) Multimodal Meme Sentiment Analysis with Image Inpainting [paper](https://ieeexplore.ieee.org/document/9667557)]
- (*IEEE EECSI*) Sentiment Analysis of Text Memes: A Comparison Among Supervised Machine Learning Methods [[paper](https://ieeexplore.ieee.org/document/9946506)]
- (*IEEE ICMLA*) Is GPT Powerful Enough to Analyze the Emotions of Memes [[paper](https://ieeexplore.ieee.org/document/10459760)]
- (*Knowledge-Based Systems'24*) What do they “meme”? A metaphor-aware multi-modal multi-task framework for fine-grained meme understanding [[paper](https://www.sciencedirect.com/science/article/pii/S095070512400412X)]

### Memes Generation
- (*arXiv 2023.10*) On the proactive generation of unsafe images from text-to-image models using benign prompts [[paper](https://arxiv.org/pdf/2310.16613)]
- (*arXiv 2023.05*)Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models [[paper](https://arxiv.org/pdf/2305.13873)]
- (*IEEE SSIM*) Memer: News Meme Generator Using Gpt-3: Attention Of Young Generations To Current Affairs [[paper](https://ieeexplore.ieee.org/document/10469091)]
- (*IEEE Access*) Text Location-Aware Framework for Chinese Meme Generation [[paper](https://ieeexplore.ieee.org/document/10530875)]
- (*WWW'24*) MemeCraft: Contextual and Stance-Driven Multimodal Meme Generation [[paper](https://arxiv.org/abs/2403.14652)]
- (*IEEE'21*) Automatic Chinese Meme Generation Using Deep Neural Networks [[paper](https://ieeexplore.ieee.org/document/9611242)]
- (*arXiv 2020.04*) memeBot: Towards Automatic Image Meme Generation [[paper](https://arxiv.org/pdf/2004.14571)]
- (*AAAI'23*) What Do You MEME? Generating Explanations for Visual Semantic Role Labelling in Memes [[paper](https://arxiv.org/pdf/2212.00715)]
- (*arXiv 2021.12*) Multi-modal application: Image Memes Generation [[paper](https://arxiv.org/pdf/2112.01651)]
- (*COMAD'20*) Memeify: A Large-Scale Meme Generation System [[paper](https://dl.acm.org/doi/abs/10.1145/3371158.3371403)]
- (*Springer'23*) Generative Model of Suitable Meme Sentences for Images Using AutoEncoder [[paper](https://link.springer.com/content/pdf/10.1007/978-981-99-7019-3_23)]
- (*ResearchGate*) DeepHumor: Image-Based Meme Generation Using Deep Learning [[paper](https://www.researchgate.net/profile/Ilya-Borovik/publication/377272553_DeepHumor_Image-Based_Meme_Generation_Using_Deep_Learning/links/659e6eeeaf617b0d873b7e34/DeepHumor-Image-Based-Meme-Generation-Using-Deep-Learning.pdf|)]
- (*ICCC'24*) Computational Creativity in Meme Generation:A Multimodal Approach [[paper](https://computationalcreativity.net/iccc24/papers/ICCC24_paper_189.pdf)]
- (*Yanbu Journal of Engineering and Science*) Meme Generation Using Deep Neural Network to Engage Viewers on Social Media [[paper](https://www.semanticscholar.org/paper/Meme-Generation-Using-Deep-Neural-Network-to-Engage-Alandjani-Bouk/a36a98e4face807e4cbb9c052a38f6727b8c7808)]













