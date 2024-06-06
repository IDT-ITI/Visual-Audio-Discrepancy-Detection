# Visual-Audio-Discrepancy-Detection

This repository contains the materials (Python code, pre-trained models, extracted embedding, and dataset) for our paper "Visual and audio scene classification for detecting discrepancies in video: a baseline method and experimental protocol" [1].

## Introduction

Digital disinformation, the intentional dissemination of false or misleading information through digital media, encompasses various deceptive tactics, including fabricated news, tampered images, manipulated videos, and misleading narratives. Professionals across fields, such as journalists, security experts, and emergency management officers, are increasingly concerned about discerning genuine content from manipulated material. AI-based content verification tools and integrated toolboxes have emerged to assess the authenticity and integrity of digital content, allowing rapid multimedia analysis and aiding in prioritizing human effort.

In audio editing, incorporating the soundscape of an acoustic scene has become standard practice, since it helps to mask editing points and thus enhance immersion. However, when malicious users attempt to manipulate multimedia content, they are unlikely to have access to authentic audio soundscapes from the actual events. Instead, they may use pre-existing ambient sounds, which may not align with the visual content they are manipulating. As a result, the manipulated content produced by malicious users could contain inconsistencies between the audio and video modalities. These inconsistencies could range from subtle discrepancies in background noise to more obvious mismatches in environmental cues, leading to a loss of the video's credibility.

Our proposed method focuses on detecting disparities in audio-visual streams, such as incongruous audio tracks with visual scenes or conflicting environmental cues, which can indicate that a video has been fabricated. Our approach leverages visual- and audio-scene classification techniques to detect such discrepancies and provides a benchmark dataset to promote further research in this field.

## Visual-Audio-Discrepancy-Detection experimental protocol
The TAU Audio-Visual Urban Scenes 2021, used in Task 1B of the DCASE 2021’s challenge, involves scene estimation on categorizing videos based on their audiovisual content.

Aiming to leverage the wealth of visual and auditory data already available in the already existing TAU dataset we created the Visual-Audio-Discrepancy-Detection (VADD) dataset. This involves selecting videos where the visual content belongs to one category, but the audio is from another category. We selected videos from the TAU dataset and exchanged their audio tracks, resulting in a "manipulated" set, while keeping some videos unaltered forming a "pristine" set. These two sets comprise the VADD dataset. We provide both a 10-class and a 3-class version of the dataset, to accommodate different difficulty levels.

## Proposed Method
We propose a baseline method that can identify differences between visual and audio content in videos which we evaluate on the VADD dataset. Our methodology involves training a joint audio-visual scene classifier using both audio and visual modalities. We utilize pre-trained models for feature extraction:

- **Visual scene representations**:
  - ViT embeddings
  - CLIP embeddings
  - ResNet embeddings
- **Audio scene representations**:
  - OpenL3 embeddings
  - PANN embeddings
  - IOV embeddings

For classification, we aggregate these embeddings using a self-attention mechanism, followed by fully connected layers. Data augmentation techniques are employed to enrich the training dataset and improve generalization.

## Provided Materials
This repository includes the source code for the following tasks:
1. **Extracting embeddings** (1_extract_features.py): Code for extracting the necessary features from the visual and audio streams of videos of the TAU Audio-Visual Urban Scenes 2021 dataset.
2. **Experiments on audio-visual joint classifiers** (2_train_av_classifiers.py): Code for conducting experiments on joint classifiers that utilize both audio and visual data.
3. **Manipulated video detection and evaluation** (3_vadd_experiments.py): Code for evaluating the identified manipulated videos within the VADD dataset and evaluating.

The extracted embeddings for the TAU dataset videos are provided in the "features" folder. The best-performing visual-audio scene classification (VASC) model, as well as the separate visual- and audio-scene classification models (VSC and ASC respectively), are provided in the "models" folder. The VASC, VSC, and ASC models are evaluated in the table below.
The VADD dataset's split into "pristine" and "manipulated" sets of the TAU dataset videos is provided in the "VADD dataset" folder.

## Evaluation
We evaluated our proposed method on the TAU dataset for scene classification, comparing it with the winner of Task 1B of the DCASE 2021’s challenge,
and on the VADD dataset for detecting visual-audio discrepancies on both 3-class and 10-class variants of the VADD dataset.

### Scene Classification Results
<table>
  <thead>
    <tr>
      <th>Approach</th>
      <th>Accuracy (%)<br>on TAU dataset<br>using the 3-class variant</th>
      <th>Accuracy (%)<br>on TAU dataset<br>using the 10-class variant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VASC</td>
      <td>99.95</td>
      <td>97.24</td>
    </tr>
    <tr>
      <td>ASC</td>
      <td>99.84</td>
      <td>78.84</td>
    </tr>
    <tr>
      <td>VSC</td>
      <td>99.93</td>
      <td>94.32</td>
    </tr>
  </tbody>
</table>

### Visual-Audio Discrepancy Detection Results
<table>
  <thead>
    <tr>
      <th>VADD dataset<br>variant used</th>
      <th>F1-score (%)<br>of the proposed method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3-class</td>
      <td>95.54</td>
    </tr>
    <tr>
      <td>10-class</td>
      <td>79.16</td>
    </tr>
  </tbody>
</table>

## Prerequisites

To run this project, you will need to have the following packages installed:

* Python 3.8 or higher
* PyTorch 1.13 or higher
* OpenCV

There are other required packages, but to simplify the setup process, you can use the provided `environment.yml` file located in the "conda" folder to create a conda environment with all necessary dependencies.


## Citations

If you utilize any part of this repository in your research, please cite our paper 
[1] K. Apostolidis, J. Abesser, L. Cuccovillo, V. Mezaris, "Visual and audio scene classification for detecting discrepancies in video: a baseline method and experimental protocol", Proc. ACM Int. Workshop on Multimedia AI against Disinformation (MAD’24) at the ACM Int. Conf. on Multimedia Retrieval (ICMR’24), Thailand, June 2024.
The accepted version of the paper is also available on [arXiv](https://arxiv.org/abs/2405.00384).
```
@article{apostolidis2024visual,
  title={Visual and audio scene classification for detecting discrepancies in video: a baseline method and experimental protocol},
  author={Apostolidis, Konstantinos and Abesser, Jakob and Cuccovillo, Luca and Mezaris, Vasileios},
  journal={arXiv preprint arXiv:2405.00384},
  year={2024}
}
```

The "TAU Audio-Visual Urban Scenes 2021" dataset, which serves as the basis for our experimental protocol, is introduced in [2]:
```
@inproceedings{Wang2021_ICASSP,
  title={A Curated Dataset of Urban Scenes for Audio-Visual Scene Analysis},
  author={Wang, Shanshan and Mesaros, Annamaria and Heittola, Toni and Virtanen, Tuomas},
  booktitle={{IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
  pages={626--630},
  year={2021},
  organization={IEEE}
}
```

## Acknowledgment
This work was supported by the EU Horizon Europe programme under grant agreement 101070093 vera.ai.
