# Visual-Audio-Discrepancy-Detection

## Introduction
This repository contains an experimental protocal and a baseline method for detecting discrepancies between the audio and video modalities in multimedia content.

Digital disinformation, the intentional dissemination of false or misleading information through digital media, encompasses various deceptive tactics, including fabricated news, tampered images, manipulated videos, and misleading narratives. Professionals across fields, such as journalists, security experts, and emergency management officers, are increasingly concerned about discerning genuine content from manipulated material. AI-based content verification tools and integrated toolboxes have emerged to assess the authenticity and integrity of digital content, allowing rapid multimedia analysis and aiding in prioritizing human effort.

Our proposed method focuses on detecting subtle but crucial disparities in audio-visual streams that often go unnoticed. These disparities, such as incongruous audio tracks with visual scenes or conflicting environmental cues, can indicate that a video has been fabricated. Our approach leverages visual- and audio-scene classification techniques to detect such discrepancies and provides a benchmark dataset to promote further research in this field.

## Visual-Audio-Discrepancy-Detection Experimental Protocol

TAU Audio-Visual Urban Scenes 2021 that was used in Task 1B of the DCASE 2021’s challenge involves scene estimation on categorizing videos based on their A/V content.

Aiming to leverage the wealth of visual and auditory data already available in the already existing TAU dataset we created the Visual-Audio-Discrepancy-Detection dataset. This includes a subset of videos in which the visual content portrays one class, while the accompanying audio track is sourced from a different class. We selected a random set of videos and swap their audio streams, creating a "pristine" set and a "manipulated" set. In a way that is ensurend that these sets are balanced. This can enable the evaluation of methods that detect discrepancies between visual and audio streams. The VADD dataset includes a 10-class as well as a 3-class variant for varying difficulty levels.

## Proposed Method
Our methodology involves training a joint audio-visual classifier for scene classification using both audio and visual modalities. We utilize pre-trained models for feature extraction:

- **Visual scene representations**:
  - ViT embeddings
  - CLIP embeddings
  - ResNet embeddings
- **Audio scene representations**:
  - OpenL3 embeddings
  - PANN embeddings
  - IOV embeddings

For classification, we aggregate these embeddings using a self-attention mechanism, followed by fully connected layers. Data augmentation techniques are employed to enrich the training dataset and improve generalization.

## Provided Source Code

This repository includes the source code for the following tasks:

1. **Extracting Embeddings**: Code for extracting the necessary features from the visual and audio streams of videos of the TAU Audio-Visual Urban Scenes 2021 dataset.
2. **Experiments on Audio-Visual Joint Classifiers**: Code for conducting experiments on joint classifiers that utilize both audio and visual data.
3. **Manipulated Video Detection and Evaluation**: Code for identifying manipulated videos within the VADD dataset and evaluating the.


## Evaluation
We evaluated our proposed method on the TAU dataset for scene classification, comparing it with the winner of Task 1B of the DCASE 2021’s challenge,
and on the VADD dataset for detecting visual-audio discrepancies on both 3-class and 10-class variants of the VADD dataset.


### Scene Classification Results
<table>
  <thead>
    <tr>
      <th>Approach</th>
      <th>Accuracy (%) on {\taud} using the 3-class variant</th>
      <th>Accuracy (%) on {\taud} using the 10-class variant</th>
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
      <th>{\vadd} dataset variant used</th>
      <th>F1-score (%) of the proposed method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>3-class {\vadd}</td>
      <td>95.54</td>
    </tr>
    <tr>
      <td>10-class {\vadd}</td>
      <td>79.16</td>
    </tr>
  </tbody>
</table>

## Prerequisites

To run this project, you will need to have the following packages installed:

* Python 3.8 or higher
* PyTorch 1.13 or higher
* OpenCV

Additionally, there are other required packages. To simplify the setup process, you can use the provided `environment.yml` file located in the "conda" folder to create a conda environment with all necessary dependencies.


## Citations

If you utilize any part of this repository in your research, please cite our paper:
```
@article{apostolidis2024visual,
  title={Visual and audio scene classification for detecting discrepancies in video: a baseline method and experimental protocol},
  author={Apostolidis, Konstantinos and Abesser, Jakob and Cuccovillo, Luca and Mezaris, Vasileios},
  journal={arXiv preprint arXiv:2405.00384},
  year={2024}
}
```

The "TAU Audio-Visual Urban Scenes 2021" dataset, which serves as the basis for our experimental protocol, is introduced in:
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
