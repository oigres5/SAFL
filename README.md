# Synthetic and Authentic Forensic Lab (SAFL)

This repository contains the datasets and materials associated with the article:

**Sergio A. Falcón-López, et al.**  
*Forensic Analysis of Manipulated Images and Videos*  
Submitted to *Applied Sciences (MDPI)*, 2025.

## Relation to the Article
The SAFL dataset was created specifically for the experiments described in the article.  
It includes:
- Authentic and synthetic (Deepfake) images and videos.
- The exact subsets used for the evaluation of conventional forensic tools and modern Deepfake-detection models.
- File structures and naming conventions referenced in the manuscript tables and figures.

This repository enables full reproducibility of the experiments and verification of the reported metrics.

## Author and Ownership
The dataset and repository are authored by:

**Sergio A. Falcón-López**  
ORCID: *https://orcid.org/0009-0002-7106-6691*  
Email: *sfalcon@scc.uned.es*  

GitHub username: **oigres5**

## Repository Structure

The repository is organized as follows:

- `audios/`: Contains both real and deepfake-generated audio samples.
  - `real/`: Original, non-manipulated audio files.
  - `fake/`: Deepfake-generated audio files.

- `images/`: Contains real and deepfake-generated images.
  - `real/`: Original images used in the experiments.
  - `fake/`: Deepfake-generated images.

- `videos/`: Directory with real and deepfake videos.
  - `real/`: Original source videos.
  - `fake/`: Deepfake-generated videos.

- `src/`: Contains modified or adapted scripts based on external tools.

- `README.md`: This documentation file.


## Dataset Description
**Images**

 -Real: 2102 samples

  - These samples were selected from CelebA dataset.

 -Deepfake: 2095 (+300 Updated 06/2025)samples

  - 1009 samples generated from [https://thispersondoesnotexist.com](https://thispersondoesnotexist.com/)
  -  613 samples generate with [FaceApp](https://play.google.com/store/apps/details?id=io.faceapp&pcampaignid=web_share)
  -  453 samples from generated videos with DeepFaceLab
  -   20 samples by Dall-E2
  -  300 samples by Dall-E3 (Updated 06/2025)
	
		
	
**Videos**

 -Real: 212 samples

  - These samples were selected from Celeb-DF dataset.

 -Deepfake: 204 samples
  - 100 generated with Avatarify
  - 104 generated with DeepFaceLab
	
	
**Audios**

 -Real: 2000 samples in Spanish language extracted from LibriVox project audiobooks
	
 -Deepfake: 2000 samples generated with Text-To-Speech (TTS) method
 

## System Requirements

The `src/` directory contains helper scripts that wrap or slightly adapt existing forensic and Deepfake-related tools from external projects.  
The complete and up-to-date software requirements for each tool are documented in their original repositories, for example:

- **ManTraNet** – https://github.com/ISICV/ManTraNet  
- **Image Forgery Detection with CNN** – https://github.com/kPsarakis/Image-Forgery-Detection-CNN  
- **Mesonet** – https://github.com/DariusAf/MesoNet  

In our experiments, all scripts in `src/` were executed under the following minimum environment:

### Software
- Operating system: Linux (Ubuntu 20.04+) or Windows 10/11  
- Python: 3.8 or higher  
- Git: 2.25 or higher  
- CUDA 11+ and NVIDIA drivers (only required for GPU-accelerated training/inference)  

### Hardware
- CPU: Quad-core processor or better  
- RAM: at least 8 GB (16 GB recommended for video processing)  
- Disk space: at least 20 GB free for datasets and intermediate results  
- GPU (optional but recommended): NVIDIA GPU with ≥ 4 GB VRAM for DeepFaceLab-based workflows 
 
## Versioning and Commit Transparency

Past changes to the dataset have been reviewed and documented in this repository.
All future commits will include:
- Clear descriptions of the changes performed.
- Exact counts of affected files.
- Justification whenever aggregate statistics or percentages change.

## Dataset Version History

- 2025-06-07  Initial public release of the SAFL dataset (v1.0).
  (Git commit: `3f6f20088360acfdcfbdde5186d4c7f67c775f89` on branch `main`).

##  License
This dataset is released under the following license:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**  
https://creativecommons.org/licenses/by/4.0/

This allows reuse with proper citation.

##  How to Cite
If you use this dataset, please cite:

Sergio A. Falcón-López, Synthetic and Authentic Forensic Lab (SAFL) Dataset. https://github.com/oigres5/SAFL