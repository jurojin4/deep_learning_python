# Change Log
All notable changes to this repository will be documented in this file.

## 2026-05-06 Kaggle corrections again

### Fixed
- Forget to add the model to gpu devices after DataParallelism.

## 2026-05-06 Kaggle corrections

### Fixed
- When DataParallel is used the model can be accessed by using the attribute 'module'

## 2026-05-06 - Some corrections

### Fixed
- batch_size not attributed in trainers

## 2026-04-30 - Embellishment

### Added
- Data Augmentation
- Video Generation with YOLO predictions
- Data Parallelism

### Fixed
- Mean Average Precision corrected


## 2026-03-02 - YOLOV3 Implementation

### Added
- YOLOV3 implementation
- Realtime computer vision for YOLOV1 and YOLOV3

## 2026-02-06 - Reorganization

### Added
- main files for computer vision (trainer.py, metrics.py, dataset.py).

### Deleted
- metrics.py file in yolov1 repository.

## 2026-02-04 - Feature: Face Detection function for dataset usage

### Added
- **Face Detection** dataset function added
- Correction applied again in README.md

## 2026-02-01 - Initial Commit: correction

### Fixed
- **framework_scratch/documentations** added for visual explanations in README.md.
- Correction applied in README.md (Missing close brace, paper link, YOLOV1 loss function etc.).

## 2026-02-01 - Initial Commit

### Added
- **framework_scratch Directory**:<br>
The directory contains a Pytorch-like framework from scratch, implemented in Python with
NumPy for tensor operations and autograd (broadcasting included in it).

- **computer_vision/yolo/yolov1**:<br>
The directory contains a Pytorch implementation of **YOLOV1** from scratch.