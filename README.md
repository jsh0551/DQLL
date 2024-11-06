# DQLL
---
![dqll drawio (2)](https://github.com/user-attachments/assets/198bbd4f-5ea0-4830-bcc9-fedafbb993ba)
![제목 없는 동영상 (3)](https://github.com/user-attachments/assets/cd16a325-b99e-488b-84e1-0b74f459da81) 

운동장 트랙 라인을 탐지하는 모델입니다.

**Deep Reinforcement Learning Based Lane Detection and Localization (DQLL)** 라는 논문의 [공개 코드](https://github.com/tuzixini/DQLL)를 참고하여 작성하였습니다.

트랙 라인을 탐지하여 바운딩 박스를 구하고 각 바운딩 박스의 landmark를 구하여 트랙 라인의 정확한 위치를 찾습니다.

localization 과정에 강화학습 모델을 사용하는 기존 DQLL과 달리, 회귀 모델을 사용하여 실제 테스트 단계에서도 잘 동작하도록 하였습니다.

## Preparation
### prerequisites
- [requirements.txt](https://github.com/zhangming8/yolox-pytorch/blob/main/requirements.txt)

### Model
- object detection model : [yolox](https://github.com/zhangming8/yolox-pytorch)

- localization model : mobilenetV3(default), ghostnet, efficientNetV2

### Dataset format
- object detection
  - COCO Dataset format
- localization
  - [MyTuSimpleLane format](https://github.com/tuzixini/DQLL?tab=readme-ov-file#tusimple-dataset) : TuSimple dataset 다운로드 후 genMyData.py 실행. 
  ```
  -- $DATAROOT
    |-- test_set
    |  -- clips
    |  -- test_tasks_0627.json
    |  -- readme.md
    |-- train_set
    |  -- clips
    |  -- label_data_0313.json
    |  -- label_data_0531.json
    |  -- label_data_0601.json
    |  -- readme.md
    |-- MyTuSimpleLane
    |  -- test
    |    -- bbox
    |      -- XXXXfiles
    |    -- DRL
    |    -- img
    |    -- mask
    |    -- mask_color
    |  -- train
    |    -- bbox
    |    -- DRL
    |    -- img
    |    -- mask
    |    -- mask_color
    |-- test_label.json
    |-- failList.json
    |-- meanImgTemp.npy
    |-- train_img_list.json
    |-- train_DRL_list.json
    |-- test_img_list.json
    |-- test_DRL_list.json
  ```

## Reference
[Deep Reinforcement Learning Based Lane Detection and Localization](https://www.sciencedirect.com/science/article/abs/pii/S0925231220310833)
