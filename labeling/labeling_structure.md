# 최종 데이터셋 구조 및 이미지-라벨 매칭 가이드

이 문서는 `/home/capstone2/labeling` 디렉터리에 가공된 3-Class 라벨링 파일들과 `/home/capstone2/zroact-stage2/benchmark2/data/viz_shufflenet_full`에 보관된 실제 프레임 이미지들 간의 구조적 관계를 명세하고, 학습 데이터셋 구축 시 이 둘을 정확하게 매칭하기 위한 규칙 및 코드 예제를 제공합니다.

---

## 1. 디렉터리 하위 구조 비교

### 1.1 라벨링 디렉터리 구조 (`/home/capstone2/labeling`)
전처리가 완료된 3-Class 라벨 데이터셋입니다. 1단계 컷오프(시퀀스 개수 제한)를 건너뛰어 모든 원본 시퀀스(총 259개)에 대한 정제된 `.txt` 라벨 파일이 보존되어 있습니다.

```text
/home/capstone2/labeling/
├── normal_plant/
│   ├── train.txt           # 해당 데이터셋 내 살아남은 라벨 프레임들의 상대 경로 리스트
│   ├── obj.names           # 클래스 정의 (unsafe, danger, normal)
│   ├── obj.data            # 메타데이터 정보 (classes = 3)
│   └── obj_train_data/
│       └── intrusion_normal_rgb_xxxx_cctvX/
│           ├── 30fps_frame_001.txt
│           └── ... (1프레임 단위의 모든 Bounding Box 라벨 파일)
├── plant_climb/
│   ├── train.txt
│   ├── ...
│   └── obj_train_data/
│       └── climb-over-fence_plant_rgb/
│           └── intrusion_climb-over-fence_rgb_xxxx_cctvX/ (1098번 제외 총 79개 시퀀스)
│               ├── 30fps_frame_001.txt
│               └── ... (3단계 Cutoff로 back 시점 이후 프레임은 삭제됨)
└── smart_climb/
    ├── train.txt
    ├── ...
    └── obj_train_data/
        └── climb-over-fence_smart_rgb/
            └── intrusion_climb-over-fence_rgb_xxxx_cctvX/ (총 80개 시퀀스)
                ├── 30fps_frame_001.txt
                └── ...
```

### 1.2 이미지 디렉터리 구조 (`/home/capstone2/zroact-stage2/benchmark2/data/viz_shufflenet_full`)
학습 및 벤치마크에 사용될 실제 추출 프레임 이미지 파일들이 보관된 경로입니다. 비디오 전체 프레임이 아닌 **10 sample rate** 단위로 추출된 이미지만 보관되어 있습니다.

```text
/home/capstone2/zroact-stage2/benchmark2/data/viz_shufflenet_full/
├── normal_plant_rgb/
│   ├── images/
│   │   └── intrusion_normal_rgb_xxxx_cctvX/
│   │       ├── intrusion_normal_rgb_xxxx_cctvX_t000001.jpg
│   │       ├── intrusion_normal_rgb_xxxx_cctvX_t000011.jpg
│   │       └── ... (10 sample rate 단위로 추출된 이미지 프레임)
│   └── labels/             # (비사용) 가공 전 원본 JSON 라벨링 데이터
├── climb-over-fence_plant_rgb/
│   ├── images/
│   │   └── intrusion_climb-over-fence_rgb_xxxx_cctvX/ (1098번 제외 총 79개 시퀀스)
│   │       ├── intrusion_climb-over-fence_rgb_xxxx_cctvX_t000016.jpg
│   │       └── ...
│   └── labels/
└── climb-over-fence_smart_rgb/
    ├── images/
    │   └── intrusion_climb-over-fence_rgb_xxxx_cctvX/ (총 80개 시퀀스)
    │       ├── intrusion_climb-over-fence_rgb_xxxx_cctvX_t000016.jpg
    │       └── ...
    └── labels/
```

---

## 2. 이미지-라벨 1:1 매칭 규칙

이미지 파일은 10 sample rate 추출로 인해 개수가 적고, 라벨 파일은 30fps 비디오 전체 프레임 단위로 조밀하게 채워져 있습니다. 따라서 **이미지 파일명을 기준으로 상응하는 라벨을 조회하여 매칭**해야 합니다.

### 2.1 매칭 규칙 명세
* **이미지 파일명 규칙**: `{seq_name}_t{NNN}.jpg` (예: `intrusion_climb-over-fence_rgb_0024_cctv3_t000016.jpg`)
  * 파일명 끝 부분의 `t` 뒤에 붙는 6자리 숫자 `NNN` (예: `000016` $\rightarrow$ `16`)이 **원래 비디오의 16번째 프레임 번호**를 의미합니다.
* **라벨 파일명 규칙**: `30fps_frame_{NNN:03d}.txt` (예: `30fps_frame_016.txt`)
  * 위에서 파싱한 프레임 번호 `16`을 3자리 포맷팅하여 `30fps_frame_016.txt` 파일과 매칭합니다.

### 2.2 실제 매칭 예시
| 데이터셋 분류 | 이미지 파일 경로 (viz_shufflenet_full) | 매칭되는 라벨 파일 경로 (labeling) |
| :--- | :--- | :--- |
| **smart_climb** | `.../climb-over-fence_smart_rgb/images/intrusion_climb-over-fence_rgb_0004_cctv1/intrusion_climb-over-fence_rgb_0004_cctv1_t000016.jpg` | `.../labeling/smart_climb/obj_train_data/intrusion_climb-over-fence_rgb_0004_cctv1/30fps_frame_016.txt` |
| **plant_climb** | `.../climb-over-fence_plant_rgb/images/intrusion_climb-over-fence_rgb_0024_cctv3/intrusion_climb-over-fence_rgb_0024_cctv3_t000216.jpg` | `.../labeling/plant_climb/obj_train_data/climb-over-fence_plant_rgb/intrusion_climb-over-fence_rgb_0024_cctv3/30fps_frame_216.txt` |
| **normal_plant** | `.../normal_plant_rgb/images/intrusion_normal_rgb_0166_cctv1/intrusion_normal_rgb_0166_cctv1_t000111.jpg` | `.../labeling/normal_plant/obj_train_data/intrusion_normal_rgb_0166_cctv1/30fps_frame_111.txt` |

---

## 3. 다중 라벨 학습용 데이터셋 로더 파이썬 코드

학습용 Dataset 클래스를 설계할 때 아래 코드를 활용하면, 물리적인 이미지 경로와 1098번 제외 및 3단계 Cutoff로 지워진 라벨 프레임을 예외 처리하여 안전하게 이미지-라벨 정답 쌍을 매칭할 수 있습니다.

```python
import os
import re
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class ZroActMatchingDataset(Dataset):
    def __init__(self, labeling_base_dir, viz_base_dir, dataset_type='plant_climb', transform=None):
        """
        Args:
            labeling_base_dir: 'labeling' 디렉터리 절대 경로 (예: '/home/capstone2/labeling')
            viz_base_dir: 'viz_shufflenet_full' 디렉터리 절대 경로 (예: '/home/capstone2/zroact-stage2/benchmark2/data/viz_shufflenet_full')
            dataset_type: 'smart_climb', 'plant_climb', 또는 'normal_plant'
            transform: 이미지 전처리 transforms
        """
        self.transform = transform
        self.samples = []
        
        # 1. 디렉터리 매핑
        type_mapping = {
            'smart_climb': {
                'label_subdir': 'smart_climb/obj_train_data',
                'img_subdir': 'climb-over-fence_smart_rgb/images'
            },
            'plant_climb': {
                'label_subdir': 'plant_climb/obj_train_data/climb-over-fence_plant_rgb',
                'img_subdir': 'climb-over-fence_plant_rgb/images'
            },
            'normal_plant': {
                'label_subdir': 'normal_plant/obj_train_data',
                'img_subdir': 'normal_plant_rgb/images'
            }
        }
        
        cfg = type_mapping[dataset_type]
        label_dir = Path(labeling_base_dir) / cfg['label_subdir']
        img_dir = Path(viz_base_dir) / cfg['img_subdir']
        
        # 2. 이미지 파일을 순회하며 매칭되는 라벨 탐색
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = Path(root) / f
                    seq_name = img_path.parent.name
                    
                    # 1098번 시퀀스는 라벨이 없으므로 자동 제외
                    if seq_name == 'intrusion_climb-over-fence_rgb_1098_cctv4':
                        continue
                        
                    # 이미지 파일명에서 프레임 번호(tXXXXXX) 추출
                    match = re.search(r'_t(\d+)\.(jpg|jpeg|png)$', f)
                    if not match:
                        continue
                    
                    frame_num = int(match.group(1))
                    
                    # 대응하는 라벨 텍스트 파일명 빌드
                    label_filename = f"30fps_frame_{frame_num:03d}.txt"
                    label_path = label_dir / seq_name / label_filename
                    
                    # 3단계 Cutoff 및 이탈로 인해 라벨 파일이 삭제된 프레임인 경우 자동 건너뜀
                    if label_path.exists():
                        self.samples.append((str(img_path), str(label_path)))
                        
        print(f"[{dataset_type}] Successfully matched {len(self.samples)} image-label pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # YOLO 형태의 라벨 파싱 (class_id x_center y_center width height)
        labels = []
        with open(label_path, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    labels.append((class_id, bbox))
                    
        if self.transform:
            image = self.transform(image)
            
        return image, labels

# 사용 예시:
# dataset = ZroActMatchingDataset(
#     labeling_base_dir='/home/capstone2/labeling',
#     viz_base_dir='/home/capstone2/zroact-stage2/benchmark2/data/viz_shufflenet_full',
#     dataset_type='plant_climb'
# )
```

---

## 4. 최종 정답 클래스 정보 (정답 매핑 가이드)

YOLO 바운딩 박스 정보 외에, 프레임이 가진 최종 정답 상태(Ground Truth)를 추출하여 분류 및 예측 정답으로 활용할 수 있습니다.

* **Class `0` (`unsafe`)**: 작업자 불안전 상태 (작업자 크기 바운딩 박스)
* **Class `1` (`danger`)**: 침입 상태 (울타리를 넘어감 - 펜스 크기 바운딩 박스)
* **Class `2` (`normal`)**: 정상 상태 (위 0, 1번이 없는 상태로 전체 화면 영역 `2 0.5 0.5 1.0 1.0` 고정값)

### 프레임 정답 구성 예시
1. **정상 상태 프레임**: 라벨 파일 내에 `2 0.5 0.5 1.0 1.0`만 존재 $\rightarrow$ **`['normal']`**
2. **작업자 불안전 상태만 감지된 프레임**: 라벨 파일 내에 `0` 클래스 라인만 존재 $\rightarrow$ **`['unsafe']`**
3. **불안전 상태와 침입 상태가 모두 감지된 프레임**: 라벨 파일 내에 `0`과 `1`이 모두 존재 $\rightarrow$ **`['unsafe', 'danger']`** (다중 라벨 분류 활용 가능)
