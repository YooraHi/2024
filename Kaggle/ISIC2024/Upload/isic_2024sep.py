import os
import torch
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from io import BytesIO
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import warnings

# albumentations 버전 확인 경고 비활성화
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
warnings.filterwarnings("ignore", category=UserWarning, module='albumentations')

# 설정
train_image_dir = '/kaggle/input/isic-2024-challenge/train-image/image'
train_meta_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
test_image_hdf5 = '/kaggle/input/isic-2024-challenge/test-image.hdf5'
test_meta_path = '/kaggle/input/isic-2024-challenge/test-metadata.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 설정
N_WORKERS = 4
BATCH_SIZE = 64  # 배치 크기를 늘려 학습 시간 단축
LR = 1e-4
WD = 1e-5
EPOCHS = 5
EARLY_STOPPING_EPOCH = 10
IN_CHANS = 3
N_CLASSES = 1
MODEL_NAME = 'resnet18'

# 데이터 로드 및 정규화
train_df = pd.read_csv(train_meta_path, low_memory=False)
test_df = pd.read_csv(test_meta_path, low_memory=False)

# 레이블 처리
train_df['tbp_lv_nevi_confidence'] = (train_df['tbp_lv_nevi_confidence'] > 0.5).astype(int)

# 클래스 불균형 확인 및 pos_weight 계산
class_counts = train_df['tbp_lv_nevi_confidence'].value_counts()
num_negatives = class_counts.get(0, 0)
num_positives = class_counts.get(1, 0)
pos_weight = torch.tensor([num_negatives / num_positives], device=device)

# 데이터 증강 설정
IMAGE_SIZE = 224
transforms_train = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

transforms_val = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class ISICDataset(Dataset):
    def __init__(self, df, image_hdf5_path=None, image_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_hdf5_path = image_hdf5_path
        self.image_dir = image_dir
        self.valid_indices = []

        if image_hdf5_path:
            self.hdf5_file = h5py.File(image_hdf5_path, 'r')
        else:
            self.hdf5_file = None

        for idx in range(len(self.df)):
            isic_id = str(self.df.iloc[idx]['isic_id'])
            if self.hdf5_file:
                data = self.hdf5_file.get(isic_id, None)
                if data is not None:
                    self.valid_indices.append(idx)
            else:
                extensions = ['.jpg', '.png', '.jpeg']
                for ext in extensions:
                    image_path = os.path.join(self.image_dir, f"{isic_id}{ext}")
                    if os.path.exists(image_path):
                        self.valid_indices.append(idx)
                        break

        if len(self.valid_indices) == 0:
            raise ValueError("데이터셋에 유효한 데이터가 없습니다.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        isic_id = str(self.df.iloc[valid_idx]['isic_id'])

        img = None
        if self.hdf5_file:
            data = self.hdf5_file[isic_id]
            binary_data = data[()]
            try:
                img = Image.open(BytesIO(binary_data)).convert('RGB')
            except Exception:
                return None, None
        else:
            extensions = ['.jpg', '.png', '.jpeg']
            for ext in extensions:
                image_path = os.path.join(self.image_dir, f"{isic_id}{ext}")
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert('RGB')
                    break
            if img is None:
                return None, None

        if self.transform:
            img = self.transform(image=np.array(img))['image']

        if 'tbp_lv_nevi_confidence' in self.df.columns:
            label = self.df.iloc[valid_idx]['tbp_lv_nevi_confidence']
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(0, dtype=torch.float32)

        return img, label

    def close(self):
        if self.hdf5_file:
            self.hdf5_file.close()

    @staticmethod
    def custom_collate_fn(batch):
        batch = [item for item in batch if item[0] is not None and item[1] is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.default_collate(batch)

# 데이터셋 및 데이터로더 생성
train_ds = ISICDataset(train_df, image_dir=train_image_dir, transform=transforms_train)
val_ds = ISICDataset(train_df, image_dir=train_image_dir, transform=transforms_val)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True,
                      num_workers=N_WORKERS, collate_fn=ISICDataset.custom_collate_fn)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False,
                    num_workers=N_WORKERS, collate_fn=ISICDataset.custom_collate_fn)

# 모델 정의
class Maxout(nn.Module):
    def __init__(self, in_features, out_features, pool_size=2):
        super(Maxout, self).__init__()
        self.fc = nn.Linear(in_features, out_features * pool_size)
        self.pool_size = pool_size
        self.out_features = out_features

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.out_features, self.pool_size)
        x, _ = torch.max(x, dim=2)
        return x

class ISICModel(nn.Module):
    def __init__(self, model_name, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_c,
            num_classes=0,
            global_pool=''
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        num_features = self.model.num_features
        self.adabin_fc = Maxout(num_features, n_classes, pool_size=2)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.adabin_fc(x)
        return x

model = ISICModel(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=False)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
model.apply(weights_init)

model.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# GradScaler의 조건부 사용 및 FutureWarning 해결
if device.type == 'cuda':
    scaler = torch.amp.GradScaler()
else:
    scaler = None

def calculate_pAUC(labels, predictions, lower_bound=0, upper_bound=0.2):
    fpr, tpr, _ = roc_curve(labels, predictions)
    mask = (fpr >= lower_bound) & (fpr <= upper_bound)
    if np.sum(mask) < 2:
        return 0.0
    p_auc = auc(fpr[mask], tpr[mask])
    return p_auc

best_pAUC = 0.0
early_stopping_counter = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')

    model.train()
    running_loss = 0.0

    for batch in tqdm(train_dl, leave=True):
        if batch is None:
            continue
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type=device.type):
                outputs = model(imgs)
                loss = criterion(outputs.squeeze(1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_dl.dataset)
    print(f'Train Loss: {epoch_loss:.4f}')

    val_preds = []
    val_labels = []
    with torch.no_grad():
        model.eval()
        for batch in val_dl:
            if batch is None:
                continue
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)

            val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten().tolist())
            val_labels.extend(labels.cpu().numpy().tolist())

    current_pAUC = calculate_pAUC(val_labels, val_preds)

    print(f'Validation pAUC: {current_pAUC:.4f}')

    if current_pAUC > best_pAUC:
        best_pAUC = current_pAUC
        torch.save(model.state_dict(), 'best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_EPOCH:
            print("Early stopping triggered.")
            break

    scheduler.step()

# 테스트 데이터셋 및 데이터로더 생성
test_ds = ISICDataset(test_df, image_hdf5_path=test_image_hdf5, transform=transforms_val)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False,
                     num_workers=N_WORKERS, collate_fn=ISICDataset.custom_collate_fn)

# 모델 로드 및 추론
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()
test_predictions = []

with torch.no_grad():
    for batch in tqdm(test_dl, leave=True):
        if batch is None:
            continue
        imgs, _ = batch
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_predictions.extend(preds.flatten().tolist())

submission_df = pd.DataFrame({
    'isic_id': test_df['isic_id'].tolist(),
    'target': test_predictions
})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file created successfully.")

best_model_path = "/kaggle/working/best_model.pth"
state_db_path = "/kaggle/working/state.db"

for file_path in [best_model_path, state_db_path]:
    if os.path.exists(file_path):
        os.remove(file_path)
print("모든 작업이 완료되었습니다.")

best_model_path = "/kaggle/working/best_model.pth"
state_db_path = "/kaggle/working/state.db"

for file_path in [best_model_path, state_db_path]:
    if os.path.exists(file_path):
        os.remove(file_path)
print("모든 작업이 완료되었습니다.")