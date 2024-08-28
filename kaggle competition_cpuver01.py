import torch
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.metrics import roc_auc_score, roc_curve, auc

# 설정
train_image_dir = r'C:\Users\CHOIWOOJEONG\Documents\Kaggle\isic-2024-challenge\train-image\image'
train_meta_path = r'C:\Users\CHOIWOOJEONG\Documents\Kaggle\isic-2024-challenge\train-metadata.csv'
test_image_dir = r'C:\Users\CHOIWOOJEONG\Documents\Kaggle\isic-2024-challenge\train-image\image'
test_meta_path = r'C:\Users\CHOIWOOJEONG\Documents\Kaggle\isic-2024-challenge\test-metadata.csv'
device = 'cpu'
N_WORKERS = 1
BATCH_SIZE = 16
LR = 1e-4
WD = 1e-2
EPOCHS = 3
EARLY_STOPPING_EPOCH = 10
IN_CHANS = 3
N_CLASSES = 1  # 이진 분류이므로 1로 설정
MODEL_NAME = 'resnet18'

# 데이터 로드 및 정규화
train_df = pd.read_csv(train_meta_path, low_memory=False)

# 레이블 데이터 형태 확인
if not pd.api.types.is_bool_dtype(train_df['tbp_lv_nevi_confidence']):
    train_df['tbp_lv_nevi_confidence'] = (train_df['tbp_lv_nevi_confidence'] > 0.5).astype(int)

# 데이터 전처리 설정
transforms_val = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0)
])

transforms_train = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0)
])

# ISICDataset 클래스 정의
class ISICDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        isic_id = str(self.df.iloc[idx]['isic_id'])
        
        extensions = ['.jpg', '.png', '.jpeg']
        img = None
        for ext in extensions:
            image_path = os.path.join(self.image_dir, f"{isic_id}{ext}")
            if os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                break
            
            if img is None:
                raise FileNotFoundError(f"No image found for {isic_id} with supported extensions.")
        
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img.astype(np.uint8))['image']
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)
        label = self.df.iloc[idx]['tbp_lv_nevi_confidence']
        return img, label
    

if __name__ == "__main__":

    # 데이터셋과 DataLoader 설정
    train_ds = ISICDataset(train_df, image_dir=train_image_dir, transform=transforms_train)
    val_ds = ISICDataset(train_df, image_dir=train_image_dir, transform=transforms_val)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=N_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False, num_workers=N_WORKERS)

    # AdaBin과 Maxout을 포함한 모델 정의
    class Maxout(nn.Module):
        def __init__(self, in_features, out_features, pool_size=2):
            super(Maxout, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.pool_size = pool_size
            self.fc = nn.Linear(in_features, out_features * pool_size)

        def forward(self, x):
            x = self.fc(x)
            x = x.view(x.size(0), self.out_features, self.pool_size)
            x, _ = torch.max(x, dim=2)
            return x

    class ISICModel(nn.Module):
        def __init__(self, model_name, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=True):
            super().__init__()
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_c,
                num_classes=0,
                global_pool=''  # 여기에서 pooling을 제거하고 수동으로 추가
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 512 채널을 1x1로 줄이는 풀링 레이어 추가
            self.adabin_fc = Maxout(512, n_classes, pool_size=2)  # Maxout 레이어

        def forward(self, x):
            x = self.model.forward_features(x)
            x = self.pool(x)  # Global Average Pooling 적용
            x = x.view(x.size(0), -1)  # Flatten
            x = self.adabin_fc(x)
            return x

    # 모델 초기화 및 학습
    model = ISICModel(MODEL_NAME, in_c=IN_CHANS, n_classes=N_CLASSES, pretrained=True)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # pAUC 계산 함수
    def calculate_pAUC(labels, predictions, lower_bound=0, upper_bound=0.2):
        fpr, tpr, _ = roc_curve(labels, predictions)
        mask = (fpr >= lower_bound) & (fpr <= upper_bound)
        p_auc = auc(fpr[mask], tpr[mask])
        return p_auc

    best_pAUC = 0.0
    early_stopping_counter = 0

#train loop

for epoch in range(EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}')
    
    model.train()
    running_loss = 0.0
    
    
for imgs, labels in tqdm(train_dl, leave=True):
        imgs = imgs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs.squeeze(), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        
        
epoch_loss = running_loss / len(train_dl.dataset)
print(f'Train Loss: {epoch_loss:.4f}')
    
# Validation loop to calculate pAUC
val_preds = []
val_labels = []
with torch.no_grad():
    model.eval()
    for imgs, labels in val_dl:
        imgs = imgs.to(device)
        labels = labels.to(device).float()
        outputs = model(imgs)
        val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten().tolist())
        val_labels.extend(labels.cpu().numpy().tolist())

current_pAUC = calculate_pAUC(val_labels, val_preds)

if current_pAUC > best_pAUC:
    best_pAUC = current_pAUC
    torch.save(model.state_dict(), 'best_model.pth')
    early_stopping_counter = 0
else:
    early_stopping_counter += 1
    if early_stopping_counter >= EARLY_STOPPING_EPOCH:
        print("Early stopping triggered.")
        break
            
    # 테스트루프 파일명 'best_model.pth'
            

    # 예측 결과 저장
model.load_state_dict(torch.load('best_model.pth'))

test_df = pd.read_csv(test_meta_path)
test_ds = ISICDataset(test_df, image_dir=test_image_dir, transform=transforms_val)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False, num_workers=N_WORKERS)

model.eval()
test_predictions = []

with torch.no_grad():
    
    for imgs, _ in tqdm(test_dl, leave=True):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        test_predictions.extend(preds.flatten().tolist())

    # 제출 파일 생성
submission_df = pd.DataFrame({
'isic_id': test_df['isic_id'],
'prediction': test_predictions
})

submission_df.to_csv(r'C:\Users\CHOIWOOJEONG\Documents\Kaggle\submission.csv', index=False)