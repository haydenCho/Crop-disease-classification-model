import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter

# 안내 메세지 없애기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 하이퍼파라미터 설정
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.root_dir = root_dir
        self.label_map = label_map  # 라벨 맵을 인자로 받음
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        filename = os.path.basename(path).split('_')

        # 라벨 추출
        disease_label = int(filename[4])  # 다섯 번째 항목이 질병 번호
        crop_label = int(filename[5])     # 여섯 번째 항목이 작물 번호

        # 라벨을 맵핑하여 처리
        crop_label = torch.tensor(self.label_map['crop'][crop_label], dtype=torch.long)  
        disease_label = torch.tensor(self.label_map['disease'][disease_label], dtype=torch.long)

        # 이미지 로드 및 변환
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image, crop_label, disease_label

# 모델 정의
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_crops, num_diseases):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone  # ResNet-34 backbone
        self.n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 마지막 fully connected layer 제거
        
        # Task-specific heads
        self.crop_head = nn.Linear(self.n_features, num_crops)          # 작물 클래스
        self.disease_head = nn.Linear(self.n_features, num_diseases)    # 질병 클래스
    
    def forward(self, x):
        features = self.backbone(x)  # Shared backbone
        crop_output = self.crop_head(features)  # Crop classification output
        disease_output = self.disease_head(features)  # Disease classification output
        return crop_output, disease_output

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(true_labels, pred_labels, classes, title, save_path=None):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))

    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은고딕
    plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 지정된 경로에 이미지 저장
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# 메인 실행 코드
if __name__ == '__main__':
    # 라벨 맵 정의
    label_map = {
        'crop': {1: 0, 2: 1, 3: 2, 6: 3, 9: 4},  # 숫자와 인덱스를 매핑
        'disease': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 11: 7, 12: 8, 16: 9, 17: 10, 18: 11}
    }

    # 데이터셋 준비
    train_dataset = CustomDataset(root_dir='./mtl_dataset/Training', label_map=label_map)
    val_dataset = CustomDataset(root_dir='./mtl_dataset/Validation', label_map=label_map)
    test_dataset = CustomDataset(root_dir='./mtl_dataset/Test', label_map=label_map)

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 생성
    num_crops = 5  # 작물 클래스 수
    num_diseases = 12  # 질병 클래스 수
    backbone = resnet34(weights=ResNet34_Weights.DEFAULT)  # Pretrained ResNet-34
    model = MultiTaskModel(backbone, num_crops, num_diseases).to(device)

    # 각 클래스별 샘플 개수 계산
    crop_counts = Counter()
    disease_counts = Counter()

    # 데이터셋 순회하며 라벨 수집
    for i in range(len(train_dataset)):
        _, crop_label, disease_label = train_dataset[i]  # __getitem__ 호출
        crop_counts[crop_label] += 1
        disease_counts[disease_label] += 1

    # 클래스별 샘플 개수 출력
    print("Crop class counts:", dict(crop_counts))
    print("Disease class counts:", dict(disease_counts))

    # 클래스별 가중치 계산
    total_crop_samples = sum(crop_counts.values())
    total_disease_samples = sum(disease_counts.values())

    crop_class_weights = torch.tensor(
        [total_crop_samples / crop_counts.get(c, 1) for c in range(num_crops)],
        dtype=torch.float,
    ).to(device)

    disease_class_weights = torch.tensor(
        [total_disease_samples / disease_counts.get(d, 1) for d in range(num_diseases)],
        dtype=torch.float,
    ).to(device)

    print("Crop class weights:", crop_class_weights)
    print("Disease class weights:", disease_class_weights)

    # 각 태스크별로 가중치 적용된 손실 함수 정의
    criterion_crop = nn.CrossEntropyLoss(weight=crop_class_weights)
    criterion_disease = nn.CrossEntropyLoss(weight=disease_class_weights)

    # 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard 설정
    writer = SummaryWriter('./runs_mtl/weight_mtl/')

    best_val_disease_acc = 0
    best_model = None

    for epoch in range(num_epochs):
        # ====== Training ======
        model.train()
        running_loss = 0.0
        correct_crop = 0
        correct_disease = 0
        total = 0

        for images, crop_labels, disease_labels in train_loader:
            images = images.to(device, non_blocking=True)
            crop_labels = crop_labels.to(device, non_blocking=True)
            disease_labels = disease_labels.to(device, non_blocking=True)

            # Optimizer 초기화
            optimizer.zero_grad()

            # Forward 및 손실 계산
            crop_outputs, disease_outputs = model(images)
            crop_loss = criterion_crop(crop_outputs, crop_labels)
            disease_loss = criterion_disease(disease_outputs, disease_labels)
            total_loss = 0.5 * crop_loss + 0.5 * disease_loss

            # Backward 및 Optimizer 업데이트
            total_loss.backward()
            optimizer.step()

            # 손실 및 정확도 계산
            running_loss += total_loss.item()
            _, crop_predicted = torch.max(crop_outputs, 1)
            _, disease_predicted = torch.max(disease_outputs, 1)
            total += crop_labels.size(0)
            correct_crop += (crop_predicted == crop_labels).sum().item()
            correct_disease += (disease_predicted == disease_labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_crop_acc = 100 * correct_crop / total
        train_disease_acc = 100 * correct_disease / total

        # ====== Validation ======
        model.eval()
        val_loss = 0.0
        correct_crop = 0
        correct_disease = 0
        total = 0

        with torch.no_grad():
            for images, crop_labels, disease_labels in val_loader:
                images = images.to(device, non_blocking=True)
                crop_labels = crop_labels.to(device, non_blocking=True)
                disease_labels = disease_labels.to(device, non_blocking=True)

                # Forward 및 손실 계산
                crop_outputs, disease_outputs = model(images)
                crop_loss = criterion_crop(crop_outputs, crop_labels)
                disease_loss = criterion_disease(disease_outputs, disease_labels)
                val_loss += (0.5 * crop_loss + 0.5 * disease_loss).item()

                # 정확도 계산
                _, crop_predicted = torch.max(crop_outputs, 1)
                _, disease_predicted = torch.max(disease_outputs, 1)
                total += crop_labels.size(0)
                correct_crop += (crop_predicted == crop_labels).sum().item()
                correct_disease += (disease_predicted == disease_labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_crop_acc = 100 * correct_crop / total
        val_disease_acc = 100 * correct_disease / total

        # TensorBoard에 기록
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Crop_Accuracy', train_crop_acc, epoch)
        writer.add_scalar('Train/Disease_Accuracy', train_disease_acc, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Crop_Accuracy', val_crop_acc, epoch)
        writer.add_scalar('Validation/Disease_Accuracy', val_disease_acc, epoch)

        # 출력
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Crop Acc: {train_crop_acc:.2f}%, Train Disease Acc: {train_disease_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Crop Acc: {val_crop_acc:.2f}%, Val Disease Acc: {val_disease_acc:.2f}%')

        # 모델 저장
        if val_disease_acc > best_val_disease_acc:
            best_val_disease_acc = val_disease_acc
            best_model = model.state_dict()
    
    torch.save(best_model, 'best_weight_mtl_model.pth')
    writer.close()

    # 모델 평가 및 혼동 행렬 생성
    model.load_state_dict(torch.load('best_weight_mtl_model.pth', weights_only=True))
    model.eval()

    correct_crop = 0
    correct_disease = 0
    total = 0

    pred_crop_labels = []
    pred_disease_labels = []
    true_crop_labels = []
    true_disease_labels = []

    with torch.no_grad():
        for images, crop_labels, disease_labels in test_loader:
            images = images.to(device, non_blocking=True)
            crop_labels = crop_labels.to(device, non_blocking=True)
            disease_labels = disease_labels.to(device, non_blocking=True)

            crop_outputs, disease_outputs = model(images)
            _, crop_predicted = torch.max(crop_outputs, 1)
            _, disease_predicted = torch.max(disease_outputs, 1)

            total += crop_labels.size(0)
            correct_crop += (crop_predicted == crop_labels).sum().item()
            correct_disease += (disease_predicted == disease_labels).sum().item()

            pred_crop_labels.extend(crop_predicted.cpu().numpy())
            pred_disease_labels.extend(disease_predicted.cpu().numpy())
            true_crop_labels.extend(crop_labels.cpu().numpy())
            true_disease_labels.extend(disease_labels.cpu().numpy())

    test_crop_acc = 100 * correct_crop / total
    test_disease_acc = 100 * correct_disease / total

    print(f"Test Crop Accuracy: {test_crop_acc:.2f}%")
    print(f"Test Disease Accuracy: {test_disease_acc:.2f}%")

    # 혼동 행렬 시각화
    plot_confusion_matrix(
        true_crop_labels, 
        pred_crop_labels, 
        classes=['고추', '무', '배추', '오이', '파'], 
        title="Crop Classification Confusion Matrix",
        save_path='./weight_mtl/crop_confusion_matrix.png'
    )
    plot_confusion_matrix(
        true_disease_labels, 
        pred_disease_labels, 
        classes=['정상', '고추탄저병', '고추흰가루병', '무검은무늬병', '무노균병', '배추검음썩음병', '배추노균병', '오이노균병', '오이흰가루병', '파검은무늬병', '파노균병', '파녹병'], 
        title="Disease Classification Confusion Matrix",
        save_path='./weight_mtl/disease_confusion_matrix.png'
    )