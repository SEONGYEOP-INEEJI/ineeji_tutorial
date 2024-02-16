import numpy as np
import torch


# EarlyStopping 클래스는 학습 중에 검증 손실이 개선되지 않을 때 학습을 조기 종료하는 역할을 합니다.
class EarlyStopping:
    # 생성자에서는 patience(검증 손실이 개선되지 않아도 참을 수 있는 에폭 수), verbose(진행 상황 출력 여부), delta(개선으로 간주할 최소 변화량)를 설정합니다.
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    # 객체가 호출될 때마다 검증 손실을 확인하고, 필요한 경우 모델을 저장하거나 조기 종료 플래그를 설정합니다.
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    # 검증 손실이 개선되었을 때 모델을 저장하는 메서드입니다.
    def save_checkpoint(self, val_loss, model):
        """Validation loss decreased (improvement over best loss). Save model."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
