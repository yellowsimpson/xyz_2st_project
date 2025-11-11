# -*- coding: utf-8 -*-
"""
간단 멀티모달 정책 예제

입력:
  - 언어: "경유 3만원" 같은 텍스트 명령
  - 영상: 차량 번호판 텍스트 (예: "12가3456")
  - 로봇: 현재 로봇 상태 벡터 (joint, tcp pose 등)

출력:
  - action 벡터 (예: 고수준 명령, 또는 제어 명령)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. 언어 인코더 (Text → Embedding)
# =========================

class FuelCommandEncoder(nn.Module):
    """
    "경유 3만원", "휘발유 50000원" 같은 한 줄 명령을
    작은 벡터로 바꿔주는 간단 인코더.
    
    실제 프로젝트에서는:
      - Whisper + LLM으로 구조화된 정보까지 뽑고
      - 여기서는 그 구조화된 정보를 벡터로만 바꿔주는 역할을 하게 만들면 좋음.
    """
    def __init__(self):
        super().__init__()
        # 연료 타입을 one-hot으로 표현하기 위한 매핑
        self.fuel_types = {
            "경유": 0,
            "디젤": 0,
            "휘발유": 1,
            "가솔린": 1,
            "LPG": 2,
            "전기": 3,
        }
        self.num_fuel_types = 4  # 위에서 경유/휘발유/LPG/전기 4종으로 가정

        # amount, is_price 등 연속값을 약간 확장해줄 작은 MLP
        # fuel one-hot(4) + amount_norm(1) + is_price(1) = 6차원 → 16차원
        self.mlp = nn.Sequential(
            nn.Linear(self.num_fuel_types + 2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.output_dim = 16

    def parse_command(self, text: str):
        """
        "경유 3만원" 같은 텍스트에서
        - fuel_type one-hot
        - amount (숫자)
        - amount가 '원' 단위인지, '리터' 단위인지 플래그
        를 뽑는 간단 파서.
        """
        text = text.strip()

        # 1) 연료 타입 찾기
        fuel_idx = None
        for k, v in self.fuel_types.items():
            if k in text:
                fuel_idx = v
                break
        if fuel_idx is None:
            fuel_idx = 0  # 모르면 경유로 대충

        fuel_one_hot = [0.0] * self.num_fuel_types
        fuel_one_hot[fuel_idx] = 1.0

        # 2) 금액/수량 숫자 뽑기 (가장 첫 숫자)
        m = re.search(r"(\d+)", text)
        if m:
            amount = float(m.group(1))
        else:
            amount = 0.0

        # 3) '만원', '원', '리터' 여부
        if "만원" in text:
            amount_value = amount * 10000.0
            is_price = 1.0
        elif "원" in text:
            amount_value = amount
            is_price = 1.0
        else:
            # 단순히 리터라고 가정
            amount_value = amount
            is_price = 0.0

        # 4) 금액/수량 정규화 (대략 0~1 사이로)
        amount_norm = amount_value / 100000.0  # 10만원 기준으로 스케일링

        return fuel_one_hot, amount_norm, is_price

    def forward(self, texts):
        """
        texts: 리스트[str] 또는 단일 str
        출력: (batch, 16)짜리 텐서
        """
        if isinstance(texts, str):
            texts = [texts]

        feats = []
        for t in texts:
            fuel_one_hot, amount_norm, is_price = self.parse_command(t)
            vec = fuel_one_hot + [amount_norm, is_price]
            feats.append(vec)

        x = torch.tensor(feats, dtype=torch.float32)
        out = self.mlp(x)
        return out  # (batch, 16)


# =========================
# 2. 차량 번호 인코더 (Plate → Embedding)
# =========================

class PlateEncoder(nn.Module):
    """
    번호판 문자열 (예: "12가3456")을
    문자 단위 임베딩 후 평균내서 하나의 벡터로 만드는 인코더.
    
    실제로는 OCR에서 나온 plate string을 넣어주면 됨.
    """
    def __init__(self, embed_dim=16):
        super().__init__()

        # 번호판에서 나올 법한 문자들 (숫자 + 한글 + 몇몇 알파벳)
        chars = "0123456789가나다라마바사아자차카타파하ABCDEFGHJKLMNPRSTUVWXYZ"
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0은 PAD/UNK
        self.vocab_size = len(self.char2idx) + 1

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.output_dim = embed_dim

    def encode_plate(self, plate: str):
        idxs = []
        for ch in plate:
            idxs.append(self.char2idx.get(ch, 0))  # 모르는 문자는 0
        if not idxs:
            idxs = [0]
        return idxs

    def forward(self, plates):
        """
        plates: 리스트[str] 또는 단일 str
        출력: (batch, embed_dim)
        """
        if isinstance(plates, str):
            plates = [plates]

        encoded = [self.encode_plate(p) for p in plates]
        max_len = max(len(seq) for seq in encoded)

        padded = []
        for seq in encoded:
            padded_seq = seq + [0] * (max_len - len(seq))
            padded.append(padded_seq)

        x = torch.tensor(padded, dtype=torch.long)  # (batch, max_len)
        emb = self.embedding(x)                    # (batch, max_len, embed_dim)
        # 길이 방향 평균
        emb = emb.mean(dim=1)                      # (batch, embed_dim)
        return emb


# =========================
# 3. 로봇 상태 인코더 (State → Embedding)
# =========================

class RobotStateEncoder(nn.Module):
    """
    로봇 상태 벡터 (joint, tcp 등)를
    조금 더 compact한 표현으로 바꿔주는 인코더.
    """
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, state):
        """
        state: (batch, state_dim) 또는 (state_dim,) 텐서
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.mlp(state)


# =========================
# 4. 멀티모달 정책 (Text + Plate + State → Action)
# =========================

class MultiModalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.text_encoder = FuelCommandEncoder()
        self.plate_encoder = PlateEncoder(embed_dim=16)
        self.state_encoder = RobotStateEncoder(state_dim=state_dim, hidden_dim=32)

        fused_input_dim = (
            self.text_encoder.output_dim
            + self.plate_encoder.output_dim
            + self.state_encoder.output_dim
        )

        self.policy = nn.Sequential(
            nn.Linear(fused_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, text_cmd, plate_text, robot_state):
        """
        text_cmd: str 또는 리스트[str]
        plate_text: str 또는 리스트[str]
        robot_state: (state_dim,) 또는 (batch, state_dim) float 텐서
        """
        e_text = self.text_encoder(text_cmd)       # (batch, 16)
        e_plate = self.plate_encoder(plate_text)   # (batch, 16)

        if isinstance(robot_state, list):
            robot_state = torch.tensor(robot_state, dtype=torch.float32)
        e_state = self.state_encoder(robot_state)  # (batch, 32)

        # 배치 크기를 맞추기 위해 간단 처리
        # (여기선 text / plate / state 모두 같은 batch라고 가정)
        z = torch.cat([e_text, e_plate, e_state], dim=-1)
        action = self.policy(z)  # (batch, action_dim)
        return action


# =========================
# 5. 사용 예시
# =========================

if __name__ == "__main__":
    # 예시: 6 DOF 로봇 + TCP pose 6 = state_dim = 12 로 가정
    state_dim = 12
    action_dim = 4  # 예: [start_flag, stop_flag, flow_rate, error_prob] 같은 의미로 가정

    model = MultiModalPolicy(state_dim=state_dim, action_dim=action_dim)

    # 1) 입력 예시
    text_cmd = "경유 3만원"
    plate_text = "12가3456"
    # 예시 로봇 상태 (joint 6개 + tcp pose 6개)
    robot_state = [0.0, -0.5, 1.2, 0.3, -1.0, 0.2,  0.5, 0.3, 0.8, 0.0, 0.1, -0.2]

    # 2) forward
    action = model(text_cmd, plate_text, robot_state)
    print("action 벡터:", action.detach().numpy())

    # 3) 학습할 때는 이런 식으로 loss를 정의해서 최적화하면 됨 (예: MSE)
    # 예시 타깃 (사람이 조작한 행동, 또는 RL policy의 target 등)
    target_action = torch.zeros_like(action)
    loss = F.mse_loss(action, target_action)
    loss.backward()
    print("loss:", loss.item())
