import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path):
    # 컬럼 이름 지정 (UCI 웹사이트에서 제공하는 설명서에 따라)
    column_names = [
        "ID", "Diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    # CSV 파일 로드
    data = pd.read_csv(file_path, names=column_names)

    # ID 열 제거
    data.drop("ID", axis=1, inplace=True)

    # 진단 결과 레이블 인코딩
    label_encoder = LabelEncoder()
    data["Diagnosis"] = label_encoder.fit_transform(data["Diagnosis"])

    # 특성과 레이블 분리
    X = data.drop("Diagnosis", axis=1).values
    y = data["Diagnosis"].values

    # 특성 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 신경망 클래스 정의
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        # 바이어스 초기화
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def feedforward(self, X):
        # 은닉층
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)

        # 출력층
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_activation)

        return self.output

    def backpropagation(self, X, y, learning_rate):
        # 출력층 오류
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        # 은닉층 오류
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # 가중치 및 바이어스 업데이트
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, X_val, y_val, epochs, learning_rate):
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)
            if epoch % 1000 == 0:
                loss = self.mse_loss(y, self.output)
                val_output = self.feedforward(X_val)
                val_loss = self.mse_loss(y_val, val_output)
                print(f"Epoch {epoch}, Loss: {loss}, Validation Loss: {val_loss}")

# 데이터 로드 및 전처리
file_path = 'data/wdbc.data'
X, y = load_and_preprocess_data(file_path)

# 데이터 분할 (훈련 세트, 검증 세트, 테스트 세트)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 레이블 재구성
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 신경망 초기화
nn = NeuralNetwork(input_size=30, hidden_size=10, output_size=1)

# 훈련
nn.train(X_train, y_train, X_val, y_val, epochs=30000, learning_rate=0.01)

# 테스트
output = nn.feedforward(X_test)
predictions = (output > 0.5).astype(int)

# 성능 평가
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")
