# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000       # 훈련 데이터 개수
test_num = 10000        # 검증 데이터 개수
img_dim = (1, 28, 28)   # 이미지 형상 정보 (채널 1, height 28, width 28)
img_size = 784          # 이미지 크기 (1 X 28 X 28), 이미지를 1차원 데이터로 만들때의 크기


def _download(file_name):
    file_path = dataset_dir + "/" + file_name

    # file_path에 파일이 존재하면 이미 다운로드 된 상태(별다른 처리 필요없다!)
    if os.path.exists(file_path):
        return

    # 다운로드 시작
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def download_mnist():
    # key_file 모두 다운로드 받기
    for v in key_file.values():
       _download(v)


# 라벨 파일 읽기
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    # gzip 열어서 라벨 데이터를 Numpy Array로 읽어오기
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


# 이미지 파일 읽기
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    # gzip 열어서 이미지 데이터를 Numpy Array로 읽어오기
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 형상 변경 : 각 이미지 데이터를 flatten(평탄화)해서 1차원으로 변경 -> 데이터 수 만큼(N) 1차원 배열이 생성 (N, img_size)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


# 다운받은 파일들에서 라벨과 이미지 데이터를 Numpy Array 형태로 가져오기
def _convert_numpy():
    dataset = {}
    dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


# mnist 데이터 초기화 : 파일 다운로드 + 데이터를 Numpy Array 형태로 변형 + pickle 파일형태로 저장
def init_mnist():
    # key_file 모두 다운로드 받기
    download_mnist()
    # 다운받은 파일들에서 라벨과 이미지 데이터를 Numpy Array 형태로 가져오기
    dataset = _convert_numpy()
    # pickle 파일에 저장하기 - 초기화시에 매번 다운로드 받는 것이 아니라 pickle 파일에서 읽어온다.
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기

    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label :
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다.

    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    # mnist 데이터 초기화 : 파일 다운로드 + 데이터를 Numpy Array 형태로 변형 + pickle 파일형태로 저장
    if not os.path.exists(save_file):
        init_mnist()

    # pickle 파일 읽기
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 조건 1.
    # 이미지 데이터 정규화 (0 ~ 255 범위인 각 픽셀 값을 0.0 ~ 1 범위로 변환)
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 조건 2.
    # one-hot encoding 처리
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 조건 3.
    # 평탄화시킨 데이터를 원래 형상으로 재변형
    if not flatten:
         for key in ('train_img', 'test_img'):
            #dataset[key] = dataset[key].reshape(-1, 1, 28, 28)  # 데이터 수 만큼(N) 생성 (N, 1, 28, 28)
            # 데이터 수 만큼(N) 생성, (N, 1, 28, 28)
            dataset[key] = dataset[key].reshape(-1, img_dim[0], img_dim[1], img_dim[2])

    # 훈련용 이미지, 라벨 데이터와 검증용 이미지, 라벨 데이터를 튜플 형태로 반환
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
