from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / '.cache'


def load_mnist() -> Bunch:
    name = 'mnist_784'
    df = fetch_openml(
        name=name,
        version=1,
        as_frame=True,
        data_home=str(CACHE_DIR),
    )

    return df


if __name__ == '__main__':
    df = load_mnist()
    print(df)
