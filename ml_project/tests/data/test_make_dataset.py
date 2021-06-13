from pandas.testing import assert_frame_equal
from src.data.make_dataset import read_dataset, train_val_split
from src.params.split_params import SplitParams


def test_read_data(tmpdir, sample_data):
    data_path = tmpdir.join('sample.csv')
    sample_data.to_csv(data_path, index_label=False)
    data = read_dataset(data_path)
    assert_frame_equal(sample_data, data)


def test_split_data(sample_data):
    val_size = 0.2
    split_params = SplitParams(
        val_size=val_size,
        random_state=101,
        shuffle=True
    )
    train, val = train_val_split(
        sample_data,
        split_params
    )
    assert len(train) >= len(val)
