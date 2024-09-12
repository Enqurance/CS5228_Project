from util import DataPreprocess

train_file_path = './data/train.csv'


def main():
	DataPreprocess.PreprocessData(train_file_path)


if __name__ == '__main__':
	main()
