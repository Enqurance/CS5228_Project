from util import DataPreprocess

train_file_path = './data/train.csv'
outpath = './data/trained.csv'


def main():
	DataPreprocess.PreprocessData(train_file_path,outpath)


if __name__ == '__main__':
	main()
