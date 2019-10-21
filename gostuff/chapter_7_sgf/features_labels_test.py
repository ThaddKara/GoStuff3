from dlgo.data.processor import GoDataProcessor

if __name__ == '__main__':
    processor = GoDataProcessor()
    features, labels = processor.load_go_data('train', 100)
