from dlgo.data.parallel_processor import GoDataProcessor

if __name__ == '__main__':
    processor = GoDataProcessor()
    generator = processor.load_go_data('train', 100, use_generator=True)

    print(generator.get_num_samples())

    generator = generator.generate(batch_size=10)
    X, y = generator.next()