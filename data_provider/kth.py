class KTHInputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.img_width = input_param['img_width']
        self.img_height = input_param['img_height']
        self.seq_length = input_param['seq_length']  # 추가된 부분
        self.data = []
        self.indices = []
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.load()

    def load(self):
        print(f"Loading dataset from paths: {self.paths}")
        for path in self.paths:
            video_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.avi')]
            for video_file in video_files:
                cap = cv2.VideoCapture(video_file)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (self.img_width, self.img_height))
                    frames.append(frame)
                cap.release()
                if len(frames) < self.seq_length:
                    frames += [np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)] * (self.seq_length - len(frames))
                else:
                    frames = frames[:self.seq_length]
                self.data.append(frames)
        self.data = np.array(self.data, dtype=object)
        print(f"Loaded {len(self.data)} videos.")

    def total(self):
        return len(self.data)

    def begin(self, do_shuffle=True):
        self.indices = np.arange(self.total())
        if do_shuffle:
            np.random.shuffle(self.indices)
        self.current_position = 0
