class OCRIter(mx.io.DataIter):
def __init__(self, count, batch_size, num_label, height, width):
    super(OCRIter, self).__init__()
    self.captcha = ImageCaptcha(fonts=['./data/OpenSans-Regular.ttf'])
    self.batch_size = batch_size
    self.count = count
    self.height = height
    self.width = width
    self.provide_data = [('data', (batch_size, 3, height, width))]
    self.provide_label = [('softmax_label', (self.batch_size, num_label))]

def __iter__(self):
    for k in range(self.count / self.batch_size):
        data = []
        label = []
        for i in range(self.batch_size):
            # 生成一个四位数字的随机字符串
            num = gen_rand() 
            # 生成随机字符串对应的验证码图片
            img = self.captcha.generate(num)
            img = np.fromstring(img.getvalue(), dtype='uint8')
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.width, self.height))
            cv2.imwrite("./tmp" + str(i % 10) + ".png", img)
            img = np.multiply(img, 1/255.0)
            img = img.transpose(2, 0, 1)
            data.append(img)
            label.append(get_label(num))

        data_all = [mx.nd.array(data)]
        label_all = [mx.nd.array(label)]
        data_names = ['data']
        label_names = ['softmax_label']

        data_batch = OCRBatch(data_names, data_all, label_names, label_all)
        yield data_batch

def reset(self):
    pass