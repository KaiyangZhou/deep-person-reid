from torchreid.utils import FeatureExtractor

market1501_model = 'log/osnet_x1_0_market1501_softmax_cosinelr/model/model.pth.tar-250'
frida_model = 'log/osnet_x1_0_market1501_softmax_cosinelr/model/frida_model.pth.tar-3'

extractor_1 = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=market1501_model,
    device='cuda'
)

extractor_2 = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=frida_model,
    device='cuda'
)

image_list = [
    '../FRIDA/BBs/Segment_1/0001155/Camera_1/person_01.jpg',
    '../FRIDA/BBs/Segment_1/0001155/Camera_1/person_02.jpg',
    '../FRIDA/BBs/Segment_1/0001155/Camera_1/person_03.jpg'
]

features_market = extractor_1(image_list)
features_frida= extractor_2(image_list)
print("Features MARKET1501")
print(features_market) 
print("Features FRIDA")
print(features_frida) 