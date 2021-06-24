# testset_root = './data/validation/'
testset_root = './data/testing/'

# test_size = (854, 480)
# test_size = (640, 480)
# test_crop_size = (640, 480)
# test_crop_size = (854, 480)

mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]

inter_frames = 1

frame1 = 1
frame2 = 7
frame3 = 10

model = 'QVI'
pwc_path = './utils/pwc-checkpoint.pt'



# store_path = 'output/'
store_path = 'test_output/'
checkpoint = 'qvi_release/model.pt'


