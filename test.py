import cv2
import numpy as np
import math
from skimage.measure import compare_ssim
import os
import glob



def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

def ssim(img1, img2):
    return compare_ssim(img1.astype(np.float32)/255., img2.astype(np.float32)/255., gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True)

if __name__ == '__main__':
    import sys
    if (len(sys.argv) != 3):
        print("usage: python test.py <mode>,", "<mode>: 0 or 1 or 2")
        sys.exit()
    if (sys.argv[1] == '0'):
        """ 0_center_frame """
        from interp_frame import interp_frame
        sequences = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        for sq in sequences:
            # read inputs
            I0 = cv2.imread('./data/testing/0_center_frame/'+sq+'/input/frame10.png')
            I1 = cv2.imread('./data/testing/0_center_frame/'+sq+'/input/frame11.png')
            if not os.path.exists('./test_output/0_center_frame/'+sq+'/'):
                os.makedirs('./test_output/0_center_frame/'+sq+'/')
            # interpolate
            It = interp_frame(I0, I1, sys.argv[1])
            # size = I0.shape
            # It = test(sys.argv[2], sq, size, 1, "0_center_frame")
            # write test_output
            cv2.imwrite('./test_output/0_center_frame/'+sq+'/frame10i11.jpg', It)
        if(os.path.exists('./test_output/input')):
            os.rmdir('./test_output/input')

    elif (sys.argv[1] == '1'):
        from demo import test
        sequences = ['3', '4']
        for sq in sequences:
            for i in range(12):            
                I0 = cv2.imread('./data/testing/1_30fps_to_240fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, i*8))
                I1 = cv2.imread('./data/testing/1_30fps_to_240fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, (i+1)*8))
                # Its = interp_frame(I0, I1, sys.argv[1])
                if not os.path.exists('./test_output/1_30fps_to_240fps/'+sq+'/{}/'.format(i)):
                    os.makedirs('./test_output/1_30fps_to_240fps/'+sq+'/{}/'.format(i))
                size = I0.shape
                It = test(sys.argv[2], str(sq) + str(i), size, 7, "1_30fps_to_240fps")
        if(os.path.exists('./test_output/input')):
            os.rmdir('./test_output/input')
                # for j in range(1, 8):
                #     cv2.imwrite('./test_output/1_30fps_to_240fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*8+j), Its[j-1])
                
    elif (sys.argv[1] == "2"):
        from demo import test
        sequences = ['3', '4']
        for sq in sequences:
            for i in range(8):
                I0 = cv2.imread('./data/testing/2_24fps_to_60fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, i*10))
                I1 = cv2.imread('./data/testing/2_24fps_to_60fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, (i+1)*10))
                
                if not os.path.exists('./test_output/2_24fps_to_60fps/'+sq+'/{}/'.format(i)):
                    os.makedirs('./test_output/2_24fps_to_60fps/'+sq+'/{}/'.format(i))
                
                if (i%2):
                    # It02, It06 = interp_frame(I0, I1, sys.argv[1]+'odd')
                    size = I0.shape
                    It = test(sys.argv[2], str(sq) + str(i), size, 10, "2_24fps_to_60fps")
                    # cv2.imwrite('./test_output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+2), It02)
                    # cv2.imwrite('./test_output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+6), It06)
                else:
                    size = I0.shape
                    It = test(sys.argv[2], str(sq) + str(i), size, 10, "2_24fps_to_60fps")
                    # It04, It08 = interp_frame(I0, I1, sys.argv[1]+'even')
                    # cv2.imwrite('./test_output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+4), It04)
                    # cv2.imwrite('./test_output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+8), It08)
        if(os.path.exists('./test_output/input')):
            os.rmdir('./test_output/input')
                    
    else:
        print("mode is wrong")




