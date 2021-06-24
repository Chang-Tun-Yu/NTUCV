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
    psnrs = []
    ssims = []
    if (len(sys.argv) != 3):
        print("usage: python eval.py <mode>,", "<mode>: 0 or 1 or 2")
        sys.exit()
    print(sys.argv)
    if (sys.argv[1] == '0'):
        """ 0_center_frame """
        sequences = ['0', '1', '2', '3', '4', '5', '6']
        from interp_frame import interp_frame
        # sequences = ['0', '1']
        for sq in sequences:
            # read inputs
            I0 = cv2.imread('./data/validation/0_center_frame/'+sq+'/input/frame10.png')
            I1 = cv2.imread('./data/validation/0_center_frame/'+sq+'/input/frame11.png')
            # I0 = cv2.imread('./datasets/0_center_frame/'+sq+'/frame10.png')
            # I1 = cv2.imread('./datasets/0_center_frame/'+sq+'/frame11.png')
            # interpolate
            # size = I0.shape
            if not os.path.exists('./output/0_center_frame/'+sq+'/'):
                os.makedirs('./output/0_center_frame/'+sq+'/')
            # It = test(sys.argv[2], sq, size, 1, "0_center_frame")
            It = interp_frame(I0, I1, sys.argv[1])
            # if not os.path.exists('./output/0_center_frame/'+sq+'/'):
            #     os.makedirs('./output/0_center_frame/'+sq+'/')
            
            # write output
            # It = cv2.imread('./outputs/0_center_frame/'+sq+'/frame10i11.png')
            cv2.imwrite('./output/0_center_frame/'+sq+'/frame10i11.jpg', It)

            # evaluate
            out = cv2.imread('./output/0_center_frame/'+sq+'/frame10i11.jpg')
            out = cv2.imread('./output/0_center_frame/'+sq+'/frame10i11.jpg')
            # print(out.shape)
            gt = cv2.imread('./data/validation/0_center_frame/'+sq+'/GT/frame10i11.png')
            # print(gt.shape)
            psnr_score = psnr(gt, out)
            ssim_score = ssim(gt, out)
            print(sq, psnr_score, ssim_score)
            psnrs.append(psnr_score)
            ssims.append(ssim_score)
        if(os.path.exists('./output/input')):
            os.rmdir('./output/input')
        print("Final Score: PSNR", sum(psnrs)/len(psnrs), "SSIM" , sum(ssims)/len(ssims))
    elif (sys.argv[1] == '1'):
        sequences = ['0', '1', '2']
        from demo import test
        for sq in sequences:
            for i in range(12):            
                I0 = cv2.imread('./data/validation/1_30fps_to_240fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, i*8))
                I1 = cv2.imread('./data/validation/1_30fps_to_240fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, (i+1)*8))

                # Its = interp_frame(I0, I1, sys.argv[1])
                
                if not os.path.exists('./output/1_30fps_to_240fps/'+sq+'/{}/'.format(i)):
                    os.makedirs('./output/1_30fps_to_240fps/'+sq+'/{}/'.format(i))
                size = I0.shape
                It = test(sys.argv[2], str(sq) + str(i), size, 7, "1_30fps_to_240fps")
                for j in range(1, 8):
                    
                    # print('./output/1_30fps_to_240fps/'+sq+'/{}/'.format(j-1))
                    print('./outputs/1_30fps_to_240fps/'+sq+'/{}/000{:0>2d}.png'.format(i, i*8+j))
                    # image = cv2.imread('./outputs/1_30fps_to_240fps/'+sq+'/{}/000{:0>2d}.png'.format(i, i*8+j))
                    # cv2.imwrite('./output/1_30fps_to_240fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*8+j), image)
                
                # evaluate
                for j in range(1, 8):
                    out = cv2.imread('./output/1_30fps_to_240fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*8+j))
                    gt = cv2.imread('./data/validation/1_30fps_to_240fps/'+sq+'/{}/GT/000{:0>2d}.jpg'.format(i, i*8+j))
                    psnr_score = psnr(gt, out)
                    ssim_score = ssim(gt, out)
                    psnrs.append(psnr_score)
                    ssims.append(ssim_score)
                    print(sq, i*8+j, psnr_score, ssim_score)
        if(os.path.exists('./output/input')):
            os.rmdir('./output/input')
        print("Final Score: PSNR", sum(psnrs)/len(psnrs), "SSIM" , sum(ssims)/len(ssims))
    elif (sys.argv[1] == "2"):
        from demo import test
        sequences = ['0', '1', '2']
        for sq in sequences:
            for i in range(8):
                I0 = cv2.imread('./data/validation/2_24fps_to_60fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, i*10))
                I1 = cv2.imread('./data/validation/2_24fps_to_60fps/'+sq+'/{}/input/000{:0>2d}.jpg'.format(i, (i+1)*10))
                if not os.path.exists('./output/2_24fps_to_60fps/'+sq+'/{}/'.format(i)):
                    os.makedirs('./output/2_24fps_to_60fps/'+sq+'/{}/'.format(i))
                if (i%2):
                    size = I0.shape
                    It = test(sys.argv[2], str(sq) + str(i), size, 10, "2_24fps_to_60fps")
                    # It02, It06 = interp_frame(I0, I1, sys.argv[1]+'odd')
                    # It02 = cv2.imread('./outputs/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.png'.format(i, i*10+2))
                    # It06 = cv2.imread('./outputs/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.png'.format(i, i*10+6))
                    # cv2.imwrite('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+2), It02)
                    # cv2.imwrite('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+6), It06)
                    gt02 = cv2.imread('./data/validation/2_24fps_to_60fps/'+sq+'/{}/GT/000{:0>2d}.jpg'.format(i, i*10+2))
                    gt06 = cv2.imread('./data/validation/2_24fps_to_60fps/'+sq+'/{}/GT/000{:0>2d}.jpg'.format(i, i*10+6))
                    o02 = cv2.imread('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+2))
                    o06 = cv2.imread('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+6))
                    psnr_score = psnr(gt02, o02)
                    ssim_score = ssim(gt02, o02)
                    psnrs.append(psnr_score)
                    ssims.append(ssim_score)
                    print(sq, i*10+2, psnr_score, ssim_score)
                    psnr_score = psnr(gt06, o06)
                    ssim_score = ssim(gt06, o06)
                    psnrs.append(psnr_score)
                    ssims.append(ssim_score)
                    print(sq, i*10+6, psnr_score, ssim_score)
                else:
                    size = I0.shape
                    It = test(sys.argv[2], str(sq) + str(i), size, 10, "2_24fps_to_60fps")
                    # It04, It08 = interp_frame(I0, I1, sys.argv[1]+'
                    # It04 = cv2.imread('./outputs/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.png'.format(i, i*10+4))
                    # It08 = cv2.imread('./outputs/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.png'.format(i, i*10+8))
                    # cv2.imwrite('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+4), It04)
                    # cv2.imwrite('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+8), It08)
                    gt04 = cv2.imread('./data/validation/2_24fps_to_60fps/'+sq+'/{}/GT/000{:0>2d}.jpg'.format(i, i*10+4))
                    gt08 = cv2.imread('./data/validation/2_24fps_to_60fps/'+sq+'/{}/GT/000{:0>2d}.jpg'.format(i, i*10+8))
                    o04 = cv2.imread('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+4))
                    o08 = cv2.imread('./output/2_24fps_to_60fps/'+sq+'/{}/000{:0>2d}.jpg'.format(i, i*10+8))
                    psnr_score = psnr(gt04, o04)
                    ssim_score = ssim(gt04, o04)
                    psnrs.append(psnr_score)
                    ssims.append(ssim_score)
                    print(sq, i*10+4, psnr_score, ssim_score)
                    psnr_score = psnr(gt08, o08)
                    ssim_score = ssim(gt08, o08)
                    psnrs.append(psnr_score)
                    ssims.append(ssim_score)
                    print(sq, i*10+8, psnr_score, ssim_score)
        if(os.path.exists('./output/input')):
            os.rmdir('./output/input')
        print("Final Score: PSNR", sum(psnrs)/len(psnrs), "SSIM" , sum(ssims)/len(ssims))
                    
    else:
        print("mode is wrong")