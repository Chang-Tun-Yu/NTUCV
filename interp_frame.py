import numpy as np
import cv2 as cv

def write_images(img0, img1, flow):
    cv.imwrite("img0.png", img0)
    cv.imwrite("img1.png", img1)
    cv.imwrite("flow.png", flow)

def flow_to_color(flow, hsv):
    # brighter pixels will correspond to higher flows, and the color will correspond to the direction
    mag, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) # Computes the magnitude and angle of the 2D vectors
    hsv[..., 0] = angle*180/np.pi/2 # Sets image hue according to the optical flow direction
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) # Sets image value according to the optical flow magnitude (normalized)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR) # Converts HSV to RGB (BGR) color representation

def forward_warping(src, dst, t, flow):
    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.put_
    # prev_frame[y, x] == curr_frame[y + flow[y, x, 1], x + flow[y, x, 0]]
    h, w, ch = src.shape
    mid_frame = np.zeros_like(src)
    x_src = np.arange(w)
    y_src = np.arange(h)
    xp_src, yp_src = np.meshgrid(x_src,y_src)
    src_cood = np.vstack(list(map(np.ravel, [yp_src, xp_src]))) # image is (row, column) indexing

    flow_vectors = flow.ravel().reshape(-1, 2).T
    flow_vectors[[0,1]] = flow_vectors[[1,0]] # flow is (x,y) coordinate-based, which is (column, row) indexing

    dst_cood = (np.round(src_cood + t * flow_vectors)).astype(int) # (2, 226592)
    dst_cood[0] = np.clip(dst_cood[0], 0, h-1)
    dst_cood[1] = np.clip(dst_cood[1], 0, w-1)
    mid_frame[dst_cood[0], dst_cood[1]] = src[src_cood[0], src_cood[1]]
    occlution_mask = (mid_frame > 0).astype(np.float32)
    return mid_frame, occlution_mask

def backward_warping(src, dst, flow):
    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.put_
    # prev_frame[y, x] == curr_frame[y + flow[y, x, 1], x + flow[y, x, 0]]
    h, w, ch = src.shape
    mid_frame = np.zeros_like(src)
    x_dst = np.arange(w)
    y_dst = np.arange(h)
    xp_dst, yp_dst = np.meshgrid(x_dst,y_dst)
    dst_cood = np.vstack(list(map(np.ravel, [yp_dst, xp_dst]))) # image is (row, column) indexing

    flow_vectors = flow.ravel().reshape(-1, 2).T
    flow_vectors[[0,1]] = flow_vectors[[1,0]] # flow is (x,y) coordinate-based, which is (column, row) indexing

    src_cood = (np.round(dst_cood + flow_vectors)).astype(int) # (2, 226592)
    src_cood[0] = np.clip(src_cood[0], 0, h-1)
    src_cood[1] = np.clip(src_cood[1], 0, w-1)
    mid_frame[dst_cood[0], dst_cood[1]] = src[src_cood[0], src_cood[1]]
    return mid_frame


def optical_flow(img0, img1):
    gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 
    hsv = np.zeros_like(img0) #zero intensities with same type as img0 (388, 584, 3) 
    hsv[:,:,1] = 255 # Sets image saturation to maximum 
    
    flow = cv.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 30, 3, 5, 1.1, 0) # (384, 584, 2) dense optical flow for all pixels
    # rgb_flow = flow_to_color(flow, hsv) # show flow image, nothing to do with interpolated frame
    # write_images(img0, img1, rgb_flow)
    # gray0 = gray1 # update previous frame
    return flow

def splatting(frame0, frame1, O_mask0, O_mask1, t):
    Z = O_mask0*(1-t) + O_mask1*t
    frame0_factor = np.true_divide(O_mask0*(1-t), Z)
    return (frame0*frame0_factor + frame1*(1-frame0_factor))
    # return frame0

# def occlution(flow, t):
#     h, w, ch = flow.shape
#     one_map = np.ones((h, w, 1))
#     occlusion = forward_warping(one_map, one_map, t, flow) > 0
#     occlusion = np.logical_not(occlusion).astype(np.float32)
#     return occlusion


def interp_frame(img0, img1, mode):
    if (mode == '0'):
        flow0_1 = optical_flow(img0, img1)
        mid_frame0_1, occlution_mask0 = forward_warping(img0, img1, 0.5, flow0_1)
        flow1_0 = optical_flow(img1, img0)
        mid_frame1_0, occlution_mask1 = forward_warping(img1, img0, 0.5, flow1_0)
        mid_frame = splatting(mid_frame0_1, mid_frame1_0, occlution_mask0, occlution_mask1, 0.5)

        # backward
        # t = 0.5
        # flow0_1 = optical_flow(img0, img1)
        # flow1_0 = optical_flow(img1, img0)
        # flowt_0 = (t-1)*t*flow0_1 + (t**2)*flow1_0
        # flowt_1 = ((t-1)**2)*flow0_1 + t*(t-1)*flow1_0
        # mid_frame0_1 = backward_warping(img0, img1, flowt_0)
        # mid_frame1_0 = backward_warping(img1, img0, flowt_1)
        # mid_frame = splatting(mid_frame0_1,mid_frame1_0, -1,-1,0.5)

        return mid_frame

    elif (mode == '1'):
        frames = []
        flow0_1 = optical_flow(img0, img1)        
        flow1_0 = optical_flow(img1, img0)
        for i in range(1, 8):
            # if (i < 5):
            #     frames.append(img0)
            # else:
            #     frames.append(img1)
            
            # frame0_1, occlution_mask0 = forward_warping(img0, img1, i*0.125, flow0_1)
            # frame1_0, occlution_mask1 = forward_warping(img1, img0, 1 - i*0.125, flow1_0)
            # frame = splatting(frame0_1, frame1_0, occlution_mask0, occlution_mask1, i*0.125)
            # frames.append(frame)

            t = i*0.125
            if (i < 5):
                flowt_0 = (t-1)*t*flow0_1 + (t**2)*flow1_0
                frame = backward_warping(img0, img1, flowt_0)
            else:
                flowt_1 = ((t-1)**2)*flow0_1 + t*(t-1)*flow1_0
                frame = backward_warping(img1, img0, flowt_1)
            frames.append(frame)
        return frames
    elif (mode == '2odd'):
        flow0_1 = optical_flow(img0, img1)        
        flow1_0 = optical_flow(img1, img0)
        # frame0_1_a, occlution_mask0_a = forward_warping(img0, img1, 0.2, flow0_1)
        # frame1_0_a, occlution_mask1_a = forward_warping(img1, img0, 0.8, flow1_0)
        # frame_02 = splatting(frame0_1_a, frame1_0_a, occlution_mask0_a, occlution_mask1_a, 0.2)

        # frame0_1_b, occlution_mask0_b = forward_warping(img0, img1, 0.6, flow0_1)
        # frame1_0_b, occlution_mask1_b = forward_warping(img1, img0, 0.4, flow1_0)
        # frame_06 = splatting(frame0_1_b, frame1_0_b, occlution_mask0_b, occlution_mask1_b, 0.6)

        t = 0.2
        flowt_0 = (t-1)*t*flow0_1 + (t**2)*flow1_0
        frame_02 = backward_warping(img0, img1, flowt_0)
        t = 0.6
        flowt_1 = ((t-1)**2)*flow0_1 + t*(t-1)*flow1_0
        frame_06 = backward_warping(img1, img0, flowt_1)   

        return frame_02, frame_06

    elif (mode == '2even'):
        flow0_1 = optical_flow(img0, img1)        
        flow1_0 = optical_flow(img1, img0)
        # frame0_1_a, occlution_mask0_a = forward_warping(img0, img1, 0.4, flow0_1)
        # frame1_0_a, occlution_mask1_a = forward_warping(img1, img0, 0.6, flow1_0)
        # frame_04 = splatting(frame0_1_a, frame1_0_a, occlution_mask0_a, occlution_mask1_a, 0.4)

        # frame0_1_b, occlution_mask0_b = forward_warping(img0, img1, 0.8, flow0_1)
        # frame1_0_b, occlution_mask1_b = forward_warping(img1, img0, 0.2, flow1_0)
        # frame_08 = splatting(frame0_1_b, frame1_0_b, occlution_mask0_b, occlution_mask1_b, 0.8)

        t = 0.4
        flowt_0 = (t-1)*t*flow0_1 + (t**2)*flow1_0
        frame_04 = backward_warping(img0, img1, flowt_0)
        t = 0.8
        flowt_1 = ((t-1)**2)*flow0_1 + t*(t-1)*flow1_0
        frame_08 = backward_warping(img1, img0, flowt_1) 
        return frame_04, frame_08