import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
import collections

def undistort(directory,ny,nx):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(directory+'/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            print('Found')
            objpoints.append(objp)
            imgpoints.append(corners)


    # Test undistortion on an image
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ))

    return ret, mtx, dist

def warp(undist):
    img_size = (undist.shape[1],undist.shape[0])
    src = np.float32([[560, 465], [712, 465], [220, 670], [1030, 670]])
    dst = np.float32([[170, 0], [1030, 0], [170, 650], [1030, 650]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel=np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])]=1# is > thresh_min and < thresh_max
    binary_output = np.copy(binary) # Remove this line
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobel=np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])]=1#
    return binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobelx=np.sqrt(sobelx**2)
    abs_sobely=np.sqrt(sobely**2)
    grad_dir=np.arctan2(abs_sobely, abs_sobelx)
    binary = np.zeros_like(grad_dir)
    binary[(grad_dir>=thresh[0]) & (grad_dir<=thresh[1])]=1#
    return binary


def hls_select(img, h_thresh=(0, 255), l_thresh=(0, 255), s_thresh=(0, 255)):
    hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS) #Convert to HLS color space
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    binary_h=np.zeros_like(h_channel)   # Apply a threshold to the S channel
    binary_h[(h_channel>h_thresh[0]) & (h_channel<=h_thresh[1])]=1  # Return a binary image of threshold result
    binary_l=np.zeros_like(l_channel)   # Apply a threshold to the S channel
    binary_l[(l_channel>l_thresh[0]) & (l_channel<=l_thresh[1])]=1  # Return a binary image of threshold result
    binary_s=np.zeros_like(s_channel)   # Apply a threshold to the S channel
    binary_s[(s_channel>s_thresh[0]) & (s_channel<=s_thresh[1])]=1  # Return a binary image of threshold result

    # Combine the three binary thresholds and get the binary that satisfies all thresholds.
    hls_binary = np.zeros_like(binary_s)
    hls_binary[(binary_h == 1) & (binary_l == 1) & (binary_s == 1)] = 1
    return hls_binary

def rgb_select(img, r_thresh=(0, 255), g_thresh=(0, 255), b_thresh=(0, 255)):
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Convert to HLS color space
    r_channel = rgb[:, :, 0]
    g_channel = rgb[:, :, 1]
    b_channel = rgb[:, :, 2]
    binary_r=np.zeros_like(r_channel)   # Apply a threshold to the S channel
    binary_r[(r_channel>r_thresh[0]) & (r_channel<=r_thresh[1])]=1  # Return a binary image of threshold result
    binary_g=np.zeros_like(g_channel)   # Apply a threshold to the S channel
    binary_g[(g_channel>g_thresh[0]) & (g_channel<=g_thresh[1])]=1  # Return a binary image of threshold result
    binary_b=np.zeros_like(b_channel)   # Apply a threshold to the S channel
    binary_b[(b_channel>b_thresh[0]) & (b_channel<=b_thresh[1])]=1  # Return a binary image of threshold result

    # Combine the three binary thresholds and get the binary that satisfies all thresholds.
    rgb_binary = np.zeros_like(binary_r)
    rgb_binary[(binary_r == 1) & (binary_g == 1) & (binary_b == 1)] = 1
    return rgb_binary


def yw_combinator(warped,ry=(0, 255),gy=(0, 255),by=(0, 255),hw=(0, 255),lw=(0, 255),sw=(0, 255),hs=(0, 255),ls=(0, 255),ss=(0, 255)):
    #yello detection
    yello_binary = rgb_select(warped, r_thresh=ry, g_thresh=gy, b_thresh=by)
    #white detection
    white_binary=hls_select(warped,h_thresh = hw, l_thresh = lw, s_thresh = sw)
    #shadow detection
    shadow_binary=hls_select(warped,h_thresh = hs, l_thresh = ls, s_thresh = ss)
    line_binary=np.zeros_like(white_binary)
    line_binary[((white_binary==1) | (yello_binary==1)) & (shadow_binary==1)]=1
    return line_binary


def find_lanes(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
        #              (0, 255, 0), 2)
        #cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
        #              (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

    return out_img, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds, margin, window_img

def get_curvature(ploty, leftx, rightx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25 / 900  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 900  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad
    # Example values: 632.1 m    626.2 m


def get_offset(img, left_fitx, right_fitx ):
    xm_per_pix = 3.7/900 # meters per pixel in x dimension
    lane_center = (right_fitx[-1]+left_fitx[-1])/2
    offset_pixels = lane_center-img.shape[1]/2
    offset_meters = offset_pixels * xm_per_pix

    return offset_meters


def project_back(warped,ploty,left_fitx,right_fitx, Minv,image,undist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def main_pipeline(image):
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    dst = cv2.undistort(image, dist_pickle['mtx'], dist_pickle['dist'], None, dist_pickle['mtx'])
    warped, Minv = warp(dst)
    ksize = 15  # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(warped, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    combined_binary = yw_combinator(warped, ry=(200, 255), gy=(200, 255), by=(0, 255), hw=(0, 255), lw=(0, 255),sw=(140, 255),hs=(0, 255), ls=(140, 255),ss=(0, 255))
    #combined_all = np.zeros_like(img[:, :, 0])
    #combined_all[((gradx == 1) & (combined_binary == 1))] = 1
    out_img, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds, margin, window_img = find_lanes(combined_binary)
    left_curverad, right_curverad = get_curvature(ploty, left_fitx, right_fitx)
    radius = np.minimum(right_curverad, left_curverad)
    offset = get_offset(image, left_fitx, right_fitx)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    result=project_back(combined_binary, ploty, left_fitx, right_fitx, Minv, image, dst)
    resultRGB = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(resultRGB, 'Lane Curvature = '+"{:08.3f}".format(radius)+ ' m', (5, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(resultRGB, 'Offset = '+"{:06.3f}".format(offset)+ ' m', (5, 80), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return resultRGB

# function call to get undistort features
#ret, mtx, dist = undistort('camera_cal',6,9)
# restored pickled undistort features
# dist_pickle=pickle.load(open( "camera_cal/wide_dist_pickle.p", "rb" ))
#
# # Testing functions on all images
# test_images = glob.glob('./CarND-Vehicle-Detection/test_images/test*.jpg')
# fig=plt.figure(figsize=(15, 10))
#
# for i in range(len(test_images)):
#     img = cv2.imread(test_images[i])
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     image_proc = main_pipeline(img)
#     dst = cv2.undistort(img, dist_pickle['mtx'], dist_pickle['dist'], None, dist_pickle['mtx'])
#     warped, Minv = warp(dst)
#     warpedRGB = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
#
#     # Apply each of the filtering operations
#     ksize = 15  # Choose a larger odd number to smooth gradient measurements
#     gradx = abs_sobel_thresh(warped, orient='x', sobel_kernel=ksize, thresh=(10, 100))
#     grady = abs_sobel_thresh(warped, orient='y', sobel_kernel=ksize, thresh=(20, 100))
#     mag_binary = mag_thresh(warped, sobel_kernel=9, mag_thresh=(30, 100))
#     dir_binary = dir_threshold(warped, sobel_kernel=15, thresh=(0.7, 1.2))
#     y_binary = hls_select(warped, h_thresh=(10, 30), l_thresh=(50, 255), s_thresh=(100, 255))
#     w_binary = hls_select(warped, h_thresh=(0, 255), l_thresh=(210, 255), s_thresh=(0, 255))
#     combined_binary = yw_combinator(warped, ry=(200, 255), gy=(200, 255), by=(0, 255), hw=(0, 255), lw=(0, 255),sw=(140, 255),hs=(0, 255), ls=(140, 255),ss=(0, 255))
#
#     combined_all=np.zeros_like(img[:,:,0])
#     combined_all[((gradx == 1) & (combined_binary == 1))] = 1
#
#     out_img, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds, margin, window_img = find_lanes(combined_binary)
#     left_curverad, right_curverad = get_curvature(ploty, left_fitx, right_fitx)
#
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
#     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
#                                                                     ploty])))])
#     left_line_pts = np.hstack((left_line_window1, left_line_window2))
#     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
#     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
#                                                                      ploty])))])
#     right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
#     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
#     fitted_lane = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#     radius = np.minimum(right_curverad, left_curverad)
#     offset = get_offset(img, left_fitx, right_fitx)
#     font = cv2.FONT_HERSHEY_PLAIN
#     #cv2.putText(result, 'Lane Curvature = '+"{:08.3f}".format(radius)+ ' m', (5, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
#     #cv2.putText(result, 'Offset = '+"{:06.3f}".format(offset)+ ' m', (5, 80), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
#
#     #project back the image
#     result=project_back(combined_binary, ploty, left_fitx, right_fitx, Minv, img, dst)
#     resultRGB = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#
#     #visualize outputs on all test images
#     plt.subplot(4, 4, 2 * (i + 1) - 1)
#     plt.imshow(fitted_lane)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
#     plt.axis('off')
#     plt.subplot(4, 4, 2 * (i + 1))
#     plt.imshow(resultRGB)
#     plt.xlim(0, 1280)
#     plt.ylim(720, 0)
#     plt.axis('off')
#     plt.tight_layout()
#
# plt.show()

#fig.savefig('./writeup/back_proj2.png')

# clip = VideoFileClip("project_video.mp4")
# project_clip = clip.fl_image(main_pipeline)
# project_clip.write_videofile('project_video_out2.mp4',audio=False)