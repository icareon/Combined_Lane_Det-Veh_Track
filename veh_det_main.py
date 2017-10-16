import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from vehicle_det_main_funcs import *
from adv_lane_lines import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import random
import time
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip

train=False
video=True

if train==True and video==False:

    # Read in images of cars and notcars
    images = glob.glob('./CarND-Vehicle-Detection/train_images/*/*/*.png')
    cars_full = []
    notcars_full = []

    for image in images:
        if 'non-vehicles' in image:
            notcars_full.append(image)
        else:
            cars_full.append(image)

    # Select 500 images of each category randomly
    # cars=random.sample(cars_full[:-500],500)
    # notcars=random.sample(notcars_full[:-500],500)
    cars=cars_full
    notcars=notcars_full



    ### Parameters to get features and train SVC
    color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    #y_start_stop = [500, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)


    print('Using:', orient, 'orientations,', pix_per_cell,
          'pixels per cell, and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    classifier_data = {}
    classifier_data["svc"] = svc
    classifier_data["scaler"] = X_scaler
    classifier_data["orient"] = orient
    classifier_data["pix_per_cell"] = pix_per_cell
    classifier_data["cell_per_block"] = cell_per_block
    classifier_data["spatial_size"] = spatial_size
    classifier_data["hist_bins"] = hist_bins

    with open('svc_data.pickle', 'wb') as handle:
        pickle.dump(classifier_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if train==False and video==False:

    dist_pickle = pickle.load(open("svc_data.pickle", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]


    #testing it all test images
    test_images = glob.glob('./CarND-Vehicle-Detection/test_images/*.jpg')
    fig=plt.figure(figsize=(15, 10))

    for i in range(len(test_images)):
        boxes = []
        image=cv2.imread(test_images[i])
        box_img = np.copy(image)
        #print(np.min(image),np.max(image))
        #func='cv2.COLOR_BGR2'+color_space
        #image=cv2.cvtColor(image,eval(func))

        ystart = 380
        ystop = 720
        xstart = 750
        xstop =1280
        scales = [1.0,1.5,2.0]

        for scale in scales:
            out_img,box_list = find_cars(image, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            boxes.extend(box_list)

            for j in box_list:
                cv2.rectangle(box_img,j[0],j[1], (0, 0, 255), 6)


        # Add heat to each box in box list
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        #visualize outputs on all test images
        plt.subplot(3, 4, 2 * (i + 1) - 1)
        plt.imshow(draw_img)
        plt.axis('off')
        plt.subplot(3, 4, 2 * (i + 1))
        plt.imshow(heatmap,cmap='hot')
        plt.axis('off')
        #plt.tight_layout()

    #fig.savefig('./writeup_imgs/heat_map.png')
    plt.show()

if video==True:

    dist_pickle = pickle.load(open("svc_data.pickle", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    def main_pipeline_det_track(img):

        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
        dst = cv2.undistort(image, dist_pickle['mtx'], dist_pickle['dist'], None, dist_pickle['mtx'])
        warped, Minv = warp(dst)
        ksize = 15  # Choose a larger odd number to smooth gradient measurements
        gradx = abs_sobel_thresh(warped, orient='x', sobel_kernel=ksize, thresh=(10, 100))
        combined_binary = yw_combinator(warped, ry=(200, 255), gy=(200, 255), by=(0, 255), hw=(0, 255), lw=(0, 255),
                                        sw=(140, 255), hs=(0, 255), ls=(140, 255), ss=(0, 255))
        # combined_all = np.zeros_like(img[:, :, 0])
        # combined_all[((gradx == 1) & (combined_binary == 1))] = 1
        out_img, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds, margin, window_img = find_lanes(
            combined_binary)
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

        resultBGR = project_back(combined_binary, ploty, left_fitx, right_fitx, Minv, image, dst)
        #resultRGB = cv2.cvtColor(resultBGR, cv2.COLOR_BGR2RGB)

        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(resultBGR, 'Lane Curvature = ' + "{:08.3f}".format(radius) + ' m', (5, 50), font, 2,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(resultBGR, 'Offset = ' + "{:06.3f}".format(offset) + ' m', (5, 80), font, 2, (255, 255, 255), 2,
                    cv2.LINE_AA)


        boxes = []
        ystart = 380
        ystop = 720
        xstart = 750
        xstop = 1280
        scales = [1.0, 1.5, 2.0]
        orient = 9
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32  # Number of histogram bins

        for scale in scales:
            out_img, box_list = find_cars(resultBGR, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                                          cell_per_block, spatial_size, hist_bins)
            boxes.extend(box_list)

        # Add heat to each box in box list
        heat = np.zeros_like(resultBGR[:, :, 0]).astype(np.float)
        heat = add_heat(heat, boxes)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(resultBGR), labels)

        return draw_img


    clip = VideoFileClip("./CarND-Vehicle-Detection/project_video.mp4")
    project_clip = clip.fl_image(main_pipeline_det_track)
    project_clip.write_videofile('project_video_out_combined.mp4',audio=False)


# image generation for writeup
# # Read in images of cars and notcars
# images = glob.glob('./CarND-Vehicle-Detection/train_images/*/*/*.png')
# cars_full = []
# notcars_full = []
#
# for image in images:
#     if 'non-vehicles' in image:
#         notcars_full.append(image)
#     else:
#         cars_full.append(image)
#
#
# img1=cv2.imread(cars_full[1])
# img2=cv2.imread(notcars_full[1])
# fig=plt.figure(figsize=(10, 5))
# plt.subplot(1,2,1)
# plt.imshow(img1)
# plt.title('Car')
# plt.subplot(1,2,2)
# plt.imshow(img2)
# plt.title('Not Car')
# fig.savefig('./writeup_imgs/car_vs_notcar.png')
# plt.show()