import time

import cv2

import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t

#设备初始化
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)



def postrans(s):
    x=''
    y=''
    z=''

    x_s = s.find('x',0, len(s))
    x_p = s.find('.',0, len(s))
    x = s[x_s+3 : x_p +3]

    y_s = s.find('y',x_p +1, len(s))
    y_p = s.find('.',x_p +1, len(s))
    y = s[y_s+3 : y_p +3]

    z_s = s.find('z',y_p +1, len(s))
    z_p = s.find('.',y_p +1, len(s))
    z = s[z_s+3 : z_p +3]
    return (x,y,z)

def getdis(s):
    z_s = s.find('z',10, len(s))
    z_p = s.find('.',z_s+4, len(s))
    #print(z_p)
    #print(s)
    z = s[z_s+3 : z_p]
    #print(z)
    return int(z)

def dismap():
    pass


def use_kinect():   #返回彩图和深度图
    capture = device.update()

    # Get the color image from the capture
    ret_color, color_image = capture.get_color_image()

    # Get the colored depth
    ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

    if not ret_color or not ret_depth:
        raise Exception('kinect未能捕获图像，请检查连接')
        return

    return color_image,transformed_depth_image



if __name__ == "__main__":

    #cv2.namedWindow('Transformed Color Image', cv2.WINDOW_NORMAL)
    while True:

        # Get capture
        capture = device.update()
       # depimg = capture.get_depth_image()
        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue

        pix_x = color_image.shape[1] // 2   #宽
        pix_y = color_image.shape[0] // 2   #高
        #print(color_image.shape)
        #print(color_image[pix_x, pix_y])
        rgb_depth = transformed_depth_image[pix_y, pix_x]     #选取中间像素的高度

        pixels = k4a_float2_t((pix_x, pix_y)) #将点转换成像素对象

        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR,
                                                          K4A_CALIBRATION_TYPE_COLOR)
        pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR,
                                                          K4A_CALIBRATION_TYPE_DEPTH)
        #print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

        # Overlay body segmentation on depth image
        frame_texted = color_image.copy()
        pos_string = str(pos3d_depth)
        map3D = postrans(pos_string)


        cv2.putText(frame_texted, "o distance:"+ postrans(pos_string) +"mm", (pix_x, pix_y),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,50,50))
        cv2.imshow('Transformed Color Image', frame_texted  )

        time.sleep(0.05)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break