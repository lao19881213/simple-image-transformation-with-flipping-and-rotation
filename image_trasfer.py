import tensorflow as tf 
import os
import math
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image


input_imagefile = './source_images/'
output_imagefile = './destination_images/'
output_dir_name = 'destination_images'
xml_list = []
img_list = []
full_dir_list = {}
size_img = np.array((520,780))
flipped_size_img = np.array((520,780))
input_flipped_size_img = np.array((520,780))

# upside down the original images, there will be 1 (origin) + 1 (filpped) = 2
def upside_down(image):
    flipped = tf.image.flip_up_down(image)
    return flipped

# every image will rotate 4 times
def rot90(image, times_of_rot):
    image_rot90ed = tf.image.rot90(image, k=times_of_rot)
    return image_rot90ed


# translate to the center of the original image



### insert the min and max points of rect through human label
def first_translation(min_point, max_point, img_size_but_not_rect):
    
    center_point_but_not_rect_center = img_size_but_not_rect/2
    print("center of input image ==", center_point_but_not_rect_center)
    print("min_point ==",min_point)
    print("max_point ==",max_point)
    min_translated_point = np.subtract(min_point, center_point_but_not_rect_center)
    max_translated_point = np.subtract(max_point, center_point_but_not_rect_center)

    print("min_translated_point ==",min_translated_point)
    print("max_translated_point ==",max_translated_point)

    return min_translated_point, max_translated_point


# 2. rotate the coordinate of x and y
def coord_transform(min_point, max_point, angle):
    
    if angle < 0:
        angle = angle + 2*math.pi
        if angle == math.pi/2:
            cos = 0
            sin = 1
        if angle == 3*math.pi/2 :
            cos = 0
            sin = -1
        if angle == math.pi:
            sin = 0
            cos = -1
        if angle == 0:
            sin = 0
            cos = 1


    tri = np.array([
        (cos,-sin),
        (sin,cos)
        ])
    
    print("tri ==",tri)
    m1_result = np.matmul(tri,min_point)
    m2_result = np.matmul(tri,max_point)

    return m1_result, m2_result

# 3. translate back to the origin of rotated img 
def translate_back_to_rotated_img(rotated_coords, img_size_before_rotation):

    offset = np.array((img_size_before_rotation[1]/2, img_size_before_rotation[0]/2))
    p1 = np.add(rotated_coords[0], offset)
    p2 = np.add(rotated_coords[1], offset)

    return p1, p2

# pick up the min and max points of a boundingbox
def output_min_and_max_points_of_rect(transformed_point_one, transformed_point_two):
    
    x_comparison = np.array((transformed_point_one[0],transformed_point_two[0]))
    y_comparison = np.array((transformed_point_one[1],transformed_point_two[1]))

    out_min = np.array((x_comparison[np.argmin(x_comparison)], y_comparison[np.argmin(y_comparison)]))
    out_max = np.array((x_comparison[np.argmax(x_comparison)], y_comparison[np.argmax(y_comparison)]))
    
    print ("output transformed min and max points ==",out_min,out_max)
    return out_min, out_max




with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    input_count = 0
    output_count = 0
    
    for root, sub_folders, files in os.walk(input_imagefile,topdown=True):
        path_list = os.listdir(root)
        full_dir_list.update({root:files})
    print(full_dir_list)
    # sorting
    for name in full_dir_list:
        for i in range(0,len(full_dir_list[name])):
            if not str(full_dir_list[name][i]).startswith('.'):
                if os.path.splitext(full_dir_list[name][i])[-1] == '.xml':
                    xml_list.append(full_dir_list[name][i])
                    xml_list.sort(reverse=False)
                if os.path.splitext(full_dir_list[name][i])[-1] == '.jpg':
                    img_list.append(full_dir_list[name][i])
                    img_list.sort(reverse=False)
    

    print("xml_list ==",xml_list)
    print("img_list ==",img_list)
        

    for i in range(0,len(img_list)):
        
        # read the input image path
        image_file = os.path.join(root, img_list[i])
        xml_file = os.path.join(root, xml_list[i])
        print("xml_file ==",xml_file)
        
        
        print("input count = ",input_count)
        
        
        # read the input image files through tensorflow method
        if os.path.splitext(xml_list[i])[0] == os.path.splitext(img_list[i])[0]:
            print("image_file =",image_file)
            image_raw_data = tf.gfile.FastGFile(image_file,"rb").read()
            image_decode = tf.image.decode_jpeg(image_raw_data)
            input_count = input_count + 1
            input_size = np.array((sess.run(tf.shape(image_decode)[1]), sess.run(tf.shape(image_decode)[0])))
            print("--------original size ==", input_size )
        else:
            print("False : xml is not correspond to the img source")


        #raw rotation
        for item in range(0,4):
        #rotation the original pic
            img_origin_rot = rot90(image_decode, item)
            output_count += 1
            print('output_count =',output_count)
            encoded_origin_img_rot = tf.image.encode_jpeg(img_origin_rot)
            out_path = str(output_imagefile + str(output_count) + ".jpg")
            print("out_path ==",out_path)
            with tf.gfile.GFile(out_path, "wb") as f:
                f.write(encoded_origin_img_rot.eval())
            
            rotated_n_saved_image = tf.read_file(out_path)
            decoded = tf.image.decode_image(rotated_n_saved_image)
            size = np.array((sess.run(tf.shape(decoded)[1]), sess.run(tf.shape(decoded)[0])))
            print("--------size:", size )
            
            size_img = input_size
            input_size = size
        
        
        
        ##############
        # write correspond xml file for the rotated pic
        # read the original labelled xml
            if item == 0:
                input_tree = ET.ElementTree(file=xml_file)
            else:
                generated_filename = str(output_imagefile) + str(output_count-1)+ '.xml'
                print("generated_filename as input == ",generated_filename)
                input_tree = ET.ElementTree(file=generated_filename)

            input_root = input_tree.getroot()

            for child_of_root in input_root.iter("bndbox"):
                for i in range(4):
                    print("input")
                    print (child_of_root[i].tag , child_of_root[i].text)
                    if child_of_root[i].tag == 'xmin':
                        xmin = int(child_of_root[i].text)
                    if child_of_root[i].tag == 'ymin':
                        ymin = int(child_of_root[i].text)
                    if child_of_root[i].tag == 'xmax':
                        xmax = int(child_of_root[i].text)
                    if child_of_root[i].tag == 'ymax':
                        ymax = int(child_of_root[i].text)

            min_point = np.array((xmin, ymin))
            max_point = np.array((xmax, ymax))
            print(min_point, max_point)
            
            origin_min_point = np.array((min_point), dtype=np.float32)
            origin_max_point = np.array((max_point), dtype=np.float32)
            
            if item != 0:
            #### 3 steps of coordinate transformation###

                print("--translation--")
                t1_1, t1_2 = first_translation(origin_min_point, origin_max_point, size_img)
                print("--rotation--")
                r_1, r_2  = coord_transform(t1_1, t1_2, -math.pi/2)
                print("r_1:", r_1," | ","r_2:", r_2,"\n")
                print("--translation_back--\n")
                r_arr = np.array((r_1, r_2))
                print("r_arr ==", r_arr)

                t2_1, t2_2 = translate_back_to_rotated_img(r_arr, size_img)
            elif item == 0:
                t2_1 = origin_min_point
                t2_2 = origin_max_point
            print("t2_1 == ",t2_1)
            print("t2_2 == ",t2_2)
            ######

            out_min, out_max = output_min_and_max_points_of_rect(t2_1,t2_2)

            #####parse the correspond xml, img file and path
            if input_root[0].tag == 'folder':
                input_root[0].text = str(output_dir_name)
            else:
                print("Error happened")

            if input_root[1].tag == 'filename':
                input_root[1].text = str(str(output_count) + ".jpg")
            else:
                print("Error happened")
            
            if input_root[2].tag == 'path':
                ## reset the path setting
                input_root[2].text = str( os.path.abspath(output_imagefile) + "/" + str(output_count) + ".jpg")
            else:
                print("Error happened")
            
            for child_root in input_root.iter("size"):
                if item == 0:
                    child_root[0].text = str(size_img[0])
                    child_root[1].text = str(size_img[1])
                if item != 0:
                    child_root[0].text = str(size_img[1])
                    child_root[1].text = str(size_img[0])
                

            for child_of_root in input_root.iter("bndbox"):
                for i in range(4):
                    #print (child_of_root[i].tag , child_of_root[i].text)
                    if child_of_root[i].tag == 'xmin':
                        child_of_root[i].text = str(int(out_min[0]))
                    if child_of_root[i].tag == 'ymin':
                        child_of_root[i].text = str(int(out_min[1]))
                    if child_of_root[i].tag == 'xmax':
                        child_of_root[i].text = str(int(out_max[0]))
                    if child_of_root[i].tag == 'ymax':
                        child_of_root[i].text = str(int(out_max[1]))
            
            stri = str('./destination_images/' + str(output_count) + ".xml")
            input_tree.write(stri, encoding='utf-8', xml_declaration=False)
    
    
    
    ##############
    ##############


        #flipped upside down
        image_flipped = upside_down(image_decode)
        #preparing to write the proceed images 
        encoded_img = tf.image.encode_jpeg(image_flipped)
        
        #flipped rotation
        for items in range(0,4):
            img_rot = rot90(image_flipped, items)
            output_count += 1
            print('output_count =',output_count)
            encoded_img = tf.image.encode_jpeg(img_rot)
            out_path = output_imagefile + str(output_count) + ".jpg"
            print('flipped out path ==',out_path)
            with tf.gfile.GFile(out_path, "wb") as f:
                f.write(encoded_img.eval())
        
            rotated_flipped_saved_image = tf.read_file(out_path)
            flipped_decoded = tf.image.decode_image(rotated_flipped_saved_image)
            flipped_size = np.array((sess.run(tf.shape(flipped_decoded)[1]), sess.run(tf.shape(flipped_decoded)[0])))
            print("--------flipped size:", flipped_size )

            input_flipped_size_img = input_size
            input_size = flipped_size


            # write correspond xml file for the rotated pic
            # read the original labelled xml
            if items == 0:
                flipped_input_tree = ET.ElementTree(file= xml_file)
            elif items != 0:
                flipped_generated_filename = str(output_imagefile) + str(output_count-1)+ '.xml'
                print("flipped generated_filename as input == ",flipped_generated_filename)
                flipped_input_tree = ET.ElementTree(file= flipped_generated_filename)

            f_input_root = flipped_input_tree.getroot()

            
            for child_of_root in f_input_root.iter("bndbox"):
                for i in range(0,4):
                    print("flipped input :")
                    print (child_of_root[i].tag , child_of_root[i].text)
                    if child_of_root[i].tag == 'xmin':
                        f_xmin = int(child_of_root[i].text)
                    if child_of_root[i].tag == 'ymin':
                        f_ymin = int(child_of_root[i].text)
                    if child_of_root[i].tag == 'xmax':
                        f_xmax = int(child_of_root[i].text)
                    if child_of_root[i].tag == 'ymax':
                        f_ymax = int(child_of_root[i].text)

            f_min_point = np.array((f_xmin, f_ymin))
            f_max_point = np.array((f_xmax, f_ymax))
            print(f_min_point, f_max_point)
            
            f_origin_min_point = np.array((f_min_point), dtype=np.float32)
            f_origin_max_point = np.array((f_max_point), dtype=np.float32)
            
            if items != 0:
            #### 3 steps of coordinate transformation###
                print("-------------------flipped transformation--------------------")
                print("--translation--")
                f_t1_1, f_t1_2 = first_translation(f_origin_min_point, f_origin_max_point, input_flipped_size_img)
                print("--rotation--")
                f_r_1, f_r_2  = coord_transform(f_t1_1, f_t1_2, -math.pi/2)
                print("f_r_1:", f_r_1," | ","f_r_2:", f_r_2,"\n")
                print("--translation_back--\n")
                f_r_arr = np.array((f_r_1, f_r_2))
                print("f_r_arr ==", f_r_arr)
                f_t2_1, f_t2_2 = translate_back_to_rotated_img(f_r_arr, input_flipped_size_img)

            elif items == 0:

                if f_ymax < flipped_size[1]/2 :
                    f_ymax = f_ymax + 2*abs(f_ymax - flipped_size[1]/2)
                elif f_ymax > flipped_size[1]/2 :
                    f_ymax = f_ymax - 2*abs(f_ymax - flipped_size[1]/2)
                
                if f_ymin < flipped_size[1]/2 :
                    f_ymin = f_ymin + 2*abs(f_ymin - flipped_size[1]/2)
                elif f_ymin > flipped_size[1]/2 :
                    f_ymin = f_ymin - 2*abs(f_ymin - flipped_size[1]/2)

                
                f_t2_1 = np.array((f_xmin,f_ymin))
                f_t2_2 = np.array((f_xmax,f_ymax))
            
            print("f_t2_1 == ",f_t2_1)
            print("f_t2_2 == ",f_t2_2)
            ######

            f_out_min, f_out_max = output_min_and_max_points_of_rect(f_t2_1,f_t2_2)

            #####parse the correspond xml, img file and path
            if f_input_root[0].tag == 'folder':
                f_input_root[0].text = str(output_dir_name)
            else:
                print("Error happened")

            if f_input_root[1].tag == 'filename':
                f_input_root[1].text = str(str(output_count) + ".jpg")
            else:
                print("Error happened")
            
            if f_input_root[2].tag == 'path':
                f_input_root[2].text = str(os.path.abspath(output_imagefile) + "/"  + str(output_count) + ".jpg")
            else:
                print("Error happened")
            
            for child_root in f_input_root.iter("size"):
                if item == 0:
                    child_root[0].text = str(input_flipped_size_img[0])
                    child_root[1].text = str(input_flipped_size_img[1])
                if item != 0:
                    child_root[0].text = str(input_flipped_size_img[1])
                    child_root[1].text = str(input_flipped_size_img[0])
                

            for child_of_root in f_input_root.iter("bndbox"):
                for i in range(4):
                    #print (child_of_root[i].tag , child_of_root[i].text)
                    if child_of_root[i].tag == 'xmin':
                        child_of_root[i].text = str(int(f_out_min[0]))
                    if child_of_root[i].tag == 'ymin':
                        child_of_root[i].text = str(int(f_out_min[1]))
                    if child_of_root[i].tag == 'xmax':
                        child_of_root[i].text = str(int(f_out_max[0]))
                    if child_of_root[i].tag == 'ymax':
                        child_of_root[i].text = str(int(f_out_max[1]))
            
            f_stri = str( str(output_imagefile) + str(output_count) + ".xml")
            flipped_input_tree.write(f_stri, encoding='utf-8', xml_declaration=False)
        




    
    
    