# import codecs
# import os, sys
# from PIL import Image

# file = codecs.open('train_chars.txt', 'r', "utf-8")
# train_files = file.readlines()
# file.close()

# size = 128, 128
# # let's go through each directory and read images within it
# for line in train_files:
#     if line.strip() == "":
#         continue
#     # extract label number of subject from im_file
#     # labels.append(label)
#     # extract filename from im_file
#     image_name = line[:line.rfind(";")]
#     print('image_name', image_name)

#     outfile = 'resized_chars' + '/' + image_name
#     basewidth = 100
#     img = Image.open(image_name)
#     wpercent = (basewidth/float(img.size[0]))
#     hsize = int((float(img.size[1])*float(wpercent)))
#     img = img.resize((basewidth,hsize), Image.ANTIALIAS)
#     # print(outfile)
#     img.save(outfile) 
#     # try:
#     # im = Image.open(image_name)
#     # im.thumbnail(size, Image.ANTIALIAS)
#     # print(im)
#     # im.save(outfile, "JPEG")
#     # # except IOError:
#         # print ("cannot create thumbnail for " + image_name)


import os
for path in ['new_chars1/a/1596029053079_8.jpg',
'new_chars1/a/1596029053079_9.jpg',
'new_chars1/b/b1.jpg',
'new_chars1/b/image18.jpeg',
'new_chars1/b/IMG_0938.JPG',
'new_chars1/b/IMG_0944.JPG',
'new_chars1/b/IMG_2615.JPG',
'new_chars1/b/IMG_2617.JPG',
'new_chars1/b/IMG_3211.JPG',
'new_chars1/b/IMG_4127.JPG',
'new_chars1/b/IMG_4128.JPG',
'new_chars1/b/IMG_4133.JPG',
'new_chars1/c/1595291455058_1.jpg',
'new_chars1/c/c4.jpg',
'new_chars1/c/c6.jpg',
'new_chars1/c/c7.jpg',
'new_chars1/c/image15_1.jpeg',
'new_chars1/c/image17.jpeg',
'new_chars1/c/IMG_3296.JPG',
'new_chars1/d/1595291463749_1.jpg',
'new_chars1/d/1595291463749_2.jpg',
'new_chars1/d/1595291463749_3.jpg',
'new_chars1/d/1595291463749_4.jpg',
'new_chars1/d/1595291463749_5.jpg',
'new_chars1/d/d2.jpg',
'new_chars1/d/d4.jpg',
'new_chars1/d/IMG_0985.JPG',
'new_chars1/d/IMG_0991.JPG',
'new_chars1/d/IMG_2630.JPG',
'new_chars1/d/IMG_2632.JPG',
'new_chars1/d/IMG_2633.JPG',
'new_chars1/d/IMG_2634.JPG',
'new_chars1/d/IMG_2635.JPG',



'new_chars1/e/1595291487079_4.jpg',

'new_chars1/e/1596029222483_9.jpg',


'new_chars1/e/1595291487079_5.jpg',
'new_chars1/e/1596029222483_2.jpg',
'new_chars1/e/image44.jpeg',
'new_chars1/e/1595291487079_6.jpg',
'new_chars1/e/IMG_2649.JPG',




'new_chars1/g/IMG_2672.JPG',
'new_chars1/g/IMG_2673.JPG',
'new_chars1/g/1595291519333_1.jpg',
'new_chars1/g/1595291519333_3.jpg',
'new_chars1/g/image1.jpeg',
'new_chars1/g/1595291519333_4.jpg',

'new_chars1/g/IMG_3165.JPG',

'new_chars1/g/IMG_4212.JPG',

'new_chars1/g/IMG_4214.JPG',
'new_chars1/g/IMG_4215.JPG',



'new_chars1/h/1595291528213_1.jpg',


'new_chars1/h/1595291528213_2.jpg',
'new_chars1/h/1596029291646_7.jpg',
'new_chars1/h/IMG_4222.JPG',
'new_chars1/h/image22_1.jpeg',
'new_chars1/h/image24.jpeg',
'new_chars1/h/IMG_4219.JPG',
'new_chars1/h/IMG_3166.JPG',
'new_chars1/h/IMG_2677.JPG',
'new_chars1/h/IMG_4223.JPG',


'new_chars1/i/1595291537088_1(1).jpg',

'new_chars1/i/image14_1.jpeg',

'new_chars1/i/IMG_4237.JPG',
'new_chars1/k/1595291547601_8.jpg',
'new_chars1/k/1596029337891_1.jpg',
'new_chars1/k/1596029337891_2.jpg',
'new_chars1/k/1596029337891_5.jpg',
'new_chars1/k/1596029337891_6.jpg',
'new_chars1/k/1596029337891_7.jpg',
'new_chars1/k/1596029337891_8.jpg',
'new_chars1/k/1596029337891_9.jpg',
'new_chars1/k/image47.jpeg',
'new_chars1/k/IMG_2699.JPG',
'new_chars1/k/IMG_2701.JPG',
'new_chars1/k/IMG_2700.JPG',
'new_chars1/k/IMG_3121.JPG',
'new_chars1/k/IMG_3273.JPG',
'new_chars1/k/IMG_3357.JPG',
'new_chars1/k/IMG_4241.JPG',




'new_chars1/l/1595291554542_7.jpg',
'new_chars1/l/1595291554542_9.jpg',
'new_chars1/l/1596029362056_1.jpg',
'new_chars1/l/1596029362056_9.jpg',
'new_chars1/l/IMG_2703.JPG',
'new_chars1/l/IMG_2704.JPG',
'new_chars1/l/IMG_3244.JPG',

'new_chars1/l/IMG_3299.JPG',
'new_chars1/l/IMG_4247.JPG',
'new_chars1/l/l1.jpg',

'new_chars1/l/l2.jpg',


'new_chars1/m/1595291562486_1.jpg',

'new_chars1/m/1595291562486_4.jpg',

'new_chars1/m/1595291562486_5.jpg',
'new_chars1/m/1595291562486_6.jpg',
'new_chars1/m/IMG_1064.JPG',
'new_chars1/m/image29_1.jpeg',
'new_chars1/m/image43.jpeg',
'new_chars1/m/IMG_3186.JPG',
'new_chars1/m/IMG_1067.JPG',
'new_chars1/m/IMG_3142.JPG',
'new_chars1/m/IMG_1081.JPG',
'new_chars1/m/IMG_4257.JPG',
'new_chars1/m/IMG_4258.JPG',

'new_chars1/m/IMG_4259.JPG',



'new_chars1/n/1595291568511_1.jpg',
'new_chars1/n/1595291568511_6.jpg',
'new_chars1/n/1595291568511_3.jpg',
'new_chars1/n/1595291568511_5.jpg',
'new_chars1/n/1595291568511_4.jpg',
'new_chars1/n/1596029402375_2.jpg',
'new_chars1/n/1596029402375_9.jpg',
'new_chars1/n/image46.jpeg',
'new_chars1/n/IMG_1057.JPG',
'new_chars1/n/image13.jpeg',
'new_chars1/n/IMG_1060.JPG',
'new_chars1/n/IMG_1063.JPG',


'new_chars1/n/IMG_3129.JPG',
'new_chars1/n/IMG_3149.JPG',
'new_chars1/n/IMG_3216.JPG',
'new_chars1/n/IMG_4270.JPG',
'new_chars1/n/IMG_4267.JPG',





'new_chars1/o/image49.jpeg',
'new_chars1/o/image6.jpeg',
'new_chars1/o/IMG_1117.JPG',
'new_chars1/o/IMG_4279.JPG',
'new_chars1/o/IMG_4280.JPG',
'new_chars1/o/o1.jpg',
'new_chars1/o/o3.jpg',
'new_chars1/o/o7.jpg',



'new_chars1/p/1595291684992_1.jpg',
'new_chars1/p/1595291684992_7.jpg',
'new_chars1/p/1595291684992_3.jpg',
'new_chars1/p/1595291684992_8.jpg',
'new_chars1/p/1595291684992_9.jpg',
'new_chars1/p/1596029468602_1.jpg',
'new_chars1/p/1596029468602_2.jpg',
'new_chars1/p/1596029468602_7.jpg',
'new_chars1/p/1596029468602_6.jpg',
'new_chars1/p/IMG_1132.JPG',
'new_chars1/p/IMG_3168.JPG',
'new_chars1/p/IMG_3275.JPG',
'new_chars1/p/IMG_4328.JPG',
'new_chars1/p/p5.jpg',






'new_chars1/q/1595291693528_1.jpg',
'new_chars1/q/1595291693528_3.jpg',
'new_chars1/q/1595291693528_4.jpg',
'new_chars1/q/image21.jpeg',
'new_chars1/q/1595291693528_6.jpg',
'new_chars1/q/IMG_2749.JPG',
'new_chars1/q/IMG_3124.JPG',
'new_chars1/q/IMG_3208.JPG',
'new_chars1/q/IMG_4341.JPG',
'new_chars1/q/IMG_4342.JPG',
'new_chars1/q/q1.jpg',



'new_chars1/r/1595291700256_5.jpg',
'new_chars1/r/1595291700256_6.jpg',
'new_chars1/r/image11_1.jpeg',
'new_chars1/r/image13.jpeg',
'new_chars1/r/image20.jpeg',
'new_chars1/r/IMG_1157.JPG',
'new_chars1/r/IMG_1163.JPG',
'new_chars1/r/IMG_2758.JPG',
'new_chars1/r/IMG_3209.JPG',
'new_chars1/r/IMG_4344.JPG',
'new_chars1/r/IMG_4346.JPG',
'new_chars1/r/IMG_4349.JPG',





'new_chars1/s/1595291707263_7.jpg',
'new_chars1/s/1595291707263_9.jpg',
'new_chars1/s/1596029525499_1.jpg',
'new_chars1/s/1596029525499_8.jpg',
'new_chars1/s/IMG_2766.JPG',
'new_chars1/s/IMG_3372.JPG',
'new_chars1/s/IMG_3345.JPG',
'new_chars1/s/IMG_4352.JPG',





'new_chars1/t/1595291714105_6.jpg',
'new_chars1/t/1595291714105_5.jpg',
'new_chars1/t/1595291714105_4.jpg',
'new_chars1/t/IMG_1155.JPG',
'new_chars1/t/IMG_2774.JPG',
'new_chars1/t/IMG_2775.JPG',
'new_chars1/t/IMG_1158.JPG',
'new_chars1/t/IMG_3217.JPG',
'new_chars1/t/IMG_2777.JPG',
'new_chars1/t/IMG_2779.JPG',
'new_chars1/t/IMG_3346.JPG',
'new_chars1/t/IMG_4362.JPG',
'new_chars1/t/IMG_4364.JPG',
'new_chars1/t/IMG_4366.JPG',
'new_chars1/t/IMG_4368.JPG',
'new_chars1/t/IMG_4369.JPG',
'new_chars1/t/t3.jpg',




'new_chars1/u/1595291721215_2.jpg',
'new_chars1/u/1595291721215_4.jpg',
'new_chars1/u/IMG_2785.JPG',
'new_chars1/u/IMG_3284.JPG',
'new_chars1/u/image3.jpeg',
'new_chars1/u/IMG_4371.JPG',
'new_chars1/u/IMG_4375.JPG',
'new_chars1/u/IMG_4372.JPG',
'new_chars1/u/IMG_4376.JPG',
'new_chars1/u/IMG_4377.JPG',


'new_chars1/v/1595291734199_1.jpg',
'new_chars1/v/1595291734199_3.jpg',
'new_chars1/v/1596029612591_1.jpg',
'new_chars1/v/IMG_2805.JPG',
'new_chars1/v/IMG_3134.JPG',
'new_chars1/v/v4.jpg',



'new_chars1/x/1595291741138_9.jpg',
'new_chars1/x/IMG_3258.JPG',
'new_chars1/x/IMG_3343.JPG',
'new_chars1/x/x5.jpg',
'new_chars1/x/x2.jpg',


'new_chars1/y/1595291746398_5.jpg',
'new_chars1/y/1596029642619_2.jpg',
'new_chars1/y/image1_1.jpeg',
'new_chars1/y/IMG_3199.JPG',
'new_chars1/y/IMG_3228.JPG',
'new_chars1/y/IMG_4406.JPG',
'new_chars1/y/IMG_4407.JPG',
'new_chars1/y/IMG_4412.JPG',
'new_chars1/y/IMG_4413.JPG',
'new_chars1/y/IMG_4414.JPG',
'new_chars1/y/y4.jpg',
'new_chars1/y/y5.jpg',


'new_chars1/â/1595291434830_1.jpg',
'new_chars1/â/1595291434830_2.jpg',

]:
    if (os.path.isfile(path)):
        os.remove(path)
print("File Removed!")