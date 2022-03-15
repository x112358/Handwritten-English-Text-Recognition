from PIL import Image
import sys
import os 

data_path = '../test_data/'
output_path = '../data/'

def resize_const():
	list = os.listdir(data_path)
	num = len(list)
	for i in range(num):
		if os.path.isdir(data_path + list[i]):
			dir_list = os.listdir(data_path + list[i])
			dir_num = len(dir_list)
			for j in range(dir_num):
				type = os.path.splitext(data_path + list[i] + '/' + dir_list[j])[-1]
				if(type == ".png"):
					im = Image.open(data_path + list[i] + '/' + dir_list[j])
					#read image size
					(x,y) = im.size
					#define standard width
					x_s = 393 
					y_s = 98 
					#resize image with high-quality
					out = im.resize((x_s,y_s),Image.ANTIALIAS)
					out.save(output_path + 'adjust_' + dir_list[j])
		else:
			type = os.path.splitext(data_path + list[i])[-1]
			if(type == ".png"):
				im = Image.open(data_path + list[i])
				#read image size
				(x,y) = im.size
				#define standard width
				x_s = 393 
				y_s = 98 
				#resize image with high-quality
				out = im.resize((x_s,y_s),Image.ANTIALIAS)
				out.save(output_path + 'adjust_' + list[i])
	
def resize_ratio(width):
	list = os.listdir(data_path)
	num = len(list)
	for i in range(num):
		type = os.path.splitext(data_path + list[i])[-1]
		print(type)
		if(type == ".png"):
			im = Image.open(data_path + list[i])
			#read image size
			(x,y) = im.size
			#define standard width
			x_s = int(width) * x
			y_s = y * x_s / x
			#resize image with high-quality
			out = im.resize((int(x_s),int(y_s)),Image.ANTIALIAS)
			out.save(output_path + 'adjust_' + list[i])
	
if __name__ == '__main__':
	argc = len(sys.argv)
	if argc < 2:
		resize_const()
	else:
		resize_ratio(int(sys.argv[1]))

	