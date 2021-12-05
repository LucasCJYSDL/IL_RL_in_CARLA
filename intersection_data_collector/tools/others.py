
import cv2


def render_image(name, img):
    """using cv2
    
    Arguments:
        name {str}
        img {np.array}
    """
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(1)



import csv

def create_csv(path, csv_head):

    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)

def write_csv(path, data_row):

    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


