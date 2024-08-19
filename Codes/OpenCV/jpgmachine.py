from glob import glob                                                           
import cv2 

path = 'C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/resized/summer/conv/'
pngs = glob('./*.png')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(path +j[:-3] + 'jpg', img)