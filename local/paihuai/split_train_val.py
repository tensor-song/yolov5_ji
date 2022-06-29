import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', type=str,
                    default='/project/train/src_repo/dataset/Annotations', help='input xml label path')
parser.add_argument('--txt_path', type=str,
                    default='/project/train/src_repo/dataset/ImageSets/Main', help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
# total_xml = os.listdir(xmlfilepath)
total_xml = []
for id in ['608', '609']:
  for i in os.listdir(os.path.join(xmlfilepath, id)):
    total_xml.append(os.path.join(id, i))


if not os.path.exists(txtsavepath):
  os.makedirs(txtsavepath)

num=len(total_xml)
lists=range(num)
print(num)
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in lists:
    name=total_xml[i][:-4]+'\n'
    ftrainval.write(name)
    if i%20 == 0:
        fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

# 953  982  992
# <size>
#     <width>1920</width>
#     <height>1080</height>
#     <depth/>
#   </size>
#   <segmented>0</segmented>
#   <object>
#     <name>head</name>
#     <occluded>0</occluded>
#     <bndbox>
#       <xmin>399.12</xmin>
#       <ymin>150.65</ymin>
#       <xmax>418.62</xmax>
#       <ymax>171.45</ymax>
#     </bndbox>
#     <attributes>
#       <attribute>
#         <name>sex</name>
#         <value>male</value>
#       </attribute>
#       <attribute>
#         <name>age</name>
#         <value>41-50</value>
#       </attribute>
#     </attributes>
#   </object>