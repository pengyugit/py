import cv2
import matplotlib.pyplot as plt 
import utils
from text_proposal_connector_oriented import TextProposalConnectorOriented
import numpy as np
from keras.models import load_model


imgpath = r'img/img_4375.jpg'
#precess img
img = cv2.imread(imgpath)
h,w,c = img.shape
#zero-center by mean pixel 
m_img = img - utils.IMAGE_MEAN
m_img = np.expand_dims(m_img,axis=0)


basemodel=load_model('my_model.h5')
cls,regr,cls_prod = basemodel.predict(m_img)


anchor = utils.gen_anchor((int(h/16),int(w/16)),16)

bbox = utils.bbox_transfor_inv(anchor,regr)
bbox = utils.clip_box(bbox,[h,w])

#score > 0.7
fg = np.where(cls_prod[0,:,1]>0.7)[0]
select_anchor = bbox[fg,:]
select_score = cls_prod[0,fg,1]
select_anchor = select_anchor.astype('int32')

#filter size
keep_index = utils.filter_bbox(select_anchor,16)


#nsm
select_anchor = select_anchor[keep_index]
select_score = select_score[keep_index]
select_score = np.reshape(select_score,(select_score.shape[0],1))
nmsbox = np.hstack((select_anchor,select_score))
keep = utils.nms(nmsbox,0.3)
select_anchor = select_anchor[keep]
select_score = select_score[keep]

#text line
textConn = TextProposalConnectorOriented()
text = textConn.get_text_lines(select_anchor,select_score,[h,w])

# for i in select_anchor:
#         cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(255,0,0),2)

text= text.astype('int32')

for i in text:
    cv2.line(img,(i[0],i[1]),(i[2],i[3]),(255,0,0),2)
    cv2.line(img,(i[0],i[1]),(i[4],i[5]),(255,0,0),2)
    cv2.line(img,(i[6],i[7]),(i[2],i[3]),(255,0,0),2)
    cv2.line(img,(i[4],i[5]),(i[6],i[7]),(255,0,0),2)

plt.imshow(img)
plt.show()
