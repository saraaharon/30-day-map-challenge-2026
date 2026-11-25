import numpy as np, cv2
from sklearn.cluster import KMeans
im=cv2.imread('/mnt/user-data/uploads/unnamed.jpg')
sm=cv2.pyrMeanShiftFiltering(im,8,18)
lab=cv2.cvtColor(sm,cv2.COLOR_BGR2LAB).astype(np.float32)
w=np.array([0.7,1.5,1.5],np.float32)
f=(lab-np.array([0,128,128],np.float32))*w
K=8
km=KMeans(K,n_init=6,random_state=1).fit(f[::3,::3].reshape(-1,3))
lbl=km.predict(f.reshape(-1,3)).reshape(lab.shape[:2])
lbl=cv2.medianBlur(lbl.astype(np.uint8),7).astype(int)
# mean color per class from original
rgbim=cv2.cvtColor(sm,cv2.COLOR_BGR2RGB)
rgb=np.array([rgbim[lbl==i].mean(0) for i in range(K)]).astype(np.uint8)
cnt=np.bincount(lbl.ravel(),minlength=K)/lbl.size
for i in np.argsort(rgb.sum(1)): print(i,rgb[i],round(cnt[i]*100,1))
np.save('lbl.npy',lbl); np.save('rgb.npy',rgb)
cv2.imwrite('class.png',rgb[lbl][:,:,::-1])
