from cv2 import Sobel,CV_64F,connectedComponentsWithStats,RETR_TREE,findContours,drawContours,medianBlur,imwrite,imread,\
    GaussianBlur,CHAIN_APPROX_NONE,cvtColor,COLOR_BGR2GRAY
import numpy as np
from tqdm import tqdm
import copy
import imutils
from PyQt5 import QtWidgets
from os import getcwd
import sys
from PyQt5.QtWidgets import QFileDialog
def histogram (i,bits):
    q,p = np.shape(i)
    bars = np.zeros(bits)
    for value in range(0,bits):
        bars[value] = sum(sum(1*(i == value)))
    return bars/(p*q)

def HCT(imag):
    """
    histogram statistics
    """
    bars_g = histogram(imag,256)
    """
    basic data
    """
    mean_GRAY = np.zeros(0)
    for i in range(0, 256):
        mean_GRAY = np.append(mean_GRAY, bars_g[i] * (i))
    Le = round(sum(mean_GRAY))
    Le = np.array(Le,dtype='uint')
    L_min = np.min(imag)
    L_max = np.max(imag)
    ksi_0 = 1 / 256
    ksi_l = 2 * ksi_0 * (Le - L_min)/ (L_max - L_min)
    ksi_r = 2 * ksi_0 * (L_max - Le)/ (L_max - L_min)
    Lk_1 = Le
    Lk_2 = Le - 1
    i = Le
    k = Le - 1
    N_bars = np.zeros(256)
    imag_new = np.zeros(np.shape(imag))
    imaa = np.zeros(np.shape(imag))
    t = 0
    '''
    algorithm realization
        forward searching
    '''
    for n in (range(1000)):
        if bars_g[i] < ksi_r:
            P_1 = bars_g[i]
            CP_1 = P_1
            for r in range(0, 256):
                if i + r + 1 <= 255:
                    P_1 = np.append(P_1, bars_g[i + r + 1])
                    CP_1 = np.append(CP_1, sum(P_1))
                else:
                    P_1 = np.append(P_1, bars_g[i])
                    CP_1 = np.append(CP_1, sum(P_1))
                if i + r + 1 >= 255:
                    N_bars[Lk_1] = CP_1[r + 1]
                    break
                else:
                    if CP_1[r + 1] > ksi_r:
                        if [CP_1[r] + CP_1[r + 1]] < 2 * ksi_r:
                            N_bars[Lk_1] = CP_1[r + 1]
                            r = r + 1
                        else:
                            N_bars[Lk_1] = CP_1[r]
                        break
            for rr in range(1, r + 2):
                imag_new = imag_new + Lk_1 * (imag == (i + rr - 1))
            if r != 0:
                ima = np.zeros(np.shape(imag))
                for i_i in range(1, r + 2):
                    ima = ima + 255 * (imag == (i + i_i - 1))
                for i_i in range(1, r + 2):
                    imaa = imaa + 255 * (imag == (i + i_i - 1))
                t = t + 1
            i = i + r + 1
        else:
            N_bars[Lk_1] = bars_g[i]
            imag_new = imag_new + Lk_1 * (imag == (i))
            i = i + 1
        if i > 255:
            break
        if Lk_1 == 255:
            break
        else:
            Lk_1 = Lk_1 + 1
    '''
    algorithm realization
        backward searching
    '''
    for n in (range(1000)):
        if bars_g[k] < ksi_l:
            P_2 = bars_g[k]
            CP_2 = P_2
            for r in range(0, 256):
                if k - r > 0:
                    P_2 = np.append(P_2, bars_g[k - r - 1])
                    CP_2 = np.append(CP_2, sum(P_2))
                else:
                    P_2 = np.append(P_2, bars_g[k])
                    CP_2 = np.append(CP_2, sum(P_2))
                if k - r == 1:
                    if CP_2[r + 1] > ksi_l:
                        if [CP_2[r] + CP_2[r + 1]] < 2 * ksi_l:
                            N_bars[Lk_2] = CP_2[r + 1]
                            r = r + 1
                        else:
                            N_bars[Lk_2] = CP_2[r]
                            N_bars[Lk_2 - 1] = bars_g[k - r - 1]
                    break
                if CP_2[r + 1] > ksi_l:
                    if [CP_2[r] + CP_2[r + 1]] < 2 * ksi_l:
                        N_bars[Lk_2] = CP_2[r + 1]
                        r = r + 1
                    else:
                        N_bars[Lk_2] = CP_2[r]
                    break
            for rr in range(1, r + 2):
                imag_new = imag_new + (Lk_2) * (imag == (k - rr + 1))
            if r != 0:
                ima = np.zeros(np.shape(imag))
                for i_i in range(1, r + 2):
                    ima = ima + 255 * (imag == (k - i_i - 1))
                for i_i in range(1, r + 2):
                    imaa = imaa + 255 * (imag == (k - i_i - 1))
                t = t + 1
            k = k - r - 1
            if k < 0:
                break
            if k == 0:
                Lk_2 = Lk_2 - 1
                N_bars[Lk_2] = bars_g[k]
                imag_new = imag_new + Lk_2 * (imag == (k))
                break
        else:
            N_bars[Lk_2] = bars_g[k]
            imag_new = imag_new + Lk_2 * (imag == (k))
            k = k - 1
            if k < 0:
                break
        if Lk_2 == 0:
            break
        else:
            Lk_2 = Lk_2 - 1
    '''
    Histogram linear stretch
    '''
    imag_new = np.array(imag_new, dtype='uint8')
    ka = 255 / (imag_new.max() - imag_new.min())
    imag_afterhctlsf = ka * (imag_new - imag_new.min())
    imag_afterhctls = np.array(imag_afterhctlsf, dtype='uint8')
    return imag_afterhctls, imaa

def grad(im):
    sobelx = Sobel(im, CV_64F, 1, 0, ksize=3)
    sobely = Sobel(im, CV_64F, 0, 1)
    gm = abs(sobely)+abs(sobelx)
    return gm

def select(mask1,threshold):
    q,p = np.shape(mask1)
    number, label, boudingbox, center = connectedComponentsWithStats(mask1)
    m = np.linspace(1, number - 1, number - 1)
    m = [boudingbox[1:, 4] > threshold ] * m
    m2 = m.flatten()
    m22 = filter(lambda x: x != 0, m2)
    m22 = [L2 for L2 in m22]
    m22 = np.array(m22,dtype = "int")
    len_m22 = len(m22)
    label2 = np.zeros(label.shape)

    for i in (range(0, len_m22)):
        mi = (label == m22[i])
        label2[mi] = 255
        if (round(boudingbox[m22[i], 3])>0.9*p):
            label2[mi] = 0
        if (round(boudingbox[m22[i], 2])>0.9*q):
            label2[mi] = 0
    label2 = np.array(label2, dtype="uint8")
    return label2

def recovery_self(mask_input, tar_input, auxi_input):
    '''
    rectangular、center、max distance、r
    '''
    number_input, label_input, boudingboxinput, center_input = connectedComponentsWithStats(mask_input)
    p_input,q_input = np.shape(mask_input)

    for i in tqdm(range(1, number_input)):
        '''
        distance of image
        '''
        r = round((boudingboxinput[i][2] + boudingboxinput[i][3]) * 0.2)  ######
        mask_input_2 = copy.deepcopy(mask_input)
        mask_input_2[label_input != i] = 0

        mask = mask_input_2
        tar = tar_input
        auxi = auxi_input
        number, label, boudingbox, center = connectedComponentsWithStats(mask)
        q, p = np.shape(label)
        x = np.linspace(0, q - 1, q)
        y = np.linspace(0, p - 1, p)
        nx, ny = np.meshgrid(y, x)
        distance = abs(nx - center[0,0]*np.ones_like(label)) + abs(ny - center[0,1]*np.ones_like(label))
        distance = 1 * (label == 1) * distance
        mask_i = 1 - 1 * (label == 1)
        mask_ii = mask_i - 1

        maski = 255 * (label == 1)
        maski = np.array(maski, dtype="uint8")
        masi_deep = copy.deepcopy(maski)
        temp = np.zeros(maski.shape, np.uint8) * 255
        hierarchy, contours = findContours(maski, RETR_TREE, CHAIN_APPROX_NONE)
        contours = contours if imutils.is_cv3() else hierarchy
        drawContours(masi_deep, contours, -1, (255, 255, 255), 2)
        drawContours(temp, contours, -1, (255, 255, 255), 1)
        masi_deep = masi_deep + maski
        masi_deep[masi_deep > 0] = 255
        masi_deep[maski == 255] = 0
        masi_deep = masi_deep + temp
        masi_deep = np.array(masi_deep, dtype="uint8")
        for ii in (range(boudingboxinput[i][4])):
            tar_i = tar * mask_i
            tar_i = tar_i + mask_ii
            xx = 1 * (distance == distance.max()) * nx
            if xx.max() == 0:
                xx = 1 * (distance == distance.max()) * ny
            beginx1, beginy1 = np.nonzero(xx)
            if xx.max() == 0:
                beginx1 = [0]
                beginy1 = [0]
            beginx = beginx1[0]
            beginy = beginy1[0]
            zongxiao = beginx - r
            zongda = beginx + r
            hengxiao = beginy - r
            hengda = beginy + r
            if hengxiao < 0:
                hengxiao = 0
            elif hengda > p - 1:
                hengda = p - 1
            elif zongda > q - 1:
                zongda = q - 1
            elif zongxiao < 0:
                zongxiao = 0
            zongxiao = np.array(zongxiao,dtype='int')
            zongda = np.array(zongda,dtype='int')
            hengxiao =np.array(hengxiao,dtype='int')
            hengda = np.array(hengda,dtype='int')
            k_aux = auxi[zongxiao: zongda, hengxiao: hengda]
            k_aux = k_aux.flatten()
            if len(k_aux) == 0:
                break
            k_tar = tar[zongxiao: zongda, hengxiao: hengda]
            k_tar = k_tar.flatten()
            k_tar2 = tar_i[zongxiao: zongda, hengxiao: hengda]
            k_tar2 = k_tar2.flatten()
            k_tar_f1 = filter(lambda x: x != -1, k_tar2)
            k_tar_f1 = [L2 for L2 in k_tar_f1]

            mean_r = np.mean(k_aux)
            sigema_r = np.std(k_aux, ddof=1)
            sigema_t = np.std(k_tar_f1, ddof=1)
            mean_t = np.mean(k_tar)
            if sigema_r == 0:
                kaa = 0
            else:
                kaa = sigema_t / sigema_r
            zhongjian = kaa * auxi[beginx, beginy] + mean_t - kaa * mean_r
            if zhongjian > 255:
                zhongjian = 255
            elif zhongjian < 0:
                zhongjian = 0
            if tar[beginx, beginy] > auxi[beginx, beginy] & auxi[beginx, beginy] > zhongjian:
                tar[beginx, beginy] = auxi[beginx, beginy]
            elif tar[beginx, beginy] < auxi[beginx, beginy] & auxi[beginx, beginy] < zhongjian:
                tar[beginx, beginy] = auxi[beginx, beginy]
            else:
                tar[beginx, beginy] = zhongjian
            mask_i[beginx, beginy] = 1
            distance[beginx, beginy] = 0
        tar2 = GaussianBlur(tar, (3, 3), 1.6, dst=None, sigmaY=1.6)
        tar[masi_deep == 255] = tar2[masi_deep == 255]
    return tar

def OpensaveImag():
    ImPath2 = QtWidgets.QFileDialog.getOpenFileName(None, "select folder", DefaultImFolder)  # 这个语句有些邪门
    if ImPath2 != '':
        return ImPath2
    else:
        print('Please reselect the path')


def OpenInputImag():
    ImPath1 = QtWidgets.QFileDialog.getOpenFileName(None, "select folder", DefaultImFolder)  # 这个语句有些邪门
    if ImPath1 != '':
        return ImPath1
    else:
        print('Please reselect the path')

def selectPath():
    str_path = QFileDialog.getExistingDirectory(None,"Selecting the folder","")
    return str_path

if __name__ == '__main__':
    CurFolder = getcwd()
    DefaultImFolder = CurFolder
    app = QtWidgets.QApplication(sys.argv)
    pathinput = OpenInputImag()
    pathoutput = selectPath()
    img = imread(str(pathinput[0]))
    if len(img.shape) == 2:
        print('Grayscale image')
        input = img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        print('RGB image')
        input = cvtColor(img, COLOR_BGR2GRAY)
    else:
        print('Other type of image')
        input = img[:, :, 2]
    imag, imcap = HCT(input)
    imwrite(str(pathoutput)+'/testhct.png', imag)
    p, q = np.shape(imcap)
    imcap[imcap > 0] = 255
    imcap = np.array(imcap, dtype="uint8")
    grad_input = grad(input)
    grad_hctls = grad(imag)
    graddelta1 = grad_hctls - grad_input
    graddelta1[graddelta1 >= 0] = 0
    graddelta1[graddelta1 < 0] = 255
    graddelta = np.array(graddelta1, dtype="uint8")
    graddelta = medianBlur(graddelta, 3)
    '''
    Connected domain
    Reverse fill
    '''
    number, label, boudingbox, center = connectedComponentsWithStats(graddelta)
    for i in tqdm(range(1, number)):
        mi = (label == i)
        if boudingbox[i][4] > 0:
            label[mi] = 255
        else:
            label[mi] = 0
    label = np.array(label, dtype="uint8")

    label = 255 - label
    rnumber, rlabel, rboudingbox, rcenter = connectedComponentsWithStats(label)
    if rboudingbox[1][2] * rboudingbox[1][2] > 0.9 * p * q:
        rlabel[rlabel > 1] = 0
    else:
        for i in tqdm(range(1, rnumber)):
            mm = (rlabel == i)
            if rboudingbox[i][4] < 50:
                rlabel[mm] = 0
    rlabel[rlabel > 0] = 255
    rlabel = np.array(rlabel, dtype="uint8")
    label = 255 - rlabel

    im_con1_number, im_con1_label, im_con1_boudingbox, im_con1_center = connectedComponentsWithStats(imcap)
    medium = copy.deepcopy(im_con1_label)
    medium[label == 0] = 0
    medium_array1 = medium.flatten()
    medium_array2 = list(set(medium_array1))
    medium_array2.remove(0)
    len_medium = len(medium_array2)
    m = np.zeros(np.shape(im_con1_label))
    for i in range(len_medium):
        m[im_con1_label == medium_array2[i]] = 255
    m = np.array(m, dtype="uint8")
    m[m > 0] = 255
    m = medianBlur(m, 3)

    m = 255 - m
    rnumber, rrlabel, rboudingbox, rcenter = connectedComponentsWithStats(m)
    if rboudingbox[1][2] * rboudingbox[1][2] > 0.9 * p * q:
        rrlabel[rrlabel > 1] = 0
        pass
    else:
        for i in tqdm(range(1, rnumber)):
            mm = (rrlabel == i)
            if rboudingbox[i][4] < 50:
                rrlabel[mm] = 0
    rrlabel[rrlabel > 0] = 255
    rrlabel = np.array(rrlabel, dtype="uint8")
    rrrlabel = 255 - rrlabel

    im_T = recovery_self(rrrlabel, imag, input)
    imwrite(str(pathoutput)+'/output.png', im_T)
    imwrite(str(pathoutput)+'/region.png', rrrlabel)
