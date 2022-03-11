# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 00:07:44 2020

@author: 15634
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, argrelmax
from PIL import Image
from priorityqueue import MinHeapPriorityQueue
from vector import Vector
from math import cos, sin, atan2, sqrt, pi ,radians, degrees,acos,log
import cv2 as  cv
import random
from pointinster  import doIntersect, Point
import cairo

def get_convex(draw_line):   
    for c in range(len(draw_line)):
        # 凸包检测
        points = cv.convexHull(draw_line[c])
        total = len(points)
        draw_line = np.zeros((total,2)).astype(np.int)
        for i in range(len(points)):
            draw_line[i][:]=points[i % total]
        draw_line = np.array(draw_line, dtype=np.int32)
    return draw_line

def Interpolation(draw_line,num):
    draw_line_inter = []
    class_size = num
    for i in range(0,len(draw_line)):
        draw_line_v = Vector([draw_line[i][0],draw_line[i][1]])
        if i == len(draw_line)-1:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[0][0]-draw_line[i][0]),(draw_line[0][1]-draw_line[i][1])])
        else:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[i+1][0]-draw_line[i][0]),(draw_line[i+1][1]-draw_line[i][1])])            
        if len(draw_line_inter) == class_size:
            break
        elif right_v.magnitude() == 0 or left_v.magnitude() == 0 or left_v.angle_with(right_v) > 2.8 :
            draw_line_inter.append([draw_line[i][0],draw_line[i][1]])
        else:
            draw_line_left =  draw_line_v.plus(left_v.times_scalar(0.25)).convert_list()
            draw_line_right =  draw_line_v.plus(right_v.times_scalar(0.25)).convert_list()
            if draw_line_left != draw_line_right:     
                draw_line_inter.append(draw_line_left)
                draw_line_inter.append(draw_line_right)
            else:
                draw_line_inter.append(draw_line_left)
    if len(draw_line_inter) != len(draw_line) and len(draw_line_inter) < class_size:
        draw_line_inter_list =  draw_line_inter
        draw_line_inter = Interpolation(np.array(draw_line_inter),class_size)
        if len(draw_line_inter) > class_size :
            return draw_line_inter_list
        else:
            return draw_line_inter
    else:
        return draw_line_inter

def remove_same_list(draw_line):
    seen = set()
    newlist = []
    for item in draw_line:
        t = tuple(item)
        if t not in seen:
            newlist.append(item)
            seen.add(t)
    return newlist

def shrink_polygon(draw_line,distant,num):
    draw_line_shrink = []
    draw_line_shrink_oringin = []
    array_2 = np.zeros((2))
    center = np.zeros((2))
    center[0],center[1] =  centroid(draw_line)
    for i in range(0,draw_line.shape[0]):
        if i == draw_line.shape[0]-1:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[0][0]-draw_line[i][0]),(draw_line[0][1]-draw_line[i][1])])
        else:
            left_v = Vector([(draw_line[i-1][0]-draw_line[i][0]),(draw_line[i-1][1]-draw_line[i][1])])
            right_v = Vector([(draw_line[i+1][0]-draw_line[i][0]),(draw_line[i+1][1]-draw_line[i][1])])
        sin_a = left_v.cross(right_v)/(left_v.magnitude()*right_v.magnitude())
        if sin_a > 0.3 :         
            array_2 = left_v.normalized().plus(right_v.normalized()).times_scalar(distant/sin_a).convert_array()
            new_coodinate = [[round(draw_line[i][0] + array_2[0]),round(draw_line[i][1] + array_2[1])],[draw_line[i][0],draw_line[i][1]]]
            if i == 0 :
                draw_line_shrink.append(new_coodinate[0])
                draw_line_shrink_oringin.append(new_coodinate[1])
            else:
                signal = 0
                for k in range(0,len(draw_line_shrink)):
                    compare_coodinate = [[draw_line_shrink[k][0],draw_line_shrink[k][1]],[draw_line_shrink_oringin[k][0],draw_line_shrink_oringin[k][1]]]
                    if  doIntersect(Point(new_coodinate[0][0],new_coodinate[0][1]), Point(new_coodinate[1][0],new_coodinate[1][1]), Point(compare_coodinate[0][0],compare_coodinate[0][1]), Point(compare_coodinate[1][0],compare_coodinate[1][1])) :
                        signal = 1
                if signal == 0:
                    draw_line_shrink.append(new_coodinate[0])
                    draw_line_shrink_oringin.append(new_coodinate[1])
    if len(draw_line_shrink) >= 3:
        draw_line_shrink = Interpolation(draw_line_shrink,num)
        draw_line_shrink = np.array(draw_line_shrink, dtype=np.int32)
        draw_line_shrink = get_convex([draw_line_shrink])
    draw_line_shrink = remove_same_list(draw_line_shrink)
    draw_line_shrink = np.array(draw_line_shrink, dtype=np.int32)
    return draw_line_shrink
           
        
            
def extract_local_histogram(A, i, j, size):
    c = 0
    RGBs = np.zeros((size**2,3))
    gvs = np.zeros(size**2)
    lc_RGBs = np.zeros((size,size,3))
    half_size=(size-1)//2
    for k in range(-half_size, (half_size+1)):
        for l in range(-half_size, (half_size+1)):
            lc_RGBs[half_size+k, half_size+l,:] = A[i+k,j+l,:]
            RGBs[c,:] = A[i+k,j+l,:]
#               print( RGBs[c,:])
            gvs[c] = np.mean(RGBs[c,:])
            c += 1
    return gvs,lc_RGBs

def  iterate_image(A, size): 
    m,n,_ = A.shape
    B = np.zeros((size**2,m,n,3))
    for i in range((size-1)/2, (m-(size-1)/2)):
        for j in range((size-1)/2, (n-(size-1)/2)):
            extract_local_histogram(A, i, j, size)

def make_histogram(hist_array,num_color,func):
    # data histogram
    bin_num=256
    prob, bin_s = np.histogram(hist_array, bins =bin_num)
    
    # trim data
    #x = np.linspace(np.min(hist_array), np.max(hist_array), num=256)
    
    # find index of minimum between two modes
    #ind_max = argrelmax(prob)
    #x_max = x[ind_max]
    #y_max = prob[ind_max]
    #plt.hist(hist_array.ravel(), bins=256)
    #plt.scatter(x_max, y_max, color='b')
    #plt.show()
    
    local_size = num_color
    hist_1=smooth(prob,bin_s,bin_num,func, local_size)
    bar = hist_1.first
    n = 0
    probs = np.zeros(local_size)
    bins = np.zeros(local_size+1)
    #bins1 = np.zeros(local_size)
    while bar.next_bar is not None:
        bins[n] = bar.min_bar
        #bins1[n] = round(bar.min_bar)
        probs[n] = bar.prob
        n = n + 1
        bar = bar.next_bar
    
    bins[n] = bar.min_bar
    #bins1[n] = round(bar.min_bar)
    bins[n+1] = bar.max_bar
    probs[n] = bar.prob
        # plot
    '''plt.bar(bins1[:],probs)
    plt.xticks(bins[:]) 
    plt.xlim(min(bins), max(bins))
    plt.show()'''
    return probs,bins
    
def openopration(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresholdImg = cv.threshold(gray, 170, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(9, 9))
    iClose = cv.morphologyEx(thresholdImg, cv.MORPH_OPEN, kernel)
    return iClose

class Interval:
    prob = 0.0
    min_bar = 0
    max_bar = 255
    m_dist = 0
    q_loc = {}
    def _init_(self, previous_bar=None, next_bar=None):
        self.prev=previous_bar
        self.next_bar=next_bar
        
        
class Distribution:
    size = 0
    def __init__(self):
        first = Interval()
        last = Interval()
        self.first = first
        self.last = last
        self.first.next_bar = self.last
        self.last.prev = self.first
    
def initial_hist(initial_prob, bin_s, m ):
    hist = Distribution()
    hist.first.prob = initial_prob[0]
    hist.first.min_bar = bin_s[0]
    hist.first.max_bar = bin_s[1]
    hist.first.prev = None
    hist.last = hist.first
    for i in range (1,m):
        #b = hist.first.max_bar
        bar = Interval()
        bar.prob = initial_prob[i]
        bar.min_bar = bin_s[i]
        bar.max_bar = bin_s[i+1]
        bar.prev = hist.last
        hist.last.next_bar = bar
        hist.last = bar
    hist.last.next_bar = None
    hist.size = m
    return hist

def merged_bars(bar):
    new_bar = Interval()
    new_bar.prob = bar.prob + bar.next_bar.prob
    new_bar.min_bar = bar.min_bar
    new_bar.max_bar = bar.next_bar.max_bar
    new_bar.prev = bar.prev
    new_bar.next_bar = bar.next_bar.next_bar
    return new_bar

def merging(hist, dist_func, queue):
    bar=queue.pop()    
    new_bar = merged_bars(bar)
    if hist.first == bar:
        hist.first = new_bar
    else:
        queue.remove(bar.prev.q_loc)
        bar.prev.next_bar = new_bar
        bar.prev.m_dist = merge_dist(bar.prev,dist_func) 
        bar.prev.q_loc=queue.append(bar.prev,bar.prev.m_dist)
    if hist.last == bar.next_bar:
        hist.last = new_bar
    else:
        bar.next_bar.next_bar.prev = new_bar
        queue.remove(bar.next_bar.q_loc)
        new_bar.m_dist = merge_dist(new_bar,dist_func)
        b = new_bar.m_dist
        new_bar.q_loc = queue.append(new_bar,new_bar.m_dist)
    hist.size = hist.size-1
    return hist,queue

def merge_dist(bar,dist_func):
    if dist_func == "simple_merge":
        return simple_merge_dist(bar)
    elif dist_func == "kl_merge":
        return kl_merge_dist(bar)
    elif dist_func == "js_merge":
        return js_merge_dist(bar)

def initial_queue(hist,dist_func):
    hist_queue=MinHeapPriorityQueue()
    bar = Interval()
    bar = hist.first
    while bar.next_bar is not None:
        bar.m_dist = merge_dist(bar,dist_func)
        bar.q_loc=hist_queue.append(bar,bar.m_dist)
        bar = bar.next_bar
    return hist_queue

def height(bar):
    return bar.prob/(bar.max_bar-bar.min_bar)

def kl_merge_dist_rel(bar):
    new_bar = merged_bars(bar)
    if height(bar.next_bar) == 0 and height(bar) == 0 :
        return 0
    elif height(bar.next_bar) != 0 and height(bar) != 0:
        dist = height(bar)*log(height(bar)/height(new_bar))*(bar.max_bar-bar.min_bar) +\
        height(bar.next_bar)*log(height(bar.next_bar)/height(new_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return dist 
    elif height(bar.next_bar) == 0:
        dist = height(bar)*log(height(bar)/height(new_bar))*(bar.max_bar-bar.min_bar)
        return 0
    else:
        dist = height(bar.next_bar)*log(height(bar.next_bar)/height(new_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return 0        
    
def kl_merge_dist(bar):
    new_bar = merged_bars(bar)
    if height(bar.next_bar) != 0 and height(bar) != 0:
        dist = height(bar)*log(height(bar)/height(new_bar))*(bar.max_bar-bar.min_bar) +\
        height(bar.next_bar)*log(height(bar.next_bar)/height(new_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return dist 
    else:
        return 0     

def simple_merge_dist(bar):
    new_bar = merged_bars(bar)
    dist = abs(height(new_bar)-height(bar))*(bar.max_bar-bar.min_bar) +\
    height(new_bar)*(bar.next_bar.min_bar-bar.max_bar) +\
    abs(height(new_bar)-height(bar.next_bar))*(bar.next_bar.max_bar-bar.next_bar.min_bar)
    return dist

def js_merge_dist(bar):
    new_bar = merged_bars(bar)
    left_mean_height = (height(bar) + height(new_bar))/2
    right_mean_height = (height(bar.next_bar) + height(new_bar))/2
    if height(bar.next_bar) == 0 or height(bar) == 0 :
        return 0
    else:
        kl_dist1 = height(bar)*log(height(bar)/left_mean_height)*(bar.max_bar-bar.min_bar) +\
        height(bar.next_bar)*log(height(bar.next_bar)/right_mean_height)*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        kl_dist2 = height(new_bar)*log(height(new_bar)/left_mean_height)*(bar.max_bar-bar.min_bar) +\
        +bar.next_bar.min_bar-bar.max_bar+log(height(new_bar)/right_mean_height)*(bar.next_bar.max_bar-bar.next_bar.min_bar)
        return (kl_dist1 + kl_dist2)/2
    
    
def smooth(initial_prob,bin_s, m, dist_func,k):
    hist = initial_hist(initial_prob, bin_s, m)
    queue = initial_queue(hist, dist_func)
    while hist.size > k :
        hist,queue = merging(hist,dist_func,queue)
    return hist

def back_projection(probs, bins, num_color, h_size, local_photo):
    probs_list = sorted(probs)
    second_max = probs_list[-1]
    for m in range (0,num_color):
        if probs[m] == second_max:
            second_max_left = bins[m]
            second_max_right = bins[m+1]
    half_size=(h_size-1)//2
    grey_photo=np.zeros((h_size,h_size))
    for k in range(-half_size, (half_size+1)):
        for l in range(-half_size, (half_size+1)):
            mean = np.mean(local_photo[half_size+k,half_size+l,:])
            if  mean > second_max_left  and mean < second_max_right :
                grey_photo[half_size+k,half_size+l]=255
            else:
                grey_photo[half_size+k,half_size+l]=0
    return grey_photo

def get_contur(image):   
    close_image = openopration(image)
    contours, hierarchy = cv.findContours(close_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if  len(contours) == 0:
        center = None
        draw_line = None
    else:
        for c in range(len(contours)):
        # 是否为凸包
            ret = cv.isContourConvex(contours[c])
            # 凸包检测
            points = cv.convexHull(contours[c])
            total = len(points)
            draw_line = np.zeros((total,2)).astype(np.int)
            for i in range(len(points)):
                draw_line[i][:]=points[i % total]
                center = centroid(draw_line)
    return center,draw_line

def draw_common(local_photo,max_color,sencond_color,third_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size):
    m,n,_ = local_photo.shape
    if m > n:
        h_size = n
    else:
        h_size = m
    cr.translate(height_k,weight_l)
    half_size=(h_size-1)//2
    t_size = (h_size-1)// 8
    t_size_n_r = n - t_size
    t_size_m_r = m - t_size
    half_size_n_short = n//2 - 1
    half_size_n_long = n//2 + 1
    half_size_m_short = m//2 - 1
    half_size_m_long = m//2 + 1 
    for k in range(0, m):
        for l in range(0, n):
            local_photo[k,l,:] = max_color[:]
    max_color[:] = max_color[:]/255
    cr.set_source_rgb(max_color[0], max_color[1],max_color[2])
    cr.new_path()
    cr.move_to(0, 0)
    cr.line_to(n*beilv, 0)
    cr.line_to(n*beilv, m*beilv)
    cr.line_to(0, m*beilv)
    cr.close_path()
    cr.fill_preserve()
    '''rand = random.randint(0,3)
    if rand == 0:
        draw_line = np.array([[t_size,t_size],[t_size,random.randint(half_size,t_size_r)],\
                               [t_size_r,t_size_r],[random.randint(half_size,t_size_r),t_size-1]])    
        
    elif rand == 1:
        draw_line = np.array([[t_size,t_size_r],[t_size,random.randint(t_size,half_size)],\
                               [t_size_r,t_size],[random.randint(half_size,t_size_r),t_size_r]])    
    
    elif rand == 2:
        draw_line = np.array([[t_size_r,t_size],[random.randint(t_size,half_size),t_size],\
                               [t_size,t_size_r],[t_size_r,random.randint(half_size,t_size_r)]])
    else :
        draw_line = np.array([[t_size,t_size],[t_size_r,random.randint(t_size,half_size)],\
                               [t_size_r,t_size_r],[random.randint(t_size,half_size),t_size_r]])'''
    draw_line = np.array([[t_size,random.randint(t_size+1,half_size_m_short)],[t_size,random.randint(half_size_m_long,t_size_m_r-1)],\
                           [random.randint(t_size+1,half_size_n_short),t_size_m_r],[random.randint(half_size_n_long,t_size_n_r-1),t_size_m_r],\
                               [t_size_n_r,random.randint(half_size_m_long,t_size_m_r-1)],[t_size_n_r,random.randint(t_size+1,half_size_m_short)],\
                               [random.randint(half_size_n_long,t_size_n_r-1),t_size],[random.randint(t_size+1,half_size_n_short),t_size]])
    draw_line[:] = draw_line[:]*beilv
    draw_line = Interpolation(draw_line,21)
    draw_line = remove_same_list(draw_line)
    draw_line = np.array(draw_line, dtype=np.int32)
    draw_line = get_convex([draw_line])     
    long = compute_long(draw_line)//2.6
    draw_line_second = shrink_polygon(draw_line,long,51)
    short = compute_long(draw_line_second)//2
    draw_line_third = shrink_polygon(draw_line_second,short,17)
    svg_draw(draw_line,sencond_color,svg_hist_size//15,cr)
    svg_draw(draw_line_second,third_color,long/3,cr)
    #cv.fillPoly(local_photo, [draw_line],(sencond_color[0], sencond_color[1],sencond_color[2]))
    #cv.fillPoly(local_photo, [draw_line_second],(third_color[0], third_color[1],third_color[2]))
    draw = 0
    if draw_line_third.shape[0] > 2 and compute_polygon_area(draw_line_second) > (n*m*beilv*beilv//6) :
        svg_draw(draw_line_third,fourth_color,short/3,cr)
    return local_photo

def find_background_color(local_photo,h_size):
    half_size=(h_size-1)//2
    max_color = np.zeros((3))
    sencond_color = np.zeros((3))
    third_color = np.zeros((3))
    fourth_color = np.zeros((3))
    a = 0
    for k in range(-half_size, (half_size+1)):
        for l in range(-half_size, (half_size+1)):
                max_color[:] = local_photo[half_size+k,half_size+l,:] + max_color[:]
                a = a+1
    rand = random.randint(10,20)
    switch = random.randint(0,2)
    switch_color = np.array([[rand,0,0],[0,rand,0],[0,0,rand]])
    switch_2 = random.randint(0,1)
    for i in range(0, 3):
        max_color[i] = max_color[i]//a
        if max_color[i] < 20:
            switch_color[i] = -switch_color[i]

    sencond_color[:] = max_color[:] - switch_color[switch]       
    third_color[:] =  max_color[:] - switch_color[switch-switch_2-1]
    fourth_color[:] =  max_color[:] - switch_color[switch-2+switch_2]
        
    return max_color,sencond_color,third_color,fourth_color

def compute_polygon_area(points):
    point_num = len(points)
    if(point_num < 3): 
        return 0.0
    s = points[0][1] * (points[point_num-1][0] - points[1][0])
    for i in range(1, point_num):
        s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
    return abs(s/2.0)

def compute_long(draw_line):
    center = centroid(draw_line)
    a = 0 
    for i in range(0,draw_line.shape[0]):
        a = a + sqrt((draw_line[0][0] - center[0])**2 + (draw_line[0][1] - center[1])**2)
    a = a // draw_line.shape[0]
    return a
   
def draw_hist(probs,bins,local_photo,num_color,center,draw_line,weight_l,height_k,cr,beilv,svg_hist_size):
    probs_list = sorted(probs)
    for m in range (0,num_color):
        if probs[m] == probs_list[-1]:
            max_left = bins[m]
            max_right = bins[m+1]
            probs_list[-1] = 0
        elif probs[m] == probs_list[-2]:
            sencond_max_left = bins[m]
            sencond_max_right = bins[m+1] 
            probs_list[-2] = 0
        elif probs[m] == probs_list[-3]:
            third_max_left = bins[m]
            third_max_right = bins[m+1]
            probs_list[-3] = 0
        elif probs[m] == probs_list[-4]:
            fourth_max_left = bins[m]
            fourth_max_right = bins[m+1]
            probs_list[-4] = 0
    m,n,_ = local_photo.shape
    if m > n:
        weight_l = weight_l - svg_hist_size
    elif n > m:
        height_k = height_k - svg_hist_size
    max_color = np.zeros((3))
    sencond_color = np.zeros((3))
    third_color = np.zeros((3))
    fourth_color = np.zeros((3))
    a=0
    b=0
    c=0
    d=0
    for k in range(0, m):
        for l in range(0, n):
            mean = np.mean(local_photo[k,l,:])            
            if  mean < max_right and mean > max_left :
                max_color[:] = local_photo[k,l,:] + max_color[:]
                a = a+1
            elif  mean < sencond_max_right and mean > sencond_max_left:
                sencond_color[:] = local_photo[k,l,:] + sencond_color[:]
                b = b+1                
            elif  mean < third_max_right and mean > third_max_left:
                third_color[:] = local_photo[k,l,:] + third_color[:]
                c = c+1               
            elif  mean < fourth_max_right and mean > fourth_max_left:
                fourth_color[:] = local_photo[k,l,:] + fourth_color[:]
                d = d+1
    for i in range(0, 3):
        max_color[i] = max_color[i]//a
        if b > 0:
            sencond_color[i] = sencond_color[i]//b
        else:
            sencond_color[i] = max_color[i]
        if c > 0:
            third_color[i] = third_color[i]//c
        else: 
            third_color[i] = sencond_color[i]
        if d > 0:
            fourth_color[i] = fourth_color[i]//d
        else:
            fourth_color[i] = third_color[i]
    #pts = np.vstack((draw_line[:][0],draw_line[:][1])).astype(np.int32).T
    '''rect = np.zeros((num_color-1))
    rect[0] = probs_list[-1]/(probs_list[-1]+probs_list[-3]+probs_list[-4])
    rect[1] = probs_list[-3]/(probs_list[-1]+probs_list[-3]+probs_list[-4])
    rect[2] = probs_list[-4]/(probs_list[-1]+probs_list[-3]+probs_list[-4]) 
    distant = np.zeros((3,draw_line.shape[0]))'''
    long = compute_long(draw_line)//2.6
    draw_line_second = shrink_polygon(draw_line,long,17)
    short = compute_long(draw_line_second)//2
    if draw_line_second.shape[0] < 3 or compute_polygon_area(draw_line) < (n*m*beilv*beilv//5):
        local_photo = draw_common(local_photo,max_color,third_color,sencond_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size)                
    else:
        sencond_color[:]=sencond_color[:]/255
        cr.translate(height_k,weight_l)
        cr.set_source_rgb(sencond_color[0], sencond_color[1],sencond_color[2])
        cr.new_path()
        cr.move_to(0, 0)
        cr.line_to(n*beilv, 0)
        cr.line_to(n*beilv, m*beilv)
        cr.line_to(0, m*beilv)
        cr.close_path()
        cr.fill_preserve()
        draw_line_third = shrink_polygon(draw_line_second,short,17)
        svg_draw(draw_line,max_color,svg_hist_size/15,cr)
        svg_draw(draw_line_second,third_color,long/3,cr)
        if draw_line_third.shape[0] > 2 and compute_polygon_area(draw_line_second) > (n*m*beilv*beilv//6) :
            svg_draw(draw_line_third,fourth_color,short/3,cr)
            #cv.fillPoly(local_photo, [draw_line_third],(fourth_color[0], fourth_color[1],fourth_color[2]))
            '''for point in draw_line_third:
                cv.circle(local_photo, (point[0],point[1]), 1, (255, 0, 0) , 2)
            for point in draw_line_second:
                cv.circle(local_photo, (point[0],point[1]), 1, (0, 255, 0) , 2)
            cv.circle(local_photo, (int(center[0]),int(center[1])), 1, (0, 255, 0) , 2)'''
        i = 0     
        '''while i < 17 :
            if i < draw_line_third.shape[0] and draw == 1:
                cv.line(local_photo, (draw_line_third[i-1][0],draw_line_third[i-1][1]), (draw_line_third[i][0],draw_line_third[i][1]), (fourth_color[0], fourth_color[1],fourth_color[2]), 2, 8, 0)
            if i < draw_line_second.shape[0]:
                cv.line(local_photo, (draw_line_second[i-1][0],draw_line_second[i-1][1]), (draw_line_second[i][0],draw_line_second[i][1]), (third_color[0], third_color[1],third_color[2]), int(short//2+1), 8, 0)
            if i < draw_line.shape[0]:
                cv.line(local_photo, (draw_line[i-1][0],draw_line[i-1][1]), (draw_line[i][0],draw_line[i][1]), (max_color[0], max_color[1],max_color[2]), int(long//2+1), 8, 0)
            i=i+1'''
            
        
    
    '''for j in range(0, draw_line.shape[0]):
        distant[0][j] = math.sqrt((draw_line[j][0] - center[0])**2 + (draw_line[j][1] - center[1])**2)*rect[0]
        distant[1][j] = math.sqrt((draw_line[j][0] - center[0])**2 + (draw_line[j][1] - center[1])**2)*rect[1]
        distant[2][j] = math.sqrt((draw_line[j][0] - center[0])**2 + (draw_line[j][1] - center[1])**2)*rect[2]
        cv.circle(local_photo, (draw_line[j][0], draw_line[j][1]), 1,(0, 0, 255), -1)
    for j in range(0, draw_line_second.shape[0]):
        cv.circle(local_photo, (draw_line_second[j][0], draw_line_second[j][1]), 1, (0, 255, 0), -1)
    #draw_line_third[j][0] = int(draw_line_second[j][0] - ((draw_line_second[j][0] - center[0])*rect[1]))
    #draw_line_third[j][1] = int(draw_line_second[j][0] - ((draw_line_second[j][0] - center[0])*rect[1]))'''
    
    #cv.circle(local_photo, (int(center[0]), int(center[1])), 5,(fourth_color[0][0], fourth_color[0][1],fourth_color[0][2]), -1)
    #cv.fillPoly(local_photo, [draw_line],(max_color[0][0], max_color[0][1],max_color[0][2]))
    #cv.fillPoly(local_photo, [draw_line_second],(third_color[0][0], third_color[0][1],third_color[0][2]))
    #cv.fillPoly(local_photo, [draw_line_third],(fourth_color[0][0], fourth_color[0][1],fourth_color[0][2]))
    #cv.fillPoly(local_photo, [draw_line],True,(third_color[0][0], third_color[0][1],third_color[0][2]),mean(distant[1][:]))
    #cv.fillPoly(local_photo, [draw_line_third],(fourth_color[0][0], fourth_color[0][1],fourth_color[0][2]))
    
    '''for r in range(0,draw_line.shape[0]-1):
        cv.line(local_photo, (int(draw_line[r][0]), int(draw_line[r][1])), (int(draw_line[r+1][0]), int(draw_line[r+1][1])), (max_color[0][0], max_color[0][1],max_color[0][2]), 6, 8, 0)
        second_l = int(draw_line[r][0] - (draw_line[r][0]-center[0])*rect[2])
        second_h = int(draw_line[r][1] - (draw_line[r][1]-center[1])*rect[2])
        second2_l = int(draw_line[r+1][0] - (draw_line[r+1][0]-center[0])*rect[2])
        second2_h = int(draw_line[r+1][1] - (draw_line[r+1][1]-center[1])*rect[2])
        cv.line(local_photo, (second_l, second_h), (second2_l,second2_h), (third_color[0][0], third_color[0][1],third_color[0][2]), 3, 8, 0)
    cv.line(local_photo, (int(draw_line[r+1][0]), int(draw_line[r+1][1])), (int(draw_line[0][0]), int(draw_line[0][1])), (max_color[0][0], max_color[0][1],max_color[0][2]), 6, 8, 0)'''
    return local_photo

def centroid(vertexes):
     _x_list = [vertex [0] for vertex in vertexes]
     _y_list = [vertex [1] for vertex in vertexes]
     _len = len(vertexes)
     _x = sum(_x_list) / _len
     _y = sum(_y_list) / _len
     return(_x, _y)

def dist_match(bins,bins_b,probs,probs_b,num_color,hist_size):
    dist = 0
    for i in range(0,num_color):
        dist = sqrt((bins[i]-bins_b[i])**2+ (bins[i+1]-bins_b[i+1])**2)*sqrt(abs(probs[i]-probs_b[i])/hist_size**2) +dist
    return dist 
    
def match_local(A,i,j,hist_size,func,num_color,threshold,match_flag):
    flag = 0
    local_histogram, local_photo = extract_local_histogram(A, i, j, hist_size)
    local_histogram_j, local_photo_j = extract_local_histogram(A, i, j-hist_size, hist_size)
    local_histogram_i, local_photo_i = extract_local_histogram(A, i-hist_size, j, hist_size)
    probs,bins = make_histogram(local_histogram,num_color,func)
    probs_j,bins_j = make_histogram(local_histogram_j,num_color,func)
    probs_i,bins_i = make_histogram(local_histogram_i,num_color,func)
    grey_photo = back_projection(probs, bins, num_color, hist_size, local_photo)
    if dist_match(bins,bins_j,probs,probs_j,num_color,hist_size) < threshold and match_flag[i][j-hist_size] == 0:
        flag = 1
        grey_photo_j = back_projection(probs, bins, num_color, hist_size, local_photo_j)
        B = np.zeros((hist_size,2*hist_size))
        local_photo_match = np.zeros((hist_size,2*hist_size,3))  
        for k in range(0, hist_size):
            for l in range(0, 2*hist_size):
                if l < hist_size:
                    local_photo_match[k][l] = local_photo_j[k][l]
                    B[k][l] = grey_photo_j[k][l]
                else:
                    local_photo_match[k][l] = local_photo[k][-hist_size+l]
                    B[k][l] = grey_photo[k][-hist_size+l]
        match_flag[i][j] = 1   
    elif dist_match(bins,bins_i,probs,probs_i,num_color,hist_size) < threshold and match_flag[i-hist_size][j] == 0:
        flag = 2
        grey_photo_i = back_projection(probs, bins, num_color, hist_size, local_photo_i)
        B = np.zeros((2*hist_size,hist_size))
        local_photo_match = np.zeros((2*hist_size,hist_size,3))   
        for k in range(0, 2*hist_size):
            for l in range(0, hist_size):
                if k < hist_size:
                    local_photo_match[k][l] = local_photo_i[k][l]
                    B[k][l] = grey_photo_i[k][l]
                else:
                    local_photo_match[k][l] = local_photo[-hist_size+k][l]
                    B[k][l] = grey_photo[-hist_size+k][l]
        match_flag[i][j] = 1    
    else:
        B = grey_photo     
        local_photo_match =  local_photo  
        match_flag[i][j] = 0
    return B,local_photo_match,flag,match_flag
          
def svg_draw(draw_line,max_color,svg_hist_size,cr):
    max_color[:]=max_color[:]/255
    cr.new_path()
    cr.move_to(draw_line[0][0],draw_line[0][1])
    for draw_num in range(1, draw_line.shape[0]-2,2):
        cr.curve_to(draw_line[draw_num][0],draw_line[draw_num][1],draw_line[draw_num+1][0],draw_line[draw_num+1][1],draw_line[draw_num+2][0],draw_line[draw_num+2][1])
    cr.close_path()
    cr.set_source_rgb(max_color[0], max_color[1],max_color[2])
    cr.fill_preserve()
    cr.set_line_width(svg_hist_size)
    cr.set_tolerance(0.1)
    cr.stroke_preserve()
    
    
def main(): 
    this_file = __file__
    spfile = this_file.split('.')           
    image1 = Image.open("lena.bmp")
    im1 = image1.convert('RGB')
    A = np.asarray(im1)
    m,n,_ = A.shape
    hist_size = 17
    num_color = 4
    threshold = 8
    color_size = 30
    func = "kl_merge"
    half_size = (hist_size - 1)//2
    B = np.zeros((m,n,3))
    match_flag = np.zeros((m,n))
    width_in_inches, height_in_inches = 1, m/n
    width_in_points, height_in_points = \
    width_in_inches * 3000, height_in_inches * 3000
    width, height = width_in_points, height_in_points
    svg_hist_size = hist_size*3000/n
    beilv = svg_hist_size/hist_size
    filename = "nest1.svg"
    surface = cairo.SVGSurface(filename, width, height)
    cr = cairo.Context(surface)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    weight_l = int(0)
    height_k = int(0)
    i = int(half_size)
    while i < (m-half_size):
        j = int(half_size)
        height_k = int(0)
        while j < (n-half_size):
             local_histogram, local_photo = extract_local_histogram(A, i, j, hist_size)
             probs,bins = make_histogram(local_histogram,num_color,func)
             if (bins[4] - bins[0]) < color_size :
                 max_color,second_color,third_color,fourth_color = find_background_color(local_photo,hist_size)
                 cr.save()
                 local_photo1 = draw_common(local_photo,max_color,third_color,second_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size)  
                 cr.restore()
                 h = 0
                 w = 0
             else:
                 grey_photo,local_photo_m,flag,match_flag = match_local(A,i,j,hist_size,func,num_color,threshold,match_flag)
                 output_img = Image.fromarray(np.uint8(grey_photo))
                 output_img.save( "test_grey_local.bmp")
                 image = cv.imread('test_grey_local.bmp')
                 center,draw_line = get_contur(image)
                 if draw_line is None :
                     max_color,second_color,third_color,fourth_color = find_background_color(local_photo,hist_size)
                     cr.save()
                     local_photo1 = draw_common(local_photo,max_color,third_color,second_color,fourth_color,cr,weight_l,height_k,beilv,svg_hist_size)  
                     cr.restore()
                     h = 0
                     w = 0
                 else:
                     draw_line[:] = draw_line[:]*beilv
                     draw_line = Interpolation(draw_line,71)
                     draw_line = remove_same_list(draw_line)
                     draw_line = np.array(draw_line, dtype=np.int32)
                     draw_line = get_convex([draw_line])                   
                     cr.save()
                     local_photo1 = draw_hist(probs,bins,local_photo_m,num_color,center,draw_line,weight_l,height_k,cr,beilv,svg_hist_size) 
                     cr.restore()
                     if flag == 0:                     
                         h = 0
                         w = 0
                     else:  
                         if flag == 1:
                             h = hist_size
                             w = 0
                         else:
                             h = 0
                             w = hist_size                   
             for k in range(-half_size-w,half_size+1):
                 for l in range(-half_size-h,half_size+1):
                     B[i+k][j+l][:] = local_photo1[half_size+k+w,half_size+l+h,:] 
             j = j + hist_size
             height_k = height_k + svg_hist_size
        i = i + hist_size
        weight_l = weight_l + svg_hist_size
    cr.show_page()
    surface.finish()
             
    '''local_histogram, local_photo = extract_local_histogram(A, 100, 100, hist_size)
    output_img = Image.fromarray(np.uint8(local_photo))
    output_img.save(spfile[0] + "_local.png")
    probs,bins,local_size = make_histogram(local_histogram,num_color)
    grey_photo = back_projection(probs, bins, local_size, hist_size, local_photo)
    output_img = Image.fromarray(np.uint8(grey_photo))
    output_img.save("test_grey_local.png")
    image = cv.imread('test_grey_local.png')
    center,draw_line = get_contur(image)
    local_photo1 = draw_hist(hist_size,probs,bins,local_photo,num_color,center,draw_line)'''
    output_img = Image.fromarray(np.uint8(B))
    output_img.save(spfile[0] + "test_grey_CLOSE_local.png")
if __name__ == '__main__':
    main()
